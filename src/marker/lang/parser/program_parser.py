"""Top-level program parsing orchestration."""
from __future__ import annotations

from typing import List
from ..ast import (
    Program, ProcedureDef, Stmt, TemplateLine,
    PythonBlock, BacktickLineStmt, TypeDefStmt, ProcDefStmt,
    ReturnStmt, ReturnBlockStmt, AssignStmt, AssignBlockStmt
)
from .preprocessing import strip_comments, merge_line_continuations, preprocess_blocks, BEGIN, END
from .segment_parser import parse_segments
from .statement_parser import StatementParser
from .block_parser import BlockParser


def parse_program(source: str) -> Program:
    """Parse Marker DSL source code into a Program AST.
    
    This is the main entry point for the parser.
    """
    parser = ProgramParser(source)
    return parser.parse()


class ProgramParser:
    """Orchestrates program parsing."""
    
    def __init__(self, source: str):
        # Preprocess source
        source_no_comments = strip_comments(source)
        merged_src = merge_line_continuations(source_no_comments)
        self.lines = preprocess_blocks(merged_src).splitlines()
        
        # Initialize parsers
        self.stmt_parser = StatementParser()
        self.block_parser = BlockParser()
        
        # State
        self.program = Program()
        self.proc_stack: List[ProcedureDef] = []
        self.current_body: List[Stmt] | None = None
        self.tests: list = []
        self.evals: list = []
        self.i = 0  # Current line index
    
    def parse(self) -> Program:
        """Parse the entire program."""
        while self.i < len(self.lines):
            line = self.lines[self.i]
            stripped = line.strip()
            
            if stripped == "":
                self.i += 1
                continue
            
            if stripped == BEGIN:
                self.i += 1
                continue
            
            if stripped == END:
                self._handle_end_marker()
                self.i += 1
                continue
            
            # Validation blocks
            if stripped.startswith("@test:") or stripped.startswith("@eval:"):
                self._parse_validation_block(stripped)
                continue
            
            # Procedure header - validate format if line starts with @
            if stripped.startswith("@"):
                if not stripped.endswith(":"):
                    raise ValueError("Procedure header must end with ':'")
                self._parse_procedure_header(stripped)
                self.i += 1
                continue
            
            # Type definition
            if self.stmt_parser.is_type_definition(stripped):
                typedef = self.stmt_parser.parse_type_definition(stripped)
                if self.current_body is None:
                    self.program.type_defs.append(typedef)
                else:
                    self.current_body.append(TypeDefStmt(typedef=typedef))
                self.i += 1
                continue
            
            # Python block
            if stripped == "```":
                self._parse_python_block()
                continue
            
            # Single-line backtick
            if (stripped.startswith("`") and stripped.endswith("`") and
                len(stripped) >= 2 and stripped.count("`") == 2):
                code = stripped[1:-1]
                stmt = BacktickLineStmt(code=code)
                if self.current_body is None:
                    self.program.toplevel_body.append(stmt)
                else:
                    self.current_body.append(stmt)
                self.i += 1
                continue
            
            # Assignment
            if self.stmt_parser.is_assignment(stripped):
                self._parse_assignment(stripped)
                continue
            
            # Return
            if self.stmt_parser.is_return(stripped):
                self._parse_return(line)
                continue
            
            # AI prompt lines (inside procedure body)
            if stripped.startswith(">") and self.current_body is not None and self.proc_stack:
                self._parse_ai_prompt()
                continue
            
            # Template line
            self._parse_template_line(stripped)
        
        # Close any open procedures
        while self.proc_stack:
            self.program.procedures.append(self.proc_stack.pop())
        
        return self.program
    
    def _handle_end_marker(self) -> None:
        """Handle END marker - closes current procedure or block."""
        if self.proc_stack:
            proc = self.proc_stack.pop()
            if not self.proc_stack:
                self.program.procedures.append(proc)
                self.current_body = None
            else:
                # Nested procedure
                parent = self.proc_stack[-1]
                parent.body.append(ProcDefStmt(proc=proc))
                self.current_body = parent.body
    
    def _parse_validation_block(self, stripped: str) -> None:
        """Parse @test: or @eval: validation block."""
        suite, next_i = self.block_parser.parse_validation_block(
            self.lines, self.i + 1, stripped
        )
        if stripped.startswith("@test:"):
            self.tests.append(suite)
        else:
            self.evals.append(suite)
        
        setattr(self.program, "_validation_tests", self.tests)
        setattr(self.program, "_validation_evals", self.evals)
        self.i = next_i
    
    def _parse_procedure_header(self, stripped: str) -> None:
        """Parse procedure header and setup procedure."""
        name, params, param_annos, ret_anno = self.stmt_parser.parse_procedure_header(stripped)
        
        proc = ProcedureDef(
            name=name,
            params=params,
            return_type_annotation=ret_anno,
            param_type_annotations=param_annos
        )
        
        self.proc_stack.append(proc)
        self.current_body = proc.body
    
    def _parse_python_block(self) -> None:
        """Parse Python block (``` ... ```)."""
        block, next_i = self.block_parser.parse_python_block(self.lines, self.i)
        
        # Check if this is the prelude (first python block before any procedures/content)
        if (self.program.prelude is None and
            not self.program.toplevel_body and
            self.program.toplevel_return is None and
            not self.program.procedures and
            not self.proc_stack):  # Not inside a procedure being parsed
            self.program.prelude = block.code
        elif self.current_body is None:
            self.program.toplevel_body.append(block)
        else:
            self.current_body.append(block)
        
        self.i = next_i
    
    def _parse_assignment(self, stripped: str) -> None:
        """Parse assignment statement."""
        # Check for block assignment
        if stripped.endswith("= ```"):
            # Extract name and type annotation
            from .segment_parser import find_matching_delimiter
            close_br = find_matching_delimiter(stripped, 0, "[", "]")
            target_inner = stripped[1:close_br].strip()
            
            type_anno: str | None = None
            if ":" in target_inner:
                name_part, type_part = target_inner.split(":", 1)
                name = name_part.strip()
                type_anno = type_part.strip()
            else:
                name = target_inner
            
            stmt, next_i = self.block_parser.parse_assignment_block(
                self.lines, self.i + 1, name, type_anno
            )
            if self.current_body is None:
                self.program.toplevel_body.append(stmt)
            else:
                self.current_body.append(stmt)
            self.i = next_i
        else:
            # Regular assignment
            stmt = self.stmt_parser.parse_assignment(stripped)
            if stmt:
                if self.current_body is None:
                    self.program.toplevel_body.append(stmt)
                else:
                    self.current_body.append(stmt)
            self.i += 1
    
    def _parse_return(self, line: str) -> None:
        """Parse return statement."""
        # Strip only the leading '=' and any leading whitespace after it
        expr_src = line.lstrip()[1:]
        
        # Inline backtick: = `expr`
        expr_trim = expr_src.strip()
        if (expr_trim.startswith("`") and expr_trim.endswith("`") and
            len(expr_trim) >= 2 and expr_trim.count("`") == 2):
            code = expr_trim[1:-1]
            ret_block = ReturnBlockStmt(code=code)
            if self.current_body is None:
                self.program.toplevel_return = ret_block
            else:
                self.current_body.append(ret_block)
            self.i += 1
            return
        
        # Block return: = ```
        if expr_src.strip() == "```":
            ret_block, next_i = self.block_parser.parse_return_block(self.lines, self.i + 1)
            if self.current_body is None:
                self.program.toplevel_return = ret_block
            else:
                self.current_body.append(ret_block)
            self.i = next_i
            return
        
        # Regular return - preserve whitespace by not stripping the expression
        ret_stmt = self.stmt_parser.parse_return(line)
        if ret_stmt:
            if self.current_body is None:
                self.program.toplevel_return = ret_stmt
            else:
                self.current_body.append(ret_stmt)
        self.i += 1
    
    def _parse_ai_prompt(self) -> None:
        """Parse AI generation prompt lines (> ...)."""
        prompt_lines: list[str] = []
        while self.i < len(self.lines):
            cur = self.lines[self.i]
            cur_s = cur.strip()
            if cur_s.startswith(">"):
                # Strip leading '>' and one space
                text = cur.lstrip()
                if text.startswith(">"):
                    text = text[1:]
                    if text.startswith(" "):
                        text = text[1:]
                prompt_lines.append(text)
                self.i += 1
                continue
            break
        
        # Attach to innermost procedure
        if self.proc_stack:
            self.proc_stack[-1].ai_prompt = "\n".join(prompt_lines)
    
    def _parse_template_line(self, stripped: str) -> None:
        """Parse a template line."""
        segments = parse_segments(stripped)
        stmt = TemplateLine(segments=segments)
        if self.current_body is None:
            self.program.toplevel_body.append(stmt)
        else:
            self.current_body.append(stmt)
        self.i += 1