"""Top-level program parsing orchestration."""

from __future__ import annotations

from typing import List

from ..ast import (
    AssignBlockStmt,
    AssignStmt,
    AutoSpec,
    BacktickLineStmt,
    OptimizeSpan,
    ProcDefStmt,
    ProcedureDef,
    Program,
    PythonBlock,
    ReturnBlockStmt,
    ReturnStmt,
    Stmt,
    TemplateLine,
    TypeDefStmt,
)
from .block_parser import BlockParser
from .preprocessing import BEGIN, END, merge_line_continuations, preprocess_blocks, strip_comments
from .segment_parser import parse_segments
from .statement_parser import StatementParser


def parse_program(source: str) -> Program:
    """Parse Kedi DSL source code into a Program AST.

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
                self._handle_end_kedi()
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
            if (
                stripped.startswith("`")
                and stripped.endswith("`")
                and len(stripped) >= 2
                and stripped.count("`") == 2
            ):
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

            # Directive lines (inside procedure body): > auto:, > optimize: name:
            if stripped.startswith(">") and self.current_body is not None and self.proc_stack:
                self._parse_directive()
                continue

            # Template line
            self._parse_template_line(stripped)

        # Close any open procedures
        while self.proc_stack:
            self.program.procedures.append(self.proc_stack.pop())

        return self.program

    def _handle_end_kedi(self) -> None:
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
        suite, next_i = self.block_parser.parse_validation_block(self.lines, self.i + 1, stripped)
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
            param_type_annotations=param_annos,
        )

        self.proc_stack.append(proc)
        self.current_body = proc.body

    def _parse_python_block(self) -> None:
        """Parse Python block (``` ... ```)."""
        block, next_i = self.block_parser.parse_python_block(self.lines, self.i)

        # Check if this is the prelude (first python block before any procedures/content)
        if (
            self.program.prelude is None
            and not self.program.toplevel_body
            and self.program.toplevel_return is None
            and not self.program.procedures
            and not self.proc_stack
        ):  # Not inside a procedure being parsed
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
        if (
            expr_trim.startswith("`")
            and expr_trim.endswith("`")
            and len(expr_trim) >= 2
            and expr_trim.count("`") == 2
        ):
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

    def _parse_directive(self) -> None:
        """Parse directive lines: > auto:, > optimize: name:.
        
        Valid directives:
        - > auto: ... (codegen spec)
        - > optimize: name: ... (optimize span)
        """
        if not self.proc_stack:
            raise ValueError("Directive must be inside a procedure")
        
        proc = self.proc_stack[-1]
        cur_line = self.lines[self.i]
        cur_s = cur_line.strip()
        
        if not cur_s.startswith(">"):
            return
        
        directive = cur_s[1:].strip()
        
        # Parse > auto:
        if directive.startswith("auto:"):
            auto_content = directive[len("auto:"):].strip()
            auto_lines = [auto_content] if auto_content else []
            self.i += 1
            
            # Check if there's an indented block (BEGIN marker)
            if self.i < len(self.lines) and self.lines[self.i].strip() == BEGIN:
                # Has indented block - collect until matching END
                self.i += 1
                depth = 1
                
                while self.i < len(self.lines):
                    cur_line = self.lines[self.i]
                    cur_s = cur_line.strip()
                    
                    if cur_s == BEGIN:
                        depth += 1
                        self.i += 1
                        continue
                    
                    if cur_s == END:
                        depth -= 1
                        if depth == 0:
                            self.i += 1
                            break
                        self.i += 1
                        continue
                    
                    # Accumulate content lines
                    auto_lines.append(cur_line.lstrip())
                    self.i += 1
            else:
                # No indented block - collect continuation lines until next directive/statement
                while self.i < len(self.lines):
                    next_line = self.lines[self.i]
                    next_s = next_line.strip()
                    
                    # Stop on another directive or template/statement
                    if next_s.startswith(">"):
                        break
                    if next_s.startswith("@") or next_s == "```" or self.stmt_parser.is_return(next_line) or self.stmt_parser.is_assignment(next_s):
                        break
                    # Skip markers that aren't part of auto scope
                    if next_s in (BEGIN, END):
                        break
                    # Accumulate content lines
                    if next_s:
                        auto_lines.append(next_line.lstrip())
                    self.i += 1
            
            proc.auto_spec = AutoSpec(prompt="\n".join(auto_lines))
        
        # Parse > optimize: name:
        elif directive.startswith("optimize:"):
            parts = directive[len("optimize:"):].strip().split(":", 1)
            if len(parts) == 0 or not parts[0].strip():
                raise ValueError("Optimize directive must have a name: > optimize: name:")
            
            optimize_name = parts[0].strip()
            if optimize_name.endswith(":"):
                optimize_name = optimize_name[:-1].strip()
            
            self.i += 1
            
            # Expect BEGIN marker for the indented scope
            if self.i >= len(self.lines) or self.lines[self.i].strip() != BEGIN:
                raise ValueError(f"Optimize span '{optimize_name}' must have an indented block after it")
            self.i += 1
            
            # Collect template lines within the indented scope
            template_lines: list[TemplateLine] = []
            depth = 1  # Track nested BEGIN/END within the optimize span
            
            while self.i < len(self.lines):
                cur_line = self.lines[self.i]
                cur_s = cur_line.strip()
                
                # Track depth for nested scopes
                if cur_s == BEGIN:
                    depth += 1
                    self.i += 1
                    continue
                
                if cur_s == END:
                    depth -= 1
                    if depth == 0:
                        # This END closes the optimize span scope
                        self.i += 1
                        break
                    self.i += 1
                    continue
                
                # Stop if we hit another directive at same level (would be preceded by END)
                # Actually, END would have been seen already, so this shouldn't happen
                # But check for safety
                if cur_s.startswith(">") and depth == 1:
                    break
                
                # Stop on statement types at same level
                if depth == 1 and (cur_s.startswith("@") or cur_s == "```" or self.stmt_parser.is_return(cur_line) or self.stmt_parser.is_assignment(cur_s)):
                    break
                
                # Empty line - continue collecting
                if not cur_s:
                    self.i += 1
                    continue
                
                # Must be a template line - parse it
                segments = parse_segments(cur_s)
                template_line = TemplateLine(segments=segments)
                template_lines.append(template_line)
                # Also add to procedure body for normal execution
                proc.body.append(template_line)
                self.i += 1
            
            if not template_lines:
                raise ValueError(f"Optimize span '{optimize_name}' must contain at least one template line")
            
            span = OptimizeSpan(name=optimize_name, template_lines=template_lines)
            proc.optimize_spans.append(span)
        
        else:
            # Legacy codegen syntax: any `>` line without explicit directive
            # Collect as ai_prompt for backward compatibility
            prompt_lines = [cur_line]
            self.i += 1
            
            # Collect continuation lines (non-directive, non-template lines)
            while self.i < len(self.lines):
                next_line = self.lines[self.i]
                next_s = next_line.strip()
                
                # Stop on another directive, template/statement, or explicit > auto:/optimize:
                if next_s.startswith(">") and (next_s.startswith("> auto:") or next_s.startswith("> optimize:")):
                    break
                if next_s.startswith("@") or next_s == "```" or self.stmt_parser.is_return(next_line) or self.stmt_parser.is_assignment(next_s):
                    break
                # Accumulate content lines
                prompt_lines.append(next_line)
                self.i += 1
            
            # Strip leading `>` from each line
            cleaned_lines = []
            for line in prompt_lines:
                stripped = line.lstrip()
                if stripped.startswith(">"):
                    cleaned = stripped[1:].strip()
                    cleaned_lines.append(cleaned)
                else:
                    cleaned_lines.append(line.lstrip())
            
            # Set legacy ai_prompt for backward compatibility
            proc.ai_prompt = "\n".join(cleaned_lines)

    def _parse_template_line(self, stripped: str) -> None:
        """Parse a template line."""
        segments = parse_segments(stripped)
        stmt = TemplateLine(segments=segments)
        if self.current_body is None:
            self.program.toplevel_body.append(stmt)
        else:
            self.current_body.append(stmt)
        self.i += 1
