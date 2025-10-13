"""Block parsing for Python blocks and validation suites."""
from __future__ import annotations

from typing import Tuple
from ..ast import (
    PythonBlock, AssignBlockStmt, ReturnBlockStmt,
    TestSuite, TestCase, EvalSuite, EvalMetric
)
from .preprocessing import BEGIN, END


class BlockParser:
    """Parses block structures (Python blocks, validation suites)."""
    
    def parse_python_block(
        self,
        lines: list[str],
        start_idx: int
    ) -> Tuple[PythonBlock, int]:
        """Parse a Python block enclosed in ``` fences.
        
        Returns: (PythonBlock, next_line_index)
        """
        if start_idx >= len(lines) or lines[start_idx].strip() != "```":
            raise ValueError("Python block must start with ``` fence")
        
        i = start_idx + 1
        block_lines: list[str] = []
        
        while i < len(lines):
            cur = lines[i]
            cur_stripped = cur.strip()
            
            if cur_stripped == "```":
                # Found closing fence
                break
            
            # Skip preprocess markers
            if cur_stripped in (BEGIN, END):
                i += 1
                continue
            
            block_lines.append(cur)
            i += 1
        
        if i >= len(lines) or lines[i].strip() != "```":
            raise ValueError("Unterminated python block; expected closing ``` line")
        
        # Dedent the block
        dedented = self._dedent_block(block_lines)
        return PythonBlock(code="\n".join(dedented)), i + 1
    
    def parse_assignment_block(
        self,
        lines: list[str],
        start_idx: int,
        name: str,
        type_annotation: str | None
    ) -> Tuple[AssignBlockStmt, int]:
        """Parse assignment with Python block RHS: [name] = ``` ... ```
        
        Note: start_idx points to the line AFTER the opening ``` (which is on the assignment line)
        """
        i = start_idx
        block_lines: list[str] = []
        
        while i < len(lines):
            cur = lines[i]
            cur_stripped = cur.strip()
            
            if cur_stripped == "```":
                # Found closing fence
                break
            
            # Skip preprocess markers
            if cur_stripped in (BEGIN, END):
                i += 1
                continue
            
            block_lines.append(cur)
            i += 1
        
        if i >= len(lines) or lines[i].strip() != "```":
            raise ValueError("Unterminated python block; expected closing ``` line")
        
        # Dedent the block
        dedented = self._dedent_block(block_lines)
        return AssignBlockStmt(
            name=name,
            code="\n".join(dedented),
            type_annotation=type_annotation
        ), i + 1
    
    def parse_return_block(
        self,
        lines: list[str],
        start_idx: int
    ) -> Tuple[ReturnBlockStmt, int]:
        """Parse return with Python block: = ``` ... ```
        
        Note: start_idx points to the line AFTER the opening ``` (which is on the return line)
        """
        i = start_idx
        block_lines: list[str] = []
        
        while i < len(lines):
            cur = lines[i]
            cur_stripped = cur.strip()
            
            if cur_stripped == "```":
                # Found closing fence
                break
            
            # Skip preprocess markers
            if cur_stripped in (BEGIN, END):
                i += 1
                continue
            
            block_lines.append(cur)
            i += 1
        
        if i >= len(lines) or lines[i].strip() != "```":
            raise ValueError("Unterminated python block; expected closing ``` line")
        
        # Dedent the block
        dedented = self._dedent_block(block_lines)
        return ReturnBlockStmt(code="\n".join(dedented)), i + 1
    
    def parse_validation_block(
        self,
        lines: list[str],
        start_idx: int,
        header: str
    ) -> Tuple[TestSuite | EvalSuite, int]:
        """Parse @test: name: or @eval: name: validation block.
        
        Returns: (TestSuite or EvalSuite, next_line_index)
        """
        # Parse header
        header = header[1:-1]  # Remove leading @ and trailing :
        kind, rest = header.split(":", 1)
        kind = kind.strip()
        target_name = rest.strip()
        
        # Expect BEGIN marker on next line
        i = start_idx
        if i >= len(lines) or lines[i].strip() != BEGIN:
            raise ValueError("Validation block must start with an indented block")
        i += 1
        
        # Collect items until matching END
        items: list[tuple[str, list[str]]] = []
        item_name: str | None = None
        item_body: list[str] = []
        depth = 0
        
        while i < len(lines):
            cur = lines[i]
            cur_s = cur.strip()
            
            if cur_s == BEGIN:
                depth += 1
                i += 1
                continue
            
            if cur_s == END:
                if depth == 0:
                    # Flush last item
                    if item_name is not None:
                        items.append((item_name, item_body))
                    break
                else:
                    depth -= 1
                    i += 1
                    continue
            
            # Item header at depth 0
            if depth == 0 and cur_s.startswith(">"):
                # Flush previous item
                if item_name is not None:
                    items.append((item_name, item_body))
                    item_body = []
                
                # Parse: "> case: name:" or "> metric: name:"
                if ":" not in cur_s:
                    raise ValueError("Invalid validation item header")
                arrow, rest2 = cur_s.split(":", 1)
                if ":" not in rest2:
                    raise ValueError("Invalid validation item header")
                label, name_part = rest2.split(":", 1)
                item_name = name_part.strip()
                if item_name.endswith(":"):
                    item_name = item_name[:-1].strip()
                i += 1
                continue
            
            # Accumulate body lines
            item_body.append(cur)
            i += 1
        
        # Consume END marker
        i += 1
        
        # Build suite
        if kind == "test":
            return self._build_test_suite(target_name, items), i
        else:  # eval
            return self._build_eval_suite(target_name, items), i
    
    def _build_test_suite(
        self,
        procedure_name: str,
        items: list[tuple[str, list[str]]]
    ) -> TestSuite:
        """Build TestSuite from parsed items."""
        suite = TestSuite(procedure_name=procedure_name)
        
        for name, body_lines in items:
            bl = [b for b in body_lines if b.strip() not in ("", BEGIN, END)]
            if not bl:
                raise ValueError("Empty test case body")
            
            first = bl[0].strip()
            
            # Single-line backtick
            if first.startswith("`") and first.endswith("`") and first.count("`") == 2:
                code = first[1:-1]
                suite.cases.append(TestCase(name=name, code=code, is_block=False))
            # Fenced block
            elif first == "```":
                j = 1
                block_lines: list[str] = []
                while j < len(bl):
                    if bl[j].strip() == "```":
                        break
                    if bl[j].strip() in (BEGIN, END):
                        j += 1
                        continue
                    block_lines.append(bl[j])
                    j += 1
                suite.cases.append(TestCase(name=name, code="\n".join(block_lines), is_block=True))
            else:
                raise ValueError("Test case must be backtick line or fenced python block")
        
        return suite
    
    def _build_eval_suite(
        self,
        procedure_name: str,
        items: list[tuple[str, list[str]]]
    ) -> EvalSuite:
        """Build EvalSuite from parsed items."""
        suite = EvalSuite(procedure_name=procedure_name)
        
        for name, body_lines in items:
            bl = [b for b in body_lines if b.strip() not in ("", BEGIN, END)]
            if not bl:
                raise ValueError("Empty metric body")
            
            first = bl[0].strip()
            
            if not first.startswith("="):
                raise ValueError("Metric must start with '='")
            
            expr = first[1:].strip()
            
            # Inline backtick: = `expr`
            if expr.startswith("`") and expr.endswith("`") and expr.count("`") == 2:
                expr = expr[1:-1]
                suite.metrics.append(EvalMetric(name=name, code=expr, is_block=False))
            # Block metric: = ```
            elif expr == "```":
                block_lines: list[str] = []
                for k in range(1, len(bl)):
                    if bl[k].strip() == "```":
                        break
                    if bl[k].strip() in (BEGIN, END):
                        continue
                    block_lines.append(bl[k])
                suite.metrics.append(EvalMetric(name=name, code="\n".join(block_lines), is_block=True))
            # Inline expression
            else:
                suite.metrics.append(EvalMetric(name=name, code=expr, is_block=False))
        
        return suite
    
    @staticmethod
    def _dedent_block(lines: list[str]) -> list[str]:
        """Dedent a block of code relative to minimal indent."""
        def indent_of(s: str) -> int:
            n = 0
            for ch in s:
                if ch == " ":
                    n += 1
                elif ch == "\t":
                    n += 4
                else:
                    break
            return n
        
        non_empty = [s for s in lines if s.strip() != ""]
        if not non_empty:
            return lines
        
        min_indent = min(indent_of(s) for s in non_empty)
        dedented = []
        
        for s in lines:
            j = 0
            removed = 0
            while j < len(s) and removed < min_indent:
                if s[j] == " ":
                    removed += 1
                    j += 1
                elif s[j] == "\t":
                    removed += 4
                    j += 1
                else:
                    break
            dedented.append(s[j:])
        
        return dedented