"""Block parsing for Python blocks and validation suites."""

from __future__ import annotations

from typing import Tuple

from ..ast import (
    AssignBlockStmt,
    EvalDataset,
    EvalMetric,
    EvalSuite,
    PythonBlock,
    ReturnBlockStmt,
    TestCase,
    TestSuite,
)
from .preprocessing import BEGIN, END


class BlockParser:
    """Parses block structures (Python blocks, validation suites)."""

    def parse_python_block(self, lines: list[str], start_idx: int) -> Tuple[PythonBlock, int]:
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
        self, lines: list[str], start_idx: int, name: str, type_annotation: str | None
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
        return (
            AssignBlockStmt(name=name, code="\n".join(dedented), type_annotation=type_annotation),
            i + 1,
        )

    def parse_return_block(self, lines: list[str], start_idx: int) -> Tuple[ReturnBlockStmt, int]:
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
        self, lines: list[str], start_idx: int, header: str
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

                # Parse directive: "> case: name:", "> metric: name(dataset):", "> data: name:", etc.
                # Store the full directive line as the item name; we'll parse it in _build_*_suite
                item_name = cur_s
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
        self, procedure_name: str, items: list[tuple[str, list[str]]]
    ) -> TestSuite:
        """Build TestSuite from parsed items.
        
        Items are tuples of (directive_line, body_lines) where directive_line is like:
        - "> case: case_name:"
        """
        suite = TestSuite(procedure_name=procedure_name)

        for directive_line, body_lines in items:
            # Parse "> case: name:"
            directive = directive_line.strip()
            if not directive.startswith(">") or ":" not in directive:
                raise ValueError(f"Invalid test case directive: {directive}")
            
            parts = directive[1:].strip().split(":", 1)
            if len(parts) != 2 or parts[0].strip() != "case":
                raise ValueError(f"Invalid test case directive format: {directive_line}")
            
            name = parts[1].strip()
            if name.endswith(":"):
                name = name[:-1].strip()
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
        self, procedure_name: str, items: list[tuple[str, list[str]]]
    ) -> EvalSuite:
        """Build EvalSuite from parsed items.
        
        Items are tuples of (directive_line, body_lines) where directive_line is like:
        - "> data: dataset_name:"
        - "> test_data: dataset_name:"
        - "> metric: metric_name(dataset_name):"
        """
        suite = EvalSuite(procedure_name=procedure_name)
        VALID_DIRECTIVES = {"data", "test_data", "metric"}

        for directive_line, body_lines in items:
            bl = [b for b in body_lines if b.strip() not in ("", BEGIN, END)]
            
            # Parse directive: "> TYPE: NAME:" or "> metric: NAME(DATASET):"
            directive = directive_line.strip()
            if not directive.startswith(">") or ":" not in directive:
                raise ValueError(f"Invalid eval directive: {directive}")
            
            # Remove leading ">"
            directive = directive[1:].strip()
            # Split on first ":"
            parts = directive.split(":", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid eval directive format: {directive_line}")
            
            directive_type = parts[0].strip()
            directive_value = parts[1].strip()
            
            # Remove trailing colon if present
            if directive_value.endswith(":"):
                directive_value = directive_value[:-1].strip()
            
            if directive_type not in VALID_DIRECTIVES:
                raise ValueError(
                    f"Unknown directive '{directive_type}' in @eval: {procedure_name}; "
                    f"expected one of: {', '.join(VALID_DIRECTIVES)}"
                )
            
            # Handle data/test_data
            if directive_type == "data" or directive_type == "test_data":
                if not bl:
                    raise ValueError(f"Empty {directive_type} body")
                
                first = bl[0].strip()
                if not first.startswith("="):
                    raise ValueError(f"{directive_type} must start with '='")
                
                expr = first[1:].strip()
                dataset_name = directive_value
                
                # Inline backtick: = `expr`
                if expr.startswith("`") and expr.endswith("`") and expr.count("`") == 2:
                    code = expr[1:-1]
                    is_block = False
                # Block: = ```
                elif expr == "```":
                    block_lines = []
                    for k in range(1, len(bl)):
                        if bl[k].strip() == "```":
                            break
                        if bl[k].strip() in (BEGIN, END):
                            continue
                        block_lines.append(bl[k])
                    code = "\n".join(block_lines)
                    is_block = True
                else:
                    code = expr
                    is_block = False
                
                dataset = EvalDataset(name=dataset_name, code=code, is_block=is_block)
                if directive_type == "data":
                    suite.datasets[dataset_name] = dataset
                else:  # test_data
                    suite.test_datasets[dataset_name] = dataset
            
            # Handle metric
            elif directive_type == "metric":
                # Validate: only one metric per procedure
                if len(suite.metrics) > 0:
                    raise ValueError(
                        f"Multiple metrics defined in @eval: {procedure_name}. "
                        f"Only one metric per procedure is allowed. "
                        f"Found: {', '.join(m.name for m in suite.metrics)} and {directive_value.split('(')[0].strip()}"
                    )
                
                # Parse "metric_name(dataset_name)" or just "metric_name"
                metric_name = directive_value
                dataset_name = None
                
                if "(" in metric_name and metric_name.endswith(")"):
                    # Extract dataset name from parentheses
                    paren_idx = metric_name.index("(")
                    dataset_name = metric_name[paren_idx + 1 : -1].strip()
                    metric_name = metric_name[:paren_idx].strip()
                
                if not bl:
                    raise ValueError("Empty metric body")
                
                first = bl[0].strip()
                if not first.startswith("="):
                    raise ValueError("Metric must start with '='")
                
                expr = first[1:].strip()
                
                # Inline backtick: = `expr`
                if expr.startswith("`") and expr.endswith("`") and expr.count("`") == 2:
                    code = expr[1:-1]
                    suite.metrics.append(
                        EvalMetric(name=metric_name, code=code, is_block=False, dataset_name=dataset_name)
                    )
                # Block metric: = ```
                elif expr == "```":
                    block_lines = []
                    for k in range(1, len(bl)):
                        if bl[k].strip() == "```":
                            break
                        if bl[k].strip() in (BEGIN, END):
                            continue
                        block_lines.append(bl[k])
                    suite.metrics.append(
                        EvalMetric(
                            name=metric_name, code="\n".join(block_lines), is_block=True, dataset_name=dataset_name
                        )
                    )
                # Inline expression
                else:
                    suite.metrics.append(
                        EvalMetric(name=metric_name, code=expr, is_block=False, dataset_name=dataset_name)
                    )

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