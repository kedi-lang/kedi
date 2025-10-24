"""Statement parsing for Marker DSL."""
from __future__ import annotations

import re
from typing import Tuple
from ..ast import (
    AssignStmt, AssignBlockStmt, ReturnStmt, ReturnBlockStmt,
    TemplateLine, TypeDef, TypeField, PythonExprNativeSegment
)
from .segment_parser import parse_segments, find_matching_delimiter


class StatementParser:
    """Parses individual statement types."""
    
    @staticmethod
    def is_assignment(line: str) -> bool:
        """Check if line is an assignment statement."""
        stripped = line.strip()
        return stripped.startswith("[") and "]" in stripped and "=" in stripped
    
    @staticmethod
    def is_return(line: str) -> bool:
        """Check if line is a return statement."""
        return line.strip().startswith("=")
    
    @staticmethod
    def is_type_definition(line: str) -> bool:
        """Check if line is a type definition."""
        return line.strip().startswith("~")
    
    @staticmethod
    def is_procedure_header(line: str) -> bool:
        """Check if line is a procedure header."""
        stripped = line.strip()
        return stripped.startswith("@") and stripped.endswith(":")
    
    def parse_assignment(self, line: str) -> AssignStmt | None:
        """Parse assignment statement [name] = expr or [name: type] = expr or [name: `type_expr`] = expr."""
        stripped = line.strip()
        if not self.is_assignment(line):
            return None
        
        # Find closing bracket using matching
        close_br = find_matching_delimiter(stripped, 0, "[", "]")
        target_inner = stripped[1:close_br].strip()
        rest = stripped[close_br + 1:].lstrip()
        
        if not rest.startswith("="):
            raise ValueError("Malformed assignment; expected '=' after [name]")
        
        rhs_src = rest[1:]
        
        # Parse target [name] or [name: type] or [name: `type_expr`]
        type_anno: str | None = None
        if ":" in target_inner:
            name_part, type_part = target_inner.split(":", 1)
            name = name_part.strip()
            type_anno = type_part.strip()
        else:
            name = target_inner
        
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ValueError("Assignment target must be an identifier")
        
        # Special case: backtick-only inline python
        rhs_trim = rhs_src.strip()
        if (rhs_trim.startswith("`") and rhs_trim.endswith("`") and 
            len(rhs_trim) >= 2 and rhs_trim.count("`") == 2):
            code = rhs_trim[1:-1]
            return AssignStmt(
                name=name,
                expr=[PythonExprNativeSegment(code=code)],
                type_annotation=type_anno
            )
        
        # Regular assignment with segments
        rhs_segments = parse_segments(rhs_src)
        return AssignStmt(name=name, expr=rhs_segments, type_annotation=type_anno)
    
    def parse_return(self, line: str) -> ReturnStmt | None:
        """Parse return statement = expr."""
        if not self.is_return(line):
            return None
        
        stripped = line.strip()
        expr_src = stripped[1:]
        
        # Inline python native return: = `expr`
        expr_trim = expr_src.strip()
        if (expr_trim.startswith("`") and expr_trim.endswith("`") and
            len(expr_trim) >= 2 and expr_trim.count("`") == 2):
            # This will be handled as ReturnBlockStmt by the caller
            return None
        
        segments = parse_segments(expr_src)
        return ReturnStmt(segments=segments)
    
    def parse_type_definition(self, line: str) -> TypeDef:
        """Parse type definition ~Name(field, ...) or ~Name(field: `type_expr`, ...)."""
        if not re.match(r"^~[A-Za-z_][A-Za-z0-9_]*\(.*\)$", line):
            raise ValueError("Type definition must follow syntax ~Name(field, ...)")
        
        name_part, rest = line[1:].split("(", 1)
        name = name_part.strip()
        
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ValueError("Type name must be an identifier")
        
        if not rest.endswith(")"):
            raise ValueError("Type definition must end with ')'")
        
        fields_src = rest[:-1].strip()
        fields: list[TypeField] = []
        seen: set[str] = set()
        
        if fields_src:
            for raw in self._split_type_fields(fields_src):
                if ":" in raw:
                    fname_part, ftype_part = raw.split(":", 1)
                    fname = fname_part.strip()
                    ftype = ftype_part.strip()
                    if ftype == "":
                        raise ValueError("Field type annotation cannot be empty")
                else:
                    fname = raw.strip()
                    ftype = None
                
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", fname):
                    raise ValueError("Field name must be an identifier")
                if fname in seen:
                    raise ValueError(f"Duplicate field '{fname}' in type definition")
                
                seen.add(fname)
                fields.append(TypeField(name=fname, type_annotation=ftype))
        
        return TypeDef(name=name, fields=fields)
    
    def parse_procedure_header(self, line: str) -> Tuple[str, list[str], dict[str, str | None], str | None]:
        """Parse procedure header @name(params) -> returntype: or @name(params) -> `returntype_expr`:.
        
        Returns: (name, params, param_type_annotations, return_type_annotation)
        """
        if not self.is_procedure_header(line):
            raise ValueError("Not a procedure header")
        
        stripped = line.strip()
        header = stripped[1:-1]  # Remove @ and :
        
        # Support optional return type: name(params) -> type or name(params) -> `type_expr`
        ret_anno: str | None = None
        if "->" in header:
            header, ret_anno_src = header.split("->", 1)
            ret_anno = ret_anno_src.strip()
        
        if "(" not in header:
            raise ValueError("Procedure header missing '('")
        
        name, params_src = header.split("(", 1)
        name = name.strip()
        params_src = params_src.strip()
        
        if not params_src.endswith(")"):
            raise ValueError("Procedure header malformed")
        
        params_csv = params_src[:-1]
        params: list[str] = []
        param_type_annos: dict[str, str | None] = {}
        
        for p in [pp.strip() for pp in params_csv.split(",") if pp.strip()]:
            if ":" in p:
                pname, pty = p.split(":", 1)
                pname = pname.strip()
                pty = pty.strip()
                params.append(pname)
                param_type_annos[pname] = pty
            else:
                params.append(p)
                param_type_annos[p] = None
        
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ValueError("Procedure name must be an identifier")
        
        return name, params, param_type_annos, ret_anno
    
    @staticmethod
    def _split_type_fields(src: str) -> list[str]:
        """Split type field definitions by commas, respecting nesting."""
        fields: list[str] = []
        buf: list[str] = []
        depth_paren = depth_bracket = depth_brace = 0
        in_string: str | None = None
        i = 0
        
        while i < len(src):
            ch = src[i]
            
            if ch in ('"', "'"):
                if in_string == ch:
                    in_string = None
                elif in_string is None:
                    in_string = ch
                buf.append(ch)
                i += 1
                continue
            
            if in_string is not None:
                buf.append(ch)
                i += 1
                continue
            
            if ch == "(":
                depth_paren += 1
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]":
                depth_bracket = max(0, depth_bracket - 1)
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace = max(0, depth_brace - 1)
            
            if ch == "," and depth_paren == depth_bracket == depth_brace == 0:
                field = "".join(buf).strip()
                if field == "":
                    raise ValueError("Empty field in type definition")
                fields.append(field)
                buf.clear()
                i += 1
                continue
            
            buf.append(ch)
            i += 1
        
        if buf:
            field = "".join(buf).strip()
            if field == "":
                raise ValueError("Empty field in type definition")
            fields.append(field)
        
        return fields