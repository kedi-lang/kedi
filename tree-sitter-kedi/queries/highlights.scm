; Syntax highlighting for Kedi (tree-sitter-kedi)
;
; Editor consumers: VS Code, Neovim (nvim-treesitter), Helix, Zed,
; tree-sitter highlight CLI. Captures use the standard tree-sitter
; capture names so existing themes pick them up without extra mapping.

; ----------------------------------------------------------------
; Comments
; ----------------------------------------------------------------

(line_comment) @comment.line
(block_comment) @comment.block

; ----------------------------------------------------------------
; Operators / punctuation
; ----------------------------------------------------------------

["@" "~" "=" ":" "(" ")" "[" "]" "<" ">" "," "|" "->"] @punctuation.delimiter
["```"] @punctuation.special
"`" @punctuation.special

; ----------------------------------------------------------------
; Procedure & validation block keywords
; ----------------------------------------------------------------

(validation_keyword) @keyword
"auto" @keyword
"optimize" @keyword
"case" @keyword
"data" @keyword
"test_data" @keyword
"metric" @keyword

; ----------------------------------------------------------------
; Procedure and type-definition names
; ----------------------------------------------------------------

(procedure_def name: (identifier) @function)
(type_def name: (identifier) @type.definition)
(param name: (identifier) @variable.parameter)
(type_field name: (identifier) @property)

; ----------------------------------------------------------------
; Template segments
; ----------------------------------------------------------------

(input_segment name: (identifier) @variable)
(call_segment name: (identifier) @function.call)
(output_segment name: (identifier) @variable.builtin)
(text_segment) @string

; ----------------------------------------------------------------
; Type expressions
; ----------------------------------------------------------------

(type_ref name: (identifier) @type)
(type_apply name: (identifier) @type)

; ----------------------------------------------------------------
; Validation suite names (after `@test:` / `@eval:`)
; ----------------------------------------------------------------

(validation_block procedure: (identifier) @function)
(test_case name: (identifier) @label)
(eval_data name: (identifier) @label)
(eval_test_data name: (identifier) @label)
(eval_metric name: (identifier) @label)
(eval_metric dataset: (identifier) @variable)

(optimize_directive name: (identifier) @label)

; ----------------------------------------------------------------
; Python source embedded in Kedi (these regions get a Python
; injection — see queries/injections.scm — but we also give them a
; fallback "code" style so they remain readable when no injection
; parser is available).
; ----------------------------------------------------------------

(python_code) @string.special
(python_inline_body) @string.special
