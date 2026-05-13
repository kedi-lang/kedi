; Scope and reference information for the Kedi LSP and editor "go to
; definition" / "find references" features.
;
; Scopes:
;   - The whole file is a top-level scope.
;   - Every `procedure_def` opens a nested scope.
;
; Definitions:
;   - Procedure names (definable at top level or inside other procedures).
;   - Type-definition names.
;   - Type-field names within a `~Type(...)` definition.
;   - Parameter names within a `procedure_def`.
;   - Output-placeholder names (variables produced by the LLM call).
;   - Assignment-target names.
;
; References:
;   - `<input>` segment names.
;   - Call segment names (callee).
;   - Type identifiers inside `type_ref` / `type_apply`.

; ----------------------------------------------------------------
; Scopes
; ----------------------------------------------------------------

(source_file) @local.scope
(procedure_def) @local.scope

; ----------------------------------------------------------------
; Definitions
; ----------------------------------------------------------------

(procedure_def name: (identifier) @local.definition.function)
(type_def name: (identifier) @local.definition.type)
(type_field name: (identifier) @local.definition.field)
(param name: (identifier) @local.definition.parameter)
(assign_target name: (identifier) @local.definition.var)
(output_segment name: (identifier) @local.definition.var)

; ----------------------------------------------------------------
; References
; ----------------------------------------------------------------

(input_segment name: (identifier) @local.reference)
(call_segment name: (identifier) @local.reference)
(type_ref name: (identifier) @local.reference)
(type_apply name: (identifier) @local.reference)
(eval_metric dataset: (identifier) @local.reference)
