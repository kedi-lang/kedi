/**
 * @file Kedi grammar for tree-sitter
 * @author Doğukan Yiğit Polat <yigit@neurograph.net>
 * @author Mert Sırakaya <mert@kedi-lang.org>
 * @license Apache-2.0
 *
 * Kedi is a typed DSL for LLM orchestration. This grammar replaces the
 * legacy hand-written parser at src/kedi/lang/parser/ in the parent repo.
 *
 * Phase 2 surface area:
 *   - procedure_def, params, type_def, type_field, type_expr hierarchy
 *     (type_ref / type_apply / type_union / type_python).
 *   - assign_stmt, return_stmt, template_line, inline_python_expr,
 *     backtick_line_stmt.
 *   - Template segments: text, input, output, call, python_expr.
 *   - Call argument list with nested inputs, calls, python exprs, and
 *     raw text-in-call (handled by the external scanner so it stops on
 *     unescaped `,` and `)`).
 *
 * Deferred:
 *   - Triple-backtick fenced Python blocks → Phase 3.
 *   - `> auto:` and `> optimize:` directives → Phase 4.
 *   - `@test:` and `@eval:` validation blocks → Phase 4.
 *
 * See /home/yigit/.claude/plans/i-have-added-tree-sitter-effervescent-marshmallow.md
 * for the full design.
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

module.exports = grammar({
  name: "kedi",

  // IMPORTANT: this list's order must match `enum TokenType` in src/scanner.c.
  //
  // Single-backtick and triple-backtick are NOT externals: tree-sitter's
  // internal lexer handles them as literal tokens (`` ` `` and `` ``` ``)
  // and its longest-match rule selects the triple form wherever three
  // consecutive backticks appear. Doing the discrimination here instead
  // of in the scanner avoids the well-known "advances during a failed
  // external scan are not rewound" problem.
  //
  // `_fenced_newline` is like `_newline` but does NOT update the indent
  // stack; it is emitted between an opening "```" and the fenced body
  // so the body's leading indent is not mistaken for a real block
  // indent that would later trigger a spurious DEDENT.
  externals: ($) => [
    $._newline,
    $._indent,
    $._dedent,
    $._text,
    $._text_in_call,
    $._fenced_body,
    $._fenced_newline,
  ],

  // Extras run between tokens. We include:
  //
  //   - `block_comment` (`###`-delimited multi-line). Body matches any
  //     char-sequence that does not contain three consecutive `#`s: each
  //     character is either non-`#`, `#` followed by non-`#`, or `##`
  //     followed by non-`#` — the standard "everything except the
  //     closing fence" trick.
  //   - `line_comment` (single `#` to EOL).
  //   - `[ \t]` for horizontal whitespace.
  //   - `\n` so tree-sitter can fill the "gap" left when the external
  //     scanner advances past blank lines during NEWLINE indent-peek.
  //     The external NEWLINE token still wins when valid_symbols asks
  //     for it (externals run before extras), so newline significance
  //     for statement termination is preserved; this only matters when
  //     the scanner has already committed advances past one or more
  //     `\n`s during exploratory peek and tree-sitter needs to skip
  //     those bytes between the emitted NEWLINE token and the next
  //     real token.
  extras: ($) => [$.block_comment, $.line_comment, /[ \t]/, /\n/],

  word: ($) => $.identifier,

  // Because ``test_block`` and ``eval_block`` share an identical
  // ``@<id>:<id>:`` prefix (the kw is matched as a generic
  // ``$.identifier``, not a keyword — see the rule comments), tree-sitter
  // needs to keep both parse stacks alive until the inner directive
  // (``> case:`` vs ``> data:`` / ``> metric:``) decides which one wins.
  // Supertypes group related alternatives so the resulting node types are
  // easier to consume from the Python CST→AST walker.
  supertypes: ($) => [$._stmt, $._segment, $._top_item, $._type_expr_term],

  rules: {
    // ============================================================
    // Top level
    // ============================================================
    source_file: ($) => repeat(choice($._top_item, $._newline)),

    _top_item: ($) =>
      choice(
        $.procedure_def,
        $.type_def,
        $.validation_block,
        $.assign_stmt,
        $.assign_block_stmt,
        $.return_stmt,
        $.return_block_stmt,
        $.python_block,
        $.backtick_line_stmt,
        $.template_line,
      ),

    // ============================================================
    // Procedures
    // ============================================================
    procedure_def: ($) =>
      seq(
        "@",
        field("name", $.identifier),
        "(",
        optional(field("params", $.param_list)),
        ")",
        optional(seq("->", field("return_type", $.type_expr))),
        ":",
        $._newline,
        field("body", $.block),
      ),

    param_list: ($) => sep1($.param, ","),

    param: ($) =>
      seq(
        field("name", $.identifier),
        optional(seq(":", field("type", $.type_expr))),
      ),

    block: ($) =>
      seq($._indent, repeat1(choice($._stmt, $._newline)), $._dedent),

    _stmt: ($) =>
      choice(
        $.procedure_def,
        $.type_def_stmt,
        $.auto_directive,
        $.optimize_directive,
        $.assign_stmt,
        $.assign_block_stmt,
        $.return_stmt,
        $.return_block_stmt,
        $.python_block,
        $.backtick_line_stmt,
        $.template_line,
      ),

    // ============================================================
    // Type definitions
    // ============================================================
    type_def: ($) =>
      seq(
        "~",
        field("name", $.identifier),
        "(",
        optional(field("fields", $.type_field_list)),
        ")",
        $._newline,
      ),

    type_def_stmt: ($) => $.type_def,

    type_field_list: ($) => sep1($.type_field, ","),

    type_field: ($) =>
      seq(
        field("name", $.identifier),
        optional(seq(":", field("type", $.type_expr))),
      ),

    // ============================================================
    // Type expressions
    //
    // Grammar order: type_python (backtick escape hatch) > type_union >
    // type_apply / type_ref. Left-associative `|` for unions.
    // ============================================================
    type_expr: ($) => choice($.type_python, $._type_expr_term),

    _type_expr_term: ($) => choice($.type_union, $.type_apply, $.type_ref),

    type_union: ($) =>
      prec.left(
        1,
        seq(field("left", $._type_expr_term), "|", field("right", $._type_expr_term)),
      ),

    type_apply: ($) =>
      seq(
        field("name", $.identifier),
        "[",
        field("args", sep1($.type_expr, ",")),
        "]",
      ),

    type_ref: ($) => field("name", $.identifier),

    type_python: ($) => seq("`", field("code", $.python_inline_body), "`"),

    // ============================================================
    // Assignment & return
    //
    // Backtick-fenced Python blocks (Phase 3) are NOT yet handled here;
    // the inline_python_expr alternative covers single-line `code`.
    // ============================================================
    assign_stmt: ($) =>
      seq(
        $.assign_target,
        "=",
        field("rhs", choice($.inline_python_expr, $.template_expr)),
        $._newline,
      ),

    assign_target: ($) =>
      seq(
        "[",
        field("name", $.identifier),
        optional(seq(":", field("type", $.type_expr))),
        "]",
      ),

    return_stmt: ($) =>
      seq(
        "=",
        field("value", choice($.inline_python_expr, $.template_expr)),
        $._newline,
      ),

    backtick_line_stmt: ($) => seq($.inline_python_expr, $._newline),

    // ============================================================
    // Fenced Python blocks (Phase 3)
    //
    // A fenced block opens and closes with a line whose stripped content
    // is exactly "```". The body content (between the two fences) is
    // captured as a single `python_code` token by the external scanner
    // (`_fenced_body`), including embedded specials (`<`, `>`, `[`, `]`,
    // `=`, `~`, `@`, single backticks, etc.) and newlines.
    //
    // The opening "```" is followed by a `_newline` token which lets the
    // scanner enter fenced-body mode. The body absorbs everything up to
    // (but not including) the closing "```" line, which the grammar
    // matches as another literal "```" token. The trailing `_newline`
    // after the close fence triggers indent-stack tracking on the next
    // statement line.
    //
    // Common-indent dedenting of body content is handled by the Phase 6
    // CST→AST walker, NOT by the scanner — the raw bytes are preserved
    // here so editors can inject a Python parser into `python_code`.
    // ============================================================
    python_block: ($) =>
      seq("```", $._fenced_newline, field("code", $.python_code), "```", $._newline),

    assign_block_stmt: ($) =>
      seq(
        $.assign_target,
        "=",
        "```",
        $._fenced_newline,
        field("code", $.python_code),
        "```",
        $._newline,
      ),

    return_block_stmt: ($) =>
      seq(
        "=",
        "```",
        $._fenced_newline,
        field("code", $.python_code),
        "```",
        $._newline,
      ),

    python_code: ($) => $._fenced_body,

    // ============================================================
    // Template lines and segments
    // ============================================================
    template_line: ($) => seq($.template_expr, $._newline),

    template_expr: ($) => repeat1($._segment),

    _segment: ($) =>
      choice(
        $.input_segment,
        $.call_segment,
        $.python_expr_segment,
        $.output_segment,
        $.text_segment,
      ),

    text_segment: ($) => $._text,

    input_segment: ($) => seq("<", field("name", $.identifier), ">"),

    call_segment: ($) =>
      seq(
        "<",
        field("name", $.identifier),
        "(",
        optional(field("args", $.call_arg_list)),
        ")",
        ">",
      ),

    call_arg_list: ($) => sep1($.call_arg, ","),

    // Call arguments are either a single pure `python_expr` (passed as a
    // native Python value) or a "mini template" (rendered to a string at
    // call time). The mini-template form lets users embed `<inputs>` and
    // `[outputs]` and arbitrary text inside an argument — this matches
    // the legacy parser, which calls `parse_segments` on each comma-
    // separated argument and supports the full segment vocabulary.
    call_arg: ($) => choice($.inline_python_expr, $.template_arg),

    template_arg: ($) =>
      repeat1(
        choice(
          $.input_segment,
          $.output_segment,
          $.call_segment,
          $.python_expr_segment,
          $.text_in_call_segment,
        ),
      ),

    text_in_call_segment: ($) => $._text_in_call,

    output_segment: ($) =>
      seq(
        "[",
        field("name", $.identifier),
        optional(seq(":", field("type", $.type_expr))),
        "]",
      ),

    python_expr_segment: ($) => seq("<", $.inline_python_expr, ">"),

    // ============================================================
    // Directives inside procedure bodies (Phase 4)
    //
    //   > auto:
    //     <indented free-text spec for an AI-generated procedure>
    //
    //   > optimize: <name>:
    //     <indented template lines for prompt optimisation>
    //
    // The `auto` body is a sequence of plain text lines (the prompt
    // text). The `optimize` body is a sequence of template_lines
    // (full Kedi template segments) — each line ends up as a span the
    // GEPA optimizer can rewrite.
    // ============================================================
    auto_directive: ($) =>
      seq(
        ">",
        "auto",
        ":",
        $._newline,
        field("body", $.auto_body),
      ),

    auto_body: ($) =>
      seq($._indent, repeat1(choice($.template_line, $._newline)), $._dedent),

    optimize_directive: ($) =>
      seq(
        ">",
        "optimize",
        ":",
        field("name", $.identifier),
        ":",
        $._newline,
        field("body", $.optimize_body),
      ),

    optimize_body: ($) =>
      seq($._indent, repeat1(choice($.template_line, $._newline)), $._dedent),

    // ============================================================
    // Validation blocks (Phase 4)
    //
    //   @test: <procedure_name>:
    //     > case: <name>:
    //       <case body — backtick line or fenced Python block>
    //
    //   @eval: <procedure_name>:
    //     > data: <name>:
    //       = ``` ... ```
    //     > test_data: <name>:
    //       = ``` ... ```
    //     > metric: <name>(<dataset_name>):
    //       = ``` ... ```
    // ============================================================
    // Validation blocks open with `@<kw>:` where <kw> is the literal
    // identifier ``test`` or ``eval``. We deliberately match these via
    // the generic ``$.identifier`` rule (not via string literals) so
    // they do NOT collide with tree-sitter's keyword extraction —
    // otherwise a user procedure named ``test`` (``@test(...)``) gets
    // steered into the validation arm and fails to parse.
    //
    // The body alternatives ``test_case`` / ``eval_data`` /
    // ``eval_test_data`` / ``eval_metric`` are deliberately ALL
    // accepted by the unified ``validation_body`` rule; the CST→AST
    // walker discriminates ``test`` vs ``eval`` from the kw text and
    // rejects mismatched body content with a structured diagnostic
    // (e.g. "Unknown directive in @eval body" when ``> case:`` shows
    // up under ``@eval:``).
    validation_block: ($) =>
      seq(
        "@",
        field("kw", alias($.identifier, $.validation_keyword)),
        ":",
        field("procedure", $.identifier),
        ":",
        $._newline,
        field("body", $.validation_body),
      ),

    validation_body: ($) =>
      seq(
        $._indent,
        repeat1(
          choice($.test_case, $.eval_data, $.eval_test_data, $.eval_metric, $._newline),
        ),
        $._dedent,
      ),

    test_case: ($) =>
      seq(
        ">",
        "case",
        ":",
        field("name", $.identifier),
        ":",
        $._newline,
        field("code", $.test_case_body),
      ),

    test_case_body: ($) =>
      seq(
        $._indent,
        choice($.backtick_line_stmt, $.python_block),
        $._dedent,
      ),

    eval_data: ($) =>
      seq(
        ">",
        "data",
        ":",
        field("name", $.identifier),
        ":",
        $._newline,
        field("body", $.eval_entry_body),
      ),

    eval_test_data: ($) =>
      seq(
        ">",
        "test_data",
        ":",
        field("name", $.identifier),
        ":",
        $._newline,
        field("body", $.eval_entry_body),
      ),

    eval_metric: ($) =>
      seq(
        ">",
        "metric",
        ":",
        field("name", $.identifier),
        optional(seq("(", field("dataset", $.identifier), ")")),
        ":",
        $._newline,
        field("body", $.eval_entry_body),
      ),

    eval_entry_body: ($) =>
      seq(
        $._indent,
        choice($.return_stmt, $.return_block_stmt),
        $._dedent,
      ),

    // ============================================================
    // Lexical
    // ============================================================
    inline_python_expr: ($) => seq("`", field("code", $.python_inline_body), "`"),

    // Body of a backtick-delimited expression. Stops at the first
    // unescaped backtick. Backslash escapes (e.g. `\\\``) are consumed as
    // pairs. Must be non-empty to disambiguate from the empty `` `` token.
    python_inline_body: ($) => token.immediate(/([^`\\]|\\.)+/),

    identifier: ($) => /[A-Za-z_][A-Za-z0-9_]*/,

    line_comment: ($) => token(seq("#", /[^\n]*/)),

    block_comment: ($) =>
      token(seq("###", repeat(choice(/[^#]/, /#[^#]/, /##[^#]/)), "###")),
  },
});

function sep1(rule, sep) {
  return seq(rule, repeat(seq(sep, rule)));
}
