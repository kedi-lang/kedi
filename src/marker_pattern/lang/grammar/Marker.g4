// Placeholder grammar for the Marker language (ANTLR4)
// This is provided for future parser generation. The runtime currently uses a hand-rolled parser.

grammar Marker;

program         : (procedure | returnStmt | templateLine)* EOF ;

procedure       : AT IDENT LPAREN paramList? RPAREN COLON NEWLINE INDENT stmt+ DEDENT ;
stmt            : templateLine | returnStmt ;

templateLine    : templateExpr NEWLINE ;
returnStmt      : EQ templateExpr NEWLINE ;

templateExpr    : (textSegment | outputSegment | inputOrCall)+ ;

outputSegment   : LBRACK IDENT RBRACK ;
inputOrCall     : LT IDENT ( LPAREN argList? RPAREN )? GT ;

paramList       : IDENT ( COMMA IDENT )* ;
argList         : arg ( COMMA arg )* ;
arg             : inputOrCall | textAtom+ ;

textSegment     : textAtom+ ;
textAtom        : TEXT | ESC_LT | ESC_GT | ESC_LB | ESC_RB | ESC_EQ | ESC_AT | ESC_COMMA ;

AT      : '@' ;
EQ      : '=' ;
COLON   : ':' ;
LT      : '<' ;
GT      : '>' ;
LBRACK  : '[' ;
RBRACK  : ']' ;
LPAREN  : '(' ;
RPAREN  : ')' ;
COMMA   : ',' ;

IDENT   : [A-Za-z_][A-Za-z0-9_]* ;

ESC_LT      : '<<' ;
ESC_GT      : '>>' ;
ESC_LB      : '[[' ;
ESC_RB      : ']]' ;
ESC_EQ      : '==' ;
ESC_AT      : '@@' ;
ESC_COMMA   : ',,' ;

TEXT    : ~[\r\n<>\[\]=@]+ ;

NEWLINE : ('\r'? '\n')+ ;

// Pseudo tokens for indentation, to be handled by a custom lexer in generation
INDENT  : '<INDENT>' -> skip ;
DEDENT  : '<DEDENT>' -> skip ;

WS      : [ \t]+ -> skip ;


