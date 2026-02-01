"""
Hybrid Lexer for Bayan Language
محلل معجمي هجين للغة بيان
"""

import re
from enum import Enum, auto

class TokenType(Enum):
    """Token types for Bayan language"""

    # Traditional tokens
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    OPERATOR = auto()
    ASSIGN = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    ARROW = auto()
    PIPE = auto()  # For list pattern [H|T]
    AT = auto()  # For decorators @

    # Keywords
    DEF = auto()
    CLASS = auto()
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    FOR = auto()
    WHILE = auto()
    IN = auto()
    PRINT = auto()
    RETURN = auto()
    YIELD = auto()
    BREAK = auto()
    CONTINUE = auto()
    PASS = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    SELF = auto()
    SUPER = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    TRY = auto()
    EXCEPT = auto()
    FINALLY = auto()
    RAISE = auto()
    WITH = auto()

    # Logical tokens
    PREDICATE = auto()
    VARIABLE = auto()
    IMPLIES = auto()
    UNIFY = auto()
    QUERY = auto()
    FACT = auto()
    RULE = auto()
    IS = auto()  # Arithmetic evaluation operator
    CUT = auto()  # Cut operator (!)

    # Async tokens
    ASYNC = auto()
    AWAIT = auto()

    # Hybrid tokens
    HYBRID = auto()
    ENTITY = auto()
    APPLY = auto()

    # Special
    NEWLINE = auto()
    WHITESPACE = auto()
    COMMENT = auto()
    EOF = auto()

class Token:
    """Represents a token"""
    def __init__(self, type_, value, line, column):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type.name}, {repr(self.value)}, {self.line}:{self.column})"

class HybridLexer:
    """Hybrid lexer for Bayan language"""

    KEYWORDS = {
        'def': TokenType.DEF,
        'class': TokenType.CLASS,
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'elif': TokenType.ELIF,
        'for': TokenType.FOR,
        'while': TokenType.WHILE,
        'in': TokenType.IN,
        'print': TokenType.PRINT,
        'return': TokenType.RETURN,
        'yield': TokenType.YIELD,
        'break': TokenType.BREAK,
        'continue': TokenType.CONTINUE,
        'pass': TokenType.PASS,
        'True': TokenType.TRUE,
        'False': TokenType.FALSE,
        'None': TokenType.NONE,
        'and': TokenType.AND,
        'or': TokenType.OR,
        'not': TokenType.NOT,
        'hybrid': TokenType.HYBRID,
        'query': TokenType.QUERY,
        'fact': TokenType.FACT,
        'rule': TokenType.RULE,
        'self': TokenType.SELF,
        'super': TokenType.SUPER,
        'import': TokenType.IMPORT,
        'from': TokenType.FROM,
        'as': TokenType.AS,
        'try': TokenType.TRY,
        'except': TokenType.EXCEPT,
        'finally': TokenType.FINALLY,
        'raise': TokenType.RAISE,
        'is': TokenType.IS,
        'async': TokenType.ASYNC,
        'await': TokenType.AWAIT,
        'with': TokenType.WITH,
        # Entity/Action keywords (bilingual)
        'entity': TokenType.ENTITY,
        'كيان': TokenType.ENTITY,
        'apply': TokenType.APPLY,
        'طبق': TokenType.APPLY,
    }

    def __init__(self, code):
        self.code = code
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []

    def tokenize(self):
        """Tokenize the code"""
        while self.position < len(self.code):
            self._skip_whitespace_and_comments()

            if self.position >= len(self.code):
                break

            if not self._match_token():
                char = self.code[self.position]
                raise SyntaxError(f"Unknown character '{char}' at {self.line}:{self.column}")

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens

    def _skip_whitespace_and_comments(self):
        """Skip whitespace and comments"""
        while self.position < len(self.code):
            # Skip whitespace
            if self.code[self.position] in ' \t\r':
                self.position += 1
                self.column += 1
            # Handle newlines
            elif self.code[self.position] == '\n':
                self.position += 1
                self.line += 1
                self.column = 1
            # Handle comments
            elif self.code[self.position] == '#':
                while self.position < len(self.code) and self.code[self.position] != '\n':
                    self.position += 1
            else:
                break


    def _match_triple_string(self):
        """Match a triple-quoted string supporting newlines."""
        if self.code.startswith('"""', self.position):
            quote = '"""'
        elif self.code.startswith("'''", self.position):
            quote = "'''"
        else:
            return False
        start_pos = self.position
        start_line = self.line
        start_col = self.column
        # Advance past opening quote
        self.position += 3
        self.column += 3
        while self.position < len(self.code):
            if self.code.startswith(quote, self.position):
                # Include closing quotes
                self.position += 3
                self.column += 3
                lexeme = self.code[start_pos:self.position]
                self.tokens.append(Token(TokenType.STRING, lexeme, start_line, start_col))
                return True
            ch = self.code[self.position]
            if ch == '\n':
                self.position += 1
                self.line += 1
                self.column = 1
            else:
                self.position += 1
                self.column += 1
        raise SyntaxError(f"Unterminated triple-quoted string at {start_line}:{start_col}")

    def _match_token(self):
        """Try to match a token at current position"""
        # Query operator (?-)
        if self._match_pattern(r'\?-', TokenType.QUERY):
            return True

        # Logical variable (?X, ?name)
        if self._match_pattern(r'\?[a-zA-Z_\u0600-\u06FF][a-zA-Z0-9_\u0600-\u06FF]*', TokenType.VARIABLE):
            return True

        # Strings (triple-quoted first)
        if self._match_triple_string():
            return True
        if self._match_pattern(r'"[^"]*"', TokenType.STRING) or self._match_pattern(r"'[^']*'", TokenType.STRING):
            return True

        # Numbers
        if self._match_pattern(r'\d+(\.\d+)?', TokenType.NUMBER):
            return True

        # Identifiers and keywords
        if self._match_pattern(r'[a-zA-Z_\u0600-\u06FF][a-zA-Z0-9_\u0600-\u06FF]*'):
            return True

        # Operators and symbols
        if self._match_pattern(r'←|:-', TokenType.IMPLIES):
            return True
        if self._match_pattern(r'==|!=|<=|>=|<|>', TokenType.OPERATOR):
            return True
        # Match ** before * to avoid splitting it
        if self._match_pattern(r'\*\*', TokenType.OPERATOR):
            return True
        if self._match_pattern(r'[+\-*/%]', TokenType.OPERATOR):
            return True
        if self._match_pattern(r'=', TokenType.ASSIGN):
            return True
        if self._match_pattern(r'\.', TokenType.DOT):
            return True
        if self._match_pattern(r',', TokenType.COMMA):
            return True
        if self._match_pattern(r';', TokenType.SEMICOLON):
            return True
        if self._match_pattern(r':', TokenType.COLON):
            return True
        if self._match_pattern(r'\(', TokenType.LPAREN):
            return True
        if self._match_pattern(r'\)', TokenType.RPAREN):
            return True
        if self._match_pattern(r'\{', TokenType.LBRACE):
            return True
        if self._match_pattern(r'\}', TokenType.RBRACE):
            return True
        if self._match_pattern(r'\[', TokenType.LBRACKET):
            return True
        if self._match_pattern(r'\]', TokenType.RBRACKET):
            return True
        if self._match_pattern(r'->', TokenType.ARROW):
            return True
        if self._match_pattern(r'\|', TokenType.PIPE):
            return True
        if self._match_pattern(r'@', TokenType.AT):
            return True
        if self._match_pattern(r'!', TokenType.CUT):
            return True

        return False

    def _match_pattern(self, pattern, token_type=None):
        """Try to match a regex pattern"""
        regex = re.compile(pattern)
        match = regex.match(self.code, self.position)

        if match:
            value = match.group(0)
            self.position = match.end()

            # Determine token type if not provided
            if token_type is None:
                if value.startswith('?'):
                    token_type = TokenType.VARIABLE
                elif value in self.KEYWORDS:
                    token_type = self.KEYWORDS[value]
                elif value[0] in '"\'':
                    token_type = TokenType.STRING
                elif value[0].isdigit():
                    token_type = TokenType.NUMBER
                else:
                    token_type = TokenType.IDENTIFIER

            self._add_token(token_type, value)
            self.column += len(value)
            return True

        return False

    def _add_token(self, token_type, value=None):
        """Add a token to the list"""
        if value is None:
            value = self.code[self.position:self.position + 1]

        token = Token(token_type, value, self.line, self.column)
        self.tokens.append(token)
