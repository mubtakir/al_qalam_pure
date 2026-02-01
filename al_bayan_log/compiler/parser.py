"""
Hybrid Parser for Bayan Language
محلل نحوي هجين للغة بيان
"""

from .lexer import TokenType
from .ast_nodes import *
from .logical_engine import Term, Predicate, Fact, Rule

class HybridParser:
    """Hybrid parser for Bayan language"""

    def __init__(self, tokens, filename=None):
        self.tokens = tokens
        self.position = 0
        self.current_token = self.tokens[0] if tokens else None
        self.filename = filename

    def advance(self):
        """Move to the next token"""
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None

    def peek(self, lookahead=1):
        """Look ahead at the next token"""
        pos = self.position + lookahead
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def match(self, *token_types):
        """Check if current token matches any of the given types"""
        if self.current_token is None:
            return False
        return self.current_token.type in token_types

    def _unescape_string(self, s: str) -> str:
        """Unescape common escape sequences in string literals."""
        out = []
        i = 0
        L = len(s)
        while i < L:
            ch = s[i]
            if ch != '\\':
                out.append(ch)
                i += 1
                continue
            i += 1
            if i >= L:
                out.append('\\')
                break
            esc = s[i]
            i += 1
            if esc == 'n': out.append('\n')
            elif esc == 't': out.append('\t')
            elif esc == 'r': out.append('\r')
            elif esc == '"': out.append('"')
            elif esc == "'": out.append("'")
            elif esc == '\\': out.append('\\')
            elif esc == 'u' and i + 4 <= L:
                hex_digits = s[i:i+4]
                try:
                    out.append(chr(int(hex_digits, 16)))
                    i += 4
                except Exception:
                    out.append('\\u')
            else:
                out.append('\\' + esc)
        return ''.join(out)

    def peek_ahead(self, offset=1):
        """Peek ahead at token at position + offset"""
        peek_pos = self.position + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None

    def eat(self, token_type):
        """Consume a token of the given type"""
        if self.current_token is None or self.current_token.type != token_type:
            raise SyntaxError(
                f"Expected {token_type}, got {self.current_token.type if self.current_token else 'EOF'}"
            )
        token = self.current_token
        self.advance()
        return token
    def _with_pos(self, node, tok):
        """Attach source position from token to node if supported"""
        if node is not None and hasattr(node, 'with_pos') and tok is not None:
            node.with_pos(getattr(tok, 'line', None), getattr(tok, 'column', None), getattr(self, 'filename', None))
        return node


    def parse(self):
        """Parse the entire program"""
        statements = []

        while self.current_token and self.current_token.type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)

        return Program(statements)

    def parse_statement(self):
        """Parse a single statement"""
        # Check for decorators
        decorators = []
        while self.match(TokenType.AT):
            decorators.append(self.parse_decorator())

        if self.match(TokenType.HYBRID):
            return self.parse_hybrid_block()
        elif self.match(TokenType.ASYNC):
            return self.parse_async_function_def(decorators)
        elif self.match(TokenType.DEF):
            return self.parse_function_def(decorators)
        elif self.match(TokenType.CLASS):
            return self.parse_class_def(decorators)
        elif self.match(TokenType.IF):
            return self.parse_if_statement()
        elif self.match(TokenType.FOR):
            return self.parse_for_loop()
        elif self.match(TokenType.WHILE):
            return self.parse_while_loop()
        elif self.match(TokenType.RETURN):
            return self.parse_return_statement()
        elif self.match(TokenType.YIELD):
            return self.parse_yield_statement()
        elif self.match(TokenType.WITH):
            return self.parse_with_statement()
        elif self.match(TokenType.TRY):
            return self.parse_try_statement()
        elif self.match(TokenType.RAISE):
            return self.parse_raise_statement()
        elif self.match(TokenType.BREAK):
            self.advance()
            return BreakStatement()
        elif self.match(TokenType.CONTINUE):
            self.advance()
            return ContinueStatement()
        elif self.match(TokenType.PRINT):
            return self.parse_print_statement()
        elif self.match(TokenType.QUERY):
            return self.parse_query()
        elif self.match(TokenType.FACT):
            return self.parse_fact()
        elif self.match(TokenType.RULE):
            return self.parse_rule()
        elif self.match(TokenType.IMPORT):
            return self.parse_import_statement()
        elif self.match(TokenType.FROM):
            return self.parse_from_import_statement()
        elif self.match(TokenType.ENTITY):
            return self.parse_entity_def()
        elif self.match(TokenType.APPLY):
            return self.parse_apply_action()
        elif self.match(TokenType.IDENTIFIER):
            # Delegate to expression statement; it will also handle assignment forms
            return self.parse_expression_statement()
        elif self.match(TokenType.SELF):
            # Delegate to expression statement; it will handle self.x = v and self[i] = v
            return self.parse_expression_statement()
        else:
            return self.parse_expression_statement()

    def parse_hybrid_block(self):
        """Parse a hybrid block"""
        self.eat(TokenType.HYBRID)
        self.eat(TokenType.LBRACE)

        traditional_stmts = []
        logical_stmts = []

        while self.current_token and self.current_token.type != TokenType.RBRACE:
            if self.match(TokenType.QUERY):
                logical_stmts.append(self.parse_query())
            elif self.match(TokenType.RULE):
                logical_stmts.append(self.parse_rule())
            elif self.match(TokenType.FACT):
                logical_stmts.append(self.parse_fact())
            elif self.is_logical_rule():
                # Parse as logical rule
                logical_stmts.append(self.parse_rule())
            elif self.is_logical_fact():
                # Parse as logical fact
                logical_stmts.append(self.parse_fact())
            elif self.is_nominal_phrase():
                # Grammar sugar: nominal phrase like "محمد الطبيب." or with relation hint: "عصير العنب[of]."
                traditional_stmts.append(self.parse_nominal_phrase())
            else:
                stmt = self.parse_statement()
                if stmt:
                    traditional_stmts.append(stmt)

        self.eat(TokenType.RBRACE)
        return HybridBlock(traditional_stmts, logical_stmts)

    def is_nominal_phrase(self):
        """Lookahead for a nominal phrase statement requiring a trailing DOT.
        Pattern: (IDENT|STRING) (IDENT|STRING) [ '[' (IDENT|STRING) ']' ] '.'
        """
        t1 = self.current_token
        t2 = self.peek(1)
        if not t1 or not t2:
            return False
        if t1.type not in (TokenType.IDENTIFIER, TokenType.STRING):
            return False
        if t2.type not in (TokenType.IDENTIFIER, TokenType.STRING):
            return False
        t3 = self.peek(2)
        if t3 and t3.type == TokenType.DOT:
            return True
        if t3 and t3.type == TokenType.LBRACKET:
            t4 = self.peek(3)
            t5 = self.peek(4)
            t6 = self.peek(5)
            return (
                t4 and t4.type in (TokenType.IDENTIFIER, TokenType.STRING)
                and t5 and t5.type == TokenType.RBRACKET
                and t6 and t6.type == TokenType.DOT
            )
        return False

    def parse_nominal_phrase(self):
        """Parse a nominal phrase into a PhraseStatement AST node."""
        t1 = self.current_token
        self.advance()
        t2 = self.current_token
        self.advance()

        def tok_to_text(tok):
            if tok.type == TokenType.STRING:
                v = tok.value
                if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                    return v[1:-1]
                return v
            return tok.value

        a = tok_to_text(t1)
        b = tok_to_text(t2)
        relation = None
        if self.current_token and self.current_token.type == TokenType.LBRACKET:
            self.eat(TokenType.LBRACKET)
            rel_tok = self.current_token
            if rel_tok.type not in (TokenType.IDENTIFIER, TokenType.STRING):
                raise SyntaxError("Expected relation name inside [ ]")
            self.advance()
            rv = tok_to_text(rel_tok)
            self.eat(TokenType.RBRACKET)
            relation = rv
        self.eat(TokenType.DOT)
        text = f"{a} {b}"
        return PhraseStatement(text, relation)

    def is_logical_fact(self):
        """Check if the current token is a logical fact (identifier followed by parentheses and dot)"""
        if not self.current_token or self.current_token.type != TokenType.IDENTIFIER:
            return False

        # Look ahead to see if it's followed by ( and eventually )
        saved_pos = self.position
        try:
            self.advance()  # Skip identifier
            if self.current_token and self.current_token.type == TokenType.LPAREN:
                # Count parentheses to find the matching closing paren
                paren_count = 1
                self.advance()
                while self.current_token and paren_count > 0:
                    if self.current_token.type == TokenType.LPAREN:
                        paren_count += 1
                    elif self.current_token.type == TokenType.RPAREN:
                        paren_count -= 1
                    self.advance()

                # Check if followed by DOT
                result = self.current_token and self.current_token.type == TokenType.DOT
                self.position = saved_pos
                self.current_token = self.tokens[self.position] if self.position < len(self.tokens) else None
                return result
        except:
            pass

        self.position = saved_pos
        self.current_token = self.tokens[self.position] if self.position < len(self.tokens) else None
        return False

    def is_logical_rule(self):
        """Check if the current token is a logical rule (predicate followed by :- and dot)"""
        if not self.current_token or self.current_token.type != TokenType.IDENTIFIER:
            return False

        # Look ahead to see if it's followed by ( and eventually :- and )
        saved_pos = self.position
        try:
            self.advance()  # Skip identifier
            if self.current_token and self.current_token.type == TokenType.LPAREN:
                # Count parentheses to find the matching closing paren
                paren_count = 1
                self.advance()
                while self.current_token and paren_count > 0:
                    if self.current_token.type == TokenType.LPAREN:
                        paren_count += 1
                    elif self.current_token.type == TokenType.RPAREN:
                        paren_count -= 1
                    self.advance()

                # Check if followed by IMPLIES (:-) or ARROW (←)
                result = self.current_token and self.current_token.type in (TokenType.IMPLIES, TokenType.ARROW)
                self.position = saved_pos
                self.current_token = self.tokens[self.position] if self.position < len(self.tokens) else None
                return result
        except:
            pass

        self.position = saved_pos
        self.current_token = self.tokens[self.position] if self.position < len(self.tokens) else None
        return False

    def parse_function_def(self, decorators=None):
        """Parse a function definition"""
        if decorators is None:
            decorators = []
        def_tok = self.eat(TokenType.DEF)
        name = self.eat(TokenType.IDENTIFIER).value
        self.eat(TokenType.LPAREN)
        params = self.parse_parameter_list()
        self.eat(TokenType.RPAREN)
        if self.match(TokenType.COLON):
            self.eat(TokenType.COLON)
        body = self.parse_block()

        return self._with_pos(FunctionDef(name, params, body, decorators), def_tok)

    def parse_decorator(self):
        """Parse a decorator: @name or @name(args)"""
        at_tok = self.eat(TokenType.AT)
        name = self.eat(TokenType.IDENTIFIER).value

        args = []
        if self.match(TokenType.LPAREN):
            self.eat(TokenType.LPAREN)
            if not self.match(TokenType.RPAREN):
                args.append(self.parse_expression())
                while self.match(TokenType.COMMA):
                    self.eat(TokenType.COMMA)
                    args.append(self.parse_expression())
            self.eat(TokenType.RPAREN)

        return self._with_pos(Decorator(name, args), at_tok)

    def parse_async_function_def(self, decorators=None):
        """Parse an async function definition: async def name(params): body"""
        if decorators is None:
            decorators = []
        async_tok = self.eat(TokenType.ASYNC)
        self.eat(TokenType.DEF)
        name = self.eat(TokenType.IDENTIFIER).value
        self.eat(TokenType.LPAREN)
        params = self.parse_parameter_list()
        self.eat(TokenType.RPAREN)
        self.eat(TokenType.COLON)
        body = self.parse_block()

        func_def = AsyncFunctionDef(name, params, body)
        func_def.decorators = decorators
        return self._with_pos(func_def, async_tok)

    def parse_parameter_list(self):
        """Parse function parameters with support for default values, *args, and **kwargs"""
        params = []

        if not self.match(TokenType.RPAREN):
            # Parse first parameter
            is_varargs = False
            is_kwargs = False

            # Check for **kwargs or *args
            if self.match(TokenType.OPERATOR):
                if self.current_token.value == '**':
                    self.advance()  # consume **
                    is_kwargs = True
                elif self.current_token.value == '*':
                    self.advance()  # consume *
                    is_varargs = True

            # Get parameter name (could be 'self' or identifier)
            if self.match(TokenType.SELF):
                param_name = self.eat(TokenType.SELF).value
            else:
                param_name = self.eat(TokenType.IDENTIFIER).value

            default_value = None

            # Check for default value (not allowed for *args/**kwargs)
            if not is_varargs and not is_kwargs and self.match(TokenType.ASSIGN):
                self.eat(TokenType.ASSIGN)
                default_value = self.parse_expression()

            params.append(Parameter(param_name, default_value, is_varargs, is_kwargs))

            # Parse remaining parameters
            while self.match(TokenType.COMMA):
                self.eat(TokenType.COMMA)

                is_varargs = False
                is_kwargs = False

                # Check for **kwargs or *args
                if self.match(TokenType.OPERATOR):
                    if self.current_token.value == '**':
                        self.advance()  # consume **
                        is_kwargs = True
                    elif self.current_token.value == '*':
                        self.advance()  # consume *
                        is_varargs = True

                # Could be 'self' or identifier
                if self.match(TokenType.SELF):
                    param_name = self.eat(TokenType.SELF).value
                else:
                    param_name = self.eat(TokenType.IDENTIFIER).value

                default_value = None

                # Check for default value (not allowed for *args/**kwargs)
                if not is_varargs and not is_kwargs and self.match(TokenType.ASSIGN):
                    self.eat(TokenType.ASSIGN)
                    default_value = self.parse_expression()

                params.append(Parameter(param_name, default_value, is_varargs, is_kwargs))

        return params

    def parse_block(self):
        """Parse a block of statements"""
        self.eat(TokenType.LBRACE)
        statements = []

        while self.current_token and self.current_token.type != TokenType.RBRACE:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)

        self.eat(TokenType.RBRACE)
        return Block(statements)

    def parse_class_def(self, decorators=None):
        """Parse a class definition"""
        if decorators is None:
            decorators = []
        class_tok = self.eat(TokenType.CLASS)
        name = self.eat(TokenType.IDENTIFIER).value

        base_class = None
        base_classes = None
        if self.match(TokenType.LPAREN):
            self.eat(TokenType.LPAREN)
            bases = []
            # Allow zero or more bases? We'll require at least one if parens present
            bases.append(self.eat(TokenType.IDENTIFIER).value)
            while self.match(TokenType.COMMA):
                self.eat(TokenType.COMMA)
                bases.append(self.eat(TokenType.IDENTIFIER).value)
            self.eat(TokenType.RPAREN)
            base_classes = bases
            base_class = bases[0] if bases else None

        if self.match(TokenType.COLON):
            self.eat(TokenType.COLON)
        body = self.parse_block()

        return self._with_pos(ClassDef(name, base_class, body, base_classes=base_classes, decorators=decorators), class_tok)

    def parse_if_statement(self):
        """Parse an if statement with optional elif chain and else"""
        if_tok = self.eat(TokenType.IF)
        condition = self.parse_expression()
        if self.match(TokenType.COLON):
            self.eat(TokenType.COLON)
        then_branch = self.parse_block()

        # Collect zero or more elif branches
        elif_branches = []
        while self.match(TokenType.ELIF):
            self.eat(TokenType.ELIF)
            elif_cond = self.parse_expression()
            if self.match(TokenType.COLON):
                self.eat(TokenType.COLON)
            elif_block = self.parse_block()
            elif_branches.append((elif_cond, elif_block))

        # Optional else branch
        else_branch = None
        if self.match(TokenType.ELSE):
            self.eat(TokenType.ELSE)
            if self.match(TokenType.COLON):
                self.eat(TokenType.COLON)
            else_branch = self.parse_block()

        # Build nested IfStatements for elif chain (right-associative)
        current_else = else_branch
        for cond, block in reversed(elif_branches):
            current_else = self._with_pos(IfStatement(cond, block, current_else), if_tok)

        return self._with_pos(IfStatement(condition, then_branch, current_else), if_tok)

    def parse_for_loop(self):
        """Parse a for loop"""
        for_tok = self.eat(TokenType.FOR)
        var_name = self.eat(TokenType.IDENTIFIER).value
        self.eat(TokenType.IN)
        iterable = self.parse_expression()
        if self.match(TokenType.COLON):
            self.eat(TokenType.COLON)
        body = self.parse_block()

        return self._with_pos(ForLoop(var_name, iterable, body), for_tok)

    def parse_while_loop(self):
        """Parse a while loop"""
        while_tok = self.eat(TokenType.WHILE)
        condition = self.parse_expression()
        if self.match(TokenType.COLON):
            self.eat(TokenType.COLON)
        body = self.parse_block()

        return self._with_pos(WhileLoop(condition, body), while_tok)

    def parse_return_statement(self):
        """Parse a return statement"""
        ret_tok = self.eat(TokenType.RETURN)
        value = None
        if not self.match(TokenType.SEMICOLON, TokenType.RBRACE, TokenType.EOF):
            value = self.parse_expression()
        return self._with_pos(ReturnStatement(value), ret_tok)

    def parse_yield_statement(self):
        """Parse a yield statement"""
        yield_tok = self.eat(TokenType.YIELD)
        value = None
        if not self.match(TokenType.SEMICOLON, TokenType.RBRACE, TokenType.EOF):
            value = self.parse_expression()
        return self._with_pos(YieldExpr(value), yield_tok)

    def parse_with_statement(self):
        """Parse a with statement: with expr as var: { body }"""
        with_tok = self.eat(TokenType.WITH)
        context_expr = self.parse_expression()

        target_var = None
        if self.match(TokenType.AS):
            self.eat(TokenType.AS)
            target_var = self.eat(TokenType.IDENTIFIER).value

        self.eat(TokenType.COLON)
        body = self.parse_block()

        return self._with_pos(WithStatement(context_expr, target_var, body), with_tok)

    def parse_raise_statement(self):
        """Parse a raise statement"""
        raise_tok = self.eat(TokenType.RAISE)
        value = None
        if not self.match(TokenType.SEMICOLON, TokenType.RBRACE, TokenType.EOF):
            value = self.parse_expression()
        return self._with_pos(RaiseStatement(value), raise_tok)

    def parse_try_statement(self):
        """Parse a try/except[/except...][finally] statement"""
        try_tok = self.eat(TokenType.TRY)
        self.eat(TokenType.COLON)
        try_block = self.parse_block()

        handlers = []
        finally_block = None

        # One or more except blocks
        while self.match(TokenType.EXCEPT):
            self.eat(TokenType.EXCEPT)
            type_name = None
            alias = None
            # except : (catch-all)
            if not self.match(TokenType.COLON):
                # except Type [as name] :
                type_name = self.eat(TokenType.IDENTIFIER).value
                if self.match(TokenType.AS):
                    self.eat(TokenType.AS)
                    alias = self.eat(TokenType.IDENTIFIER).value
            self.eat(TokenType.COLON)
            body = self.parse_block()
            handlers.append(self._with_pos(ExceptHandler(type_name, alias, body), try_tok))

        # Optional finally
        if self.match(TokenType.FINALLY):
            self.eat(TokenType.FINALLY)
            self.eat(TokenType.COLON)
            finally_block = self.parse_block()

        if not handlers and not finally_block:
            raise SyntaxError("'try' must be followed by at least one 'except' or a 'finally'")

        return self._with_pos(TryExceptFinally(try_block, handlers, finally_block), try_tok)

    def parse_print_statement(self):
        """Parse a print statement"""
        pr_tok = self.eat(TokenType.PRINT)
        self.eat(TokenType.LPAREN)
        value = self.parse_expression()
        self.eat(TokenType.RPAREN)

        return self._with_pos(PrintStatement(value), pr_tok)

    def parse_assignment(self):
        """Parse an assignment target and value
        Supports: x = v, obj.attr = v, obj[index] = v, and chained obj.attr[index] = v
        """
        # Parse the left-hand side as a primary expression
        # This allows for simple variables like 'x'
        name_tok = self.current_token
        if not self.match(TokenType.IDENTIFIER):
            raise SyntaxError(f"Expected identifier in assignment, got {self.current_token}")

        base_name = self.eat(TokenType.IDENTIFIER).value
        target_expr = self._with_pos(Variable(base_name), name_tok)

        # Parse chained attribute/subscript access on the target
        while self.match(TokenType.DOT) or self.match(TokenType.LBRACKET):
            if self.match(TokenType.DOT):
                self.eat(TokenType.DOT)
                attr_tok = self.eat(TokenType.IDENTIFIER)
                attr_name = attr_tok.value
                target_expr = self._with_pos(AttributeAccess(target_expr, attr_name), attr_tok)
            else:
                lb_tok = self.eat(TokenType.LBRACKET)
                index_expr = self.parse_subscript_or_slice()
                self.eat(TokenType.RBRACKET)
                target_expr = self._with_pos(SubscriptAccess(target_expr, index_expr), lb_tok)

        # Now parse assignment operator and value
        if not self.match(TokenType.ASSIGN):
            raise SyntaxError(f"Expected '=' in assignment, got {self.current_token}")

        assign_tok = self.eat(TokenType.ASSIGN)
        value = self.parse_expression()

        # Build appropriate assignment node
        if isinstance(target_expr, Variable):
            return self._with_pos(Assignment(target_expr.name, value), name_tok)
        elif isinstance(target_expr, AttributeAccess):
            return self._with_pos(AttributeAssignment(target_expr.object_expr, target_expr.attribute_name, value), assign_tok)
        elif isinstance(target_expr, SubscriptAccess):
            return self._with_pos(SubscriptAssignment(target_expr.object_expr, target_expr.index_expr, value), assign_tok)
        else:
            raise SyntaxError("Invalid assignment target")

    def parse_expression_statement(self):
        """Parse an expression statement or assignment"""
        expr = self.parse_expression()

        # Check if this is a logical fact or rule (ends with DOT or has IMPLIES)
        if self.match(TokenType.DOT):
            # This is a fact: predicate(args).
            self.eat(TokenType.DOT)
            # Convert FunctionCall to Predicate
            if isinstance(expr, FunctionCall):
                from .ast_nodes import Predicate, LogicalFact
                predicate = Predicate(expr.name, expr.arguments)
                return LogicalFact(predicate)
            return expr
        elif self.match(TokenType.IMPLIES):
            # This is a rule: head :- body.
            self.eat(TokenType.IMPLIES)
            body = self.parse_logical_body()
            if self.match(TokenType.DOT):
                self.eat(TokenType.DOT)
            # Convert FunctionCall to Predicate
            if isinstance(expr, FunctionCall):
                from .ast_nodes import Predicate, LogicalRule
                head = Predicate(expr.name, expr.arguments)
                return LogicalRule(head, body)
            return expr
        # Check if this is an assignment (expr = value)
        elif self.match(TokenType.ASSIGN):
            self.eat(TokenType.ASSIGN)
            value = self.parse_expression()

            # Build appropriate assignment node
            if isinstance(expr, Variable):
                return Assignment(expr.name, value)
            elif isinstance(expr, AttributeAccess):
                return AttributeAssignment(expr.object_expr, expr.attribute_name, value)
            elif isinstance(expr, SubscriptAccess):
                return SubscriptAssignment(expr.object_expr, expr.index_expr, value)
            else:
                raise SyntaxError(f"Invalid assignment target: {expr}")

        return expr

    def parse_expression(self):
        """Parse an expression"""
        return self.parse_or_expression()

    def parse_or_expression(self):
        """Parse logical OR expression"""
        left = self.parse_and_expression()

        while self.match(TokenType.OR):
            tok = self.current_token
            self.advance()
            right = self.parse_and_expression()
            left = self._with_pos(BinaryOp('or', left, right), tok)

        return left

    def parse_and_expression(self):
        """Parse logical AND expression"""
        left = self.parse_comparison()

        while self.match(TokenType.AND):
            tok = self.current_token
            self.advance()
            right = self.parse_comparison()
            left = self._with_pos(BinaryOp('and', left, right), tok)

        return left

    def parse_comparison(self):
        """Parse comparison expression"""
        left = self.parse_additive()

        while self.match(TokenType.OPERATOR) or self.match(TokenType.IN):
            if self.match(TokenType.IN):
                tok_in = self.eat(TokenType.IN)
                right = self.parse_additive()
                left = self._with_pos(BinaryOp('in', left, right), tok_in)
            else:
                op_tok = self.eat(TokenType.OPERATOR)
                op = op_tok.value
                right = self.parse_additive()
                left = self._with_pos(BinaryOp(op, left, right), op_tok)

        return left

    def parse_additive(self):
        """Parse addition/subtraction"""
        left = self.parse_multiplicative()

        while self.match(TokenType.OPERATOR) and self.current_token.value in ['+', '-']:
            op_tok = self.eat(TokenType.OPERATOR)
            op = op_tok.value
            right = self.parse_multiplicative()
            left = self._with_pos(BinaryOp(op, left, right), op_tok)

        return left

    def parse_multiplicative(self):
        """Parse multiplication/division"""
        left = self.parse_unary()

        while self.match(TokenType.OPERATOR) and self.current_token.value in ['*', '/', '%']:
            op_tok = self.eat(TokenType.OPERATOR)
            op = op_tok.value
            right = self.parse_unary()
            left = self._with_pos(BinaryOp(op, left, right), op_tok)

        return left

    def parse_unary(self):
        """Parse unary expressions"""
        if self.match(TokenType.NOT):
            tok = self.current_token
            self.advance()
            operand = self.parse_unary()
            return self._with_pos(UnaryOp('not', operand), tok)
        elif self.match(TokenType.OPERATOR) and self.current_token.value == '-':
            tok = self.current_token
            self.advance()
            operand = self.parse_unary()
            return self._with_pos(UnaryOp('-', operand), tok)

        return self.parse_primary()

    def parse_primary(self):
        """Parse primary expressions"""
        if self.match(TokenType.AWAIT):
            await_tok = self.eat(TokenType.AWAIT)
            expr = self.parse_primary()
            return self._with_pos(AwaitExpr(expr), await_tok)

        elif self.match(TokenType.NUMBER):
            tok = self.eat(TokenType.NUMBER)
            value = tok.value
            return self._with_pos(Number(float(value) if '.' in value else int(value)), tok)

        elif self.match(TokenType.STRING):
            tok = self.eat(TokenType.STRING)
            value = tok.value
            # Remove quotes (support triple-quoted strings)
            if (value.startswith('"""') and value.endswith('"""')) or (value.startswith("'''") and value.endswith("'''")):
                inner = value[3:-3]
            else:
                inner = value[1:-1]
            inner = self._unescape_string(inner)
            return self._with_pos(String(inner), tok)

        elif self.match(TokenType.TRUE):
            tok = self.eat(TokenType.TRUE)
            return self._with_pos(Boolean(True), tok)

        elif self.match(TokenType.FALSE):
            tok = self.eat(TokenType.FALSE)
            return self._with_pos(Boolean(False), tok)

        elif self.match(TokenType.NONE):
            tok = self.eat(TokenType.NONE)
            return self._with_pos(Variable('None'), tok)

        elif self.match(TokenType.VARIABLE):
            # Logical variable
            tok = self.eat(TokenType.VARIABLE)
            name = tok.value
            return self._with_pos(Variable(name), tok)

        elif self.match(TokenType.IDENTIFIER):
            name_tok = self.eat(TokenType.IDENTIFIER)
            name = name_tok.value

            if self.match(TokenType.LPAREN):
                # Function call
                self.eat(TokenType.LPAREN)
                args, named_args = self.parse_argument_list()
                self.eat(TokenType.RPAREN)
                expr = self._with_pos(FunctionCall(name, args, named_args), name_tok)

                # Check for chained access: . or [ ] after a function call result
                while self.match(TokenType.DOT) or self.match(TokenType.LBRACKET):
                    if self.match(TokenType.DOT):
                        self.eat(TokenType.DOT)
                        attr_tok = self.eat(TokenType.IDENTIFIER)
                        attr_name = attr_tok.value

                        if self.match(TokenType.LPAREN):
                            self.eat(TokenType.LPAREN)
                            args, named_args = self.parse_argument_list()
                            self.eat(TokenType.RPAREN)
                            expr = self._with_pos(MethodCall(expr, attr_name, args, named_args), attr_tok)
                        else:
                            expr = self._with_pos(AttributeAccess(expr, attr_name), attr_tok)
                    else:
                        # Subscription: expr[index] or expr[start:end:step]
                        lb_tok = self.eat(TokenType.LBRACKET)
                        index_expr = self.parse_subscript_or_slice()
                        self.eat(TokenType.RBRACKET)
                        expr = self._with_pos(SubscriptAccess(expr, index_expr), lb_tok)

                return expr
            elif self.match(TokenType.DOT) or self.match(TokenType.LBRACKET):
                # Attribute access or method call starting from a variable
                expr = self._with_pos(Variable(name), name_tok)

                while self.match(TokenType.DOT) or self.match(TokenType.LBRACKET):
                    if self.match(TokenType.DOT):
                        self.eat(TokenType.DOT)
                        attr_tok = self.eat(TokenType.IDENTIFIER)
                        attr_name = attr_tok.value

                        if self.match(TokenType.LPAREN):
                            # Method call
                            self.eat(TokenType.LPAREN)
                            args, named_args = self.parse_argument_list()
                            self.eat(TokenType.RPAREN)
                            expr = self._with_pos(MethodCall(expr, attr_name, args, named_args), attr_tok)
                        else:
                            # Attribute access
                            expr = self._with_pos(AttributeAccess(expr, attr_name), attr_tok)
                    else:
                        # Subscription: expr[index] or expr[start:end:step]
                        lb_tok = self.eat(TokenType.LBRACKET)
                        index_expr = self.parse_subscript_or_slice()
                        self.eat(TokenType.RBRACKET)
                        expr = self._with_pos(SubscriptAccess(expr, index_expr), lb_tok)

                return expr
            else:
                return self._with_pos(Variable(name), name_tok)

        elif self.match(TokenType.LBRACKET):
            return self.parse_list()

        elif self.match(TokenType.LBRACE):
            return self.parse_dict()

        elif self.match(TokenType.SELF):
            tok_self = self.current_token
            self.advance()
            expr = self._with_pos(SelfReference(), tok_self)

            # Allow chained access after self: . or [ ]
            while self.match(TokenType.DOT) or self.match(TokenType.LBRACKET):
                if self.match(TokenType.DOT):
                    self.eat(TokenType.DOT)
                    attr_tok = self.eat(TokenType.IDENTIFIER)
                    attr_name = attr_tok.value

                    if self.match(TokenType.LPAREN):
                        # Method call
                        self.eat(TokenType.LPAREN)
                        args, named_args = self.parse_argument_list()
                        self.eat(TokenType.RPAREN)
                        expr = self._with_pos(MethodCall(expr, attr_name, args, named_args), attr_tok)
                    else:
                        # Attribute access
                        expr = self._with_pos(AttributeAccess(expr, attr_name), attr_tok)
                else:
                    # Subscription: self[index] or self[start:end:step]
                    lb_tok = self.eat(TokenType.LBRACKET)
                    index_expr = self.parse_subscript_or_slice()
                    self.eat(TokenType.RBRACKET)
                    expr = self._with_pos(SubscriptAccess(expr, index_expr), lb_tok)

            return expr

        elif self.match(TokenType.SUPER):
            self.eat(TokenType.SUPER)
            self.eat(TokenType.LPAREN)
            # Support both: super(method[, args]) and super().method(args)
            if self.match(TokenType.RPAREN):
                # super().method(args)
                self.eat(TokenType.RPAREN)
                self.eat(TokenType.DOT)
                method_name = self.eat(TokenType.IDENTIFIER).value
                self.eat(TokenType.LPAREN)
                args, named_args = self.parse_argument_list()
                self.eat(TokenType.RPAREN)
                return SuperCall(method_name, args, named_args)
            else:
                # Legacy form: super(method[, args])
                method_name = self.eat(TokenType.IDENTIFIER).value
                args = []
                named_args = {}
                if self.match(TokenType.COMMA):
                    self.eat(TokenType.COMMA)
                    args, named_args = self.parse_argument_list()
                self.eat(TokenType.RPAREN)
                return SuperCall(method_name, args, named_args)

        elif self.match(TokenType.LPAREN):
            lp_tok = self.eat(TokenType.LPAREN)

            # Empty tuple: ()
            if self.match(TokenType.RPAREN):
                self.eat(TokenType.RPAREN)
                return self._with_pos(Tuple([]), lp_tok)

            # Parse first expression
            first_expr = self.parse_expression()

            # Check if this is a tuple (has comma) or just grouped expression
            if self.match(TokenType.COMMA):
                # This is a tuple
                elements = [first_expr]
                while self.match(TokenType.COMMA):
                    self.eat(TokenType.COMMA)
                    # Allow trailing comma
                    if self.match(TokenType.RPAREN):
                        break
                    elements.append(self.parse_expression())
                self.eat(TokenType.RPAREN)
                return self._with_pos(Tuple(elements), lp_tok)
            else:
                # Just a grouped expression
                self.eat(TokenType.RPAREN)
                expr = first_expr

                # Check for chained access: . or [ ] after a parenthesized expression
                while self.match(TokenType.DOT) or self.match(TokenType.LBRACKET):
                    if self.match(TokenType.DOT):
                        self.eat(TokenType.DOT)
                        attr_tok = self.eat(TokenType.IDENTIFIER)
                        attr_name = attr_tok.value

                        if self.match(TokenType.LPAREN):
                            self.eat(TokenType.LPAREN)
                            args, named_args = self.parse_argument_list()
                            self.eat(TokenType.RPAREN)
                            expr = self._with_pos(MethodCall(expr, attr_name, args, named_args), attr_tok)
                        else:
                            expr = self._with_pos(AttributeAccess(expr, attr_name), attr_tok)
                    else:
                        # Subscription: expr[index] or expr[start:end:step]
                        lb_tok = self.eat(TokenType.LBRACKET)
                        index_expr = self.parse_subscript_or_slice()
                        self.eat(TokenType.RBRACKET)
                        expr = self._with_pos(SubscriptAccess(expr, index_expr), lb_tok)

                return expr

        else:
            raise SyntaxError(f"Unexpected token: {self.current_token}")

    def parse_argument_list(self):
        """Parse function arguments with support for named arguments"""
        args = []
        named_args = {}

        if not self.match(TokenType.RPAREN):
            # Parse first argument
            # Check if it's a named argument (identifier followed by =)
            if self.match(TokenType.IDENTIFIER) and self.peek() and self.peek().type == TokenType.ASSIGN:
                name = self.eat(TokenType.IDENTIFIER).value
                self.eat(TokenType.ASSIGN)
                value = self.parse_expression()
                named_args[name] = value
            else:
                args.append(self.parse_expression())

            # Parse remaining arguments
            while self.match(TokenType.COMMA):
                self.eat(TokenType.COMMA)

                # Check if it's a named argument
                if self.match(TokenType.IDENTIFIER) and self.peek() and self.peek().type == TokenType.ASSIGN:
                    name = self.eat(TokenType.IDENTIFIER).value
                    self.eat(TokenType.ASSIGN)
                    value = self.parse_expression()
                    named_args[name] = value
                else:
                    args.append(self.parse_expression())

        return args, named_args

    def parse_dotted_name(self):
        """Parse dotted module name like 'a.b.c'"""
        parts = [self.eat(TokenType.IDENTIFIER).value]
        while self.match(TokenType.DOT):
            self.eat(TokenType.DOT)
            parts.append(self.eat(TokenType.IDENTIFIER).value)
        return '.'.join(parts)

    def parse_import_statement(self):
        """Parse 'import module [as alias]'"""
        imp_tok = self.eat(TokenType.IMPORT)
        module_name = self.parse_dotted_name()
        alias = None
        if self.match(TokenType.AS):
            self.eat(TokenType.AS)
            alias = self.eat(TokenType.IDENTIFIER).value
        return self._with_pos(ImportStatement(module_name, alias), imp_tok)

    def parse_from_import_statement(self):
        """Parse 'from module import name[, name]*'"""
        from_tok = self.eat(TokenType.FROM)
        module_name = self.parse_dotted_name()
        self.eat(TokenType.IMPORT)
        names = [self.eat(TokenType.IDENTIFIER).value]
        while self.match(TokenType.COMMA):
            self.eat(TokenType.COMMA)
            names.append(self.eat(TokenType.IDENTIFIER).value)
        return self._with_pos(FromImportStatement(module_name, names), from_tok)

    def parse_list(self):
        """Parse a list literal, list comprehension, or list pattern [H|T]"""
        lb_tok = self.eat(TokenType.LBRACKET)
        elements = []

        if not self.match(TokenType.RBRACKET):
            first_expr = self.parse_expression()

            # Check for list pattern [H|T] or [H1, H2, ...|T]
            if self.match(TokenType.PIPE):
                self.eat(TokenType.PIPE)
                tail = self.parse_expression()
                self.eat(TokenType.RBRACKET)
                # Create a ListPattern node with head elements and tail
                return self._with_pos(ListPattern([first_expr], tail), lb_tok)

            # Check for list comprehension: [expr for var in iterable (if cond)?]
            elif self.match(TokenType.FOR):
                self.eat(TokenType.FOR)
                var_name = self.eat(TokenType.IDENTIFIER).value
                self.eat(TokenType.IN)
                iterable = self.parse_expression()
                condition = None
                if self.match(TokenType.IF):
                    self.eat(TokenType.IF)
                    condition = self.parse_expression()
                self.eat(TokenType.RBRACKET)
                return self._with_pos(ListComprehension(first_expr, var_name, iterable, condition), lb_tok)
            else:
                elements.append(first_expr)
                while self.match(TokenType.COMMA):
                    self.eat(TokenType.COMMA)
                    # Check if this is [H1, H2, ...|T] pattern (after comma)
                    if self.match(TokenType.PIPE):
                        self.eat(TokenType.PIPE)
                        tail = self.parse_expression()
                        self.eat(TokenType.RBRACKET)
                        return self._with_pos(ListPattern(elements, tail), lb_tok)
                    # Otherwise, parse next element
                    next_expr = self.parse_expression()
                    elements.append(next_expr)
                    # Check if pipe comes after this element
                    if self.match(TokenType.PIPE):
                        self.eat(TokenType.PIPE)
                        tail = self.parse_expression()
                        self.eat(TokenType.RBRACKET)
                        return self._with_pos(ListPattern(elements, tail), lb_tok)

        self.eat(TokenType.RBRACKET)
        return self._with_pos(List(elements), lb_tok)

    def parse_dict(self):
        """Parse a dictionary literal or set literal"""
        lb_tok = self.eat(TokenType.LBRACE)

        # Empty dict: {}
        if self.match(TokenType.RBRACE):
            self.eat(TokenType.RBRACE)
            return self._with_pos(Dict([]), lb_tok)

        # Parse first expression
        first_expr = self.parse_expression()

        # Check if this is a dict (has colon) or set (has comma or just one element)
        if self.match(TokenType.COLON):
            # This is a dictionary
            pairs = []
            self.eat(TokenType.COLON)
            value = self.parse_expression()
            pairs.append((first_expr, value))

            while self.match(TokenType.COMMA):
                self.eat(TokenType.COMMA)
                # Allow trailing comma
                if self.match(TokenType.RBRACE):
                    break
                key = self.parse_expression()
                self.eat(TokenType.COLON)
                value = self.parse_expression()
                pairs.append((key, value))

            self.eat(TokenType.RBRACE)
            return self._with_pos(Dict(pairs), lb_tok)
        else:
            # This is a set
            elements = [first_expr]
            while self.match(TokenType.COMMA):
                self.eat(TokenType.COMMA)
                # Allow trailing comma
                if self.match(TokenType.RBRACE):
                    break
                elements.append(self.parse_expression())

            self.eat(TokenType.RBRACE)
            return self._with_pos(Set(elements), lb_tok)

    def parse_query(self):
        """Parse a logical query"""
        self.eat(TokenType.QUERY)
        goal = self.parse_logical_predicate()
        if self.current_token and self.current_token.type == TokenType.DOT:
            self.eat(TokenType.DOT)
        return LogicalQuery(goal)

    def parse_fact(self):
        """Parse a logical fact"""
        if self.match(TokenType.FACT):
            self.eat(TokenType.FACT)

        predicate = self.parse_logical_predicate()
        self.eat(TokenType.DOT)

        return LogicalFact(predicate)

    def parse_rule(self):
        """Parse a logical rule"""
        if self.match(TokenType.RULE):
            self.eat(TokenType.RULE)

        head = self.parse_logical_predicate()

        # Handle both :- and ←
        if self.match(TokenType.IMPLIES):
            self.eat(TokenType.IMPLIES)
        elif self.match(TokenType.ARROW):
            self.eat(TokenType.ARROW)
        else:
            raise SyntaxError(f"Expected :- or ← in rule, got {self.current_token}")

        body = self.parse_logical_body()

        # Handle optional dot at the end
        if self.current_token and self.current_token.type == TokenType.DOT:
            self.eat(TokenType.DOT)

        return LogicalRule(head, body)

    def parse_entity_def(self):
        """Parse an entity definition: entity <name> { ... }"""
        ent_tok = self.eat(TokenType.ENTITY)
        name = self.eat(TokenType.IDENTIFIER).value
        # Entity body is a dict-like structure
        body = self.parse_dict()
        return self._with_pos(EntityDef(name, body), ent_tok)

    def parse_apply_action(self):
        """Parse apply statement: apply <actor>.<action>(<target>, [name=expr, ...])"""
        app_tok = self.eat(TokenType.APPLY)
        actor_name = self.eat(TokenType.IDENTIFIER).value
        self.eat(TokenType.DOT)
        action_name = self.eat(TokenType.IDENTIFIER).value
        self.eat(TokenType.LPAREN)
        args, named_args = self.parse_argument_list()
        self.eat(TokenType.RPAREN)
        if not args:
            raise SyntaxError("apply requires at least a target argument")
        target_expr = args[0]
        return self._with_pos(ApplyActionStmt(actor_name, action_name, target_expr, named_args), app_tok)


    def parse_logical_predicate(self):
        """Parse a logical predicate"""
        name = self.eat(TokenType.IDENTIFIER).value
        self.eat(TokenType.LPAREN)
        args = self.parse_logical_arguments()
        self.eat(TokenType.RPAREN)

        return Predicate(name, args)

    def parse_logical_arguments(self):
        """Parse logical predicate arguments"""
        args = []

        if not self.match(TokenType.RPAREN):
            args.append(self.parse_logical_term())

            while self.match(TokenType.COMMA):
                self.eat(TokenType.COMMA)
                args.append(self.parse_logical_term())

        return args

    def parse_logical_term(self):
        """Parse a logical term (including list patterns)"""
        if self.match(TokenType.VARIABLE):
            var_name = self.eat(TokenType.VARIABLE).value[1:]  # Remove ?
            return Term(var_name, is_variable=True)

        elif self.match(TokenType.STRING):
            raw = self.eat(TokenType.STRING).value
            if (raw.startswith('"""') and raw.endswith('"""')) or (raw.startswith("'''") and raw.endswith("'''")):
                value = raw[3:-3]
            else:
                value = raw[1:-1]
            value = self._unescape_string(value)
            return Term(value, is_variable=False)

        elif self.match(TokenType.NUMBER):
            value = self.eat(TokenType.NUMBER).value
            return Term(value, is_variable=False)

        elif self.match(TokenType.IDENTIFIER):
            value = self.eat(TokenType.IDENTIFIER).value
            return Term(value, is_variable=False)

        elif self.match(TokenType.LBRACKET):
            # Parse list or list pattern
            return self.parse_list()

        else:
            raise SyntaxError(f"Unexpected token in logical term: {self.current_token}")

    def parse_logical_body(self):
        """Parse the body of a logical rule (predicates, comparisons, is expressions)"""
        goals = [self.parse_logical_goal()]

        while self.match(TokenType.COMMA):
            self.eat(TokenType.COMMA)
            goals.append(self.parse_logical_goal())

        return goals

    def parse_logical_goal(self):
        """Parse a single goal in logical body (predicate, comparison, is expression, or cut)"""
        # Check for cut operator: !
        if self.match(TokenType.CUT):
            self.eat(TokenType.CUT)
            return Cut()

        # Check if this is an 'is' expression or comparison: ?X is Expression or ?X > ?Y
        if self.match(TokenType.VARIABLE):
            # Peek ahead to see if 'is' or comparison operator follows
            saved_pos = self.position
            var_token = self.eat(TokenType.VARIABLE)

            if self.match(TokenType.IS):
                # This is an 'is' expression
                self.eat(TokenType.IS)
                expr = self.parse_additive()  # Parse arithmetic expression
                var_name = var_token.value[1:]  # Remove ?
                return IsExpression(Term(var_name, is_variable=True), expr)
            elif self.match(TokenType.OPERATOR):
                # This might be a comparison: ?X > ?Y
                op_tok = self.eat(TokenType.OPERATOR)
                op = op_tok.value
                if op in ['>', '<', '>=', '<=', '=<', '==', '!=', '=:=', '=\\=']:
                    # Restore and parse as comparison
                    self.position = saved_pos
                    self.current_token = self.tokens[self.position] if self.position < len(self.tokens) else None
                    left = self.parse_logical_term()
                    op_tok = self.eat(TokenType.OPERATOR)
                    right = self.parse_logical_term()
                    return IsExpression(left, BinaryOp(left, op_tok.value, right))
                else:
                    # Not a comparison, restore and parse as predicate
                    self.position = saved_pos
                    self.current_token = self.tokens[self.position] if self.position < len(self.tokens) else None
                    return self.parse_logical_predicate()
            else:
                # Not an 'is' expression or comparison, restore position and parse as predicate
                self.position = saved_pos
                self.current_token = self.tokens[self.position] if self.position < len(self.tokens) else None
                return self.parse_logical_predicate()

        # Check for comparison expressions: ?X > 5, ?Y < 10, etc.
        elif self.match(TokenType.IDENTIFIER):
            # Check if this is a predicate (followed by '(') or a comparison
            saved_pos = self.position
            if self.peek_ahead(1) and self.peek_ahead(1).type == TokenType.LPAREN:
                # This is a predicate call
                return self.parse_logical_predicate()
            else:
                # Try to parse as comparison
                left = self.parse_logical_term()

                # Check for comparison operators
                if self.match(TokenType.OPERATOR):
                    op_tok = self.eat(TokenType.OPERATOR)
                    op = op_tok.value
                    if op in ['>', '<', '>=', '<=', '=<', '==', '!=', '=:=', '=\\=']:
                        right = self.parse_logical_term()
                        # Return a comparison as a special predicate
                        return Predicate(f'_compare_{op}', [left, right])

                # If no comparison, this should be a predicate - error
                raise SyntaxError(f"Expected predicate or comparison, got: {left}")

        elif self.match(TokenType.NUMBER):
            # Must be a comparison
            left = self.parse_logical_term()
            if self.match(TokenType.OPERATOR):
                op_tok = self.eat(TokenType.OPERATOR)
                op = op_tok.value
                if op in ['>', '<', '>=', '<=', '=<', '==', '!=', '=:=', '=\\=']:
                    right = self.parse_logical_term()
                    return Predicate(f'_compare_{op}', [left, right])
            raise SyntaxError(f"Expected comparison operator after number")

        # Otherwise, parse as predicate
        else:
            return self.parse_logical_predicate()

    def parse_subscript_or_slice(self):
        """Parse subscript or slice expression: [index] or [start:end:step]"""
        # Check if this is a slice by looking ahead for colons
        # We need to parse the first expression, then check for colon

        # Save position to potentially backtrack
        start_pos = self.position

        # Try to parse first expression (could be start of slice or just index)
        first_expr = None
        if not self.match(TokenType.COLON):
            first_expr = self.parse_expression()

        # Check if we have a colon (indicating slice)
        if self.match(TokenType.COLON):
            # This is a slice: [start:end:step]
            start = first_expr
            self.eat(TokenType.COLON)

            # Parse end (optional)
            end = None
            if not self.match(TokenType.COLON) and not self.match(TokenType.RBRACKET):
                end = self.parse_expression()

            # Parse step (optional)
            step = None
            if self.match(TokenType.COLON):
                self.eat(TokenType.COLON)
                if not self.match(TokenType.RBRACKET):
                    step = self.parse_expression()

            return Slice(start, end, step)
        else:
            # This is just a regular index
            return first_expr
