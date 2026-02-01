import os
import sys
from .lexer import HybridLexer, TokenType
from .parser import HybridParser
from .ast_nodes import *
from .logical_engine import Term, Predicate, Fact, Rule

class BayanSupremeCompiler:
    """
    Advanced Transpiler for Bayan Language.
    Converts Bayan AST into optimized Python code with Hybrid Logic support.
    """

    def __init__(self):
        self.indent_level = 0
        self.output = []
        self.logical_rules = []
        self.in_class = False

    def indent(self):
        return "    " * self.indent_level

    def emit(self, text):
        self.output.append(self.indent() + text)

    def compile(self, code, filename="<string>"):
        """Compile Bayan code to Python"""
        lexer = HybridLexer(code)
        tokens = lexer.tokenize()
        parser = HybridParser(tokens, filename)
        ast = parser.parse()
        
        self.output = []
        self.visit(ast)
        return "\n".join(self.output)

    def visit(self, node):
        """Generic visitor"""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visitor for {node.__class__.__name__}")

    def visit_Program(self, node):
        self.emit("# Transpiled by Bayan Supreme Compiler")
        self.emit("from al_bayan_log.compiler.logical_engine import LogicalEngine, Term, Predicate, Fact, Rule")
        self.emit("")
        self.emit("# Global Logical Engine")
        self.emit("global_logical_engine = LogicalEngine()")
        self.emit("")
        for stmt in node.statements:
            self.visit(stmt)

    def visit_ClassDef(self, node):
        base_str = ""
        if node.base_classes:
            base_str = f"({', '.join(node.base_classes)})"
        elif node.base_class:
            base_str = f"({node.base_class})"
            
        self.emit(f"class {node.name}{base_str}:")
        self.indent_level += 1
        
        old_in_class = self.in_class
        self.in_class = True
        
        # Inject LogicalEngine if needed
        self.emit("def __init__(self, *args, **kwargs):")
        self.indent_level += 1
        if node.base_class:
            self.emit("super().__init__(*args, **kwargs)")
        self.emit("self.logical_engine = LogicalEngine()")
        
        # Inject hybrid logic from class body into __init__
        for stmt in node.body.statements:
            if isinstance(stmt, HybridBlock):
                for lstmt in stmt.logical_stmts:
                    if isinstance(lstmt, LogicalFact):
                        pred = lstmt.predicate
                        args_list = [self.visit_logical_arg(a) for a in pred.args]
                        self.emit(f"self.logical_engine.add_fact(Fact(Predicate({repr(pred.name)}, [{', '.join(args_list)}])))")
        
        self.indent_level -= 1
        
        # Visit the rest of the body
        for stmt in node.body.statements:
            self.visit(stmt)
            
        self.in_class = old_in_class
        self.indent_level -= 1
        self.emit("")

    def visit_logic_injection(self, block):
        """Inject logic rules into the logical_engine instance"""
        # This is a bit simplified; in a real implementation we'd gather all logic
        # from the entire class scope.
        pass

    def visit_FunctionDef(self, node):
        params = []
        for p in node.parameters:
            if isinstance(p, Parameter):
                prefix = ""
                if p.is_kwargs: prefix = "**"
                elif p.is_varargs: prefix = "*"
                param_str = f"{prefix}{p.name}"
                if p.has_default():
                    # We'd need to visit the default expression too
                    param_str += " = None" # Simplified
                params.append(param_str)
            else:
                params.append(p)
                
        self.emit(f"def {node.name}({', '.join(params)}):")
        self.indent_level += 1
        for stmt in node.body.statements:
            self.visit(stmt)
        self.indent_level -= 1
        self.emit("")

    def visit_Block(self, node):
        if not node.statements:
            self.emit("pass")
        else:
            for stmt in node.statements:
                self.visit(stmt)

    def visit_Assignment(self, node):
        value_code = self.visit_expr(node.value)
        self.emit(f"{node.name} = {value_code}")

    def visit_AttributeAssignment(self, node):
        obj = self.visit_expr(node.object_expr)
        value = self.visit_expr(node.value)
        self.emit(f"{obj}.{node.attribute_name} = {value}")

    def visit_SubscriptAssignment(self, node):
        obj = self.visit_expr(node.object_expr)
        idx = self.visit_expr(node.index_expr)
        value = self.visit_expr(node.value)
        self.emit(f"{obj}[{idx}] = {value}")

    def visit_BinaryOp(self, node):
        # When a binary op is a statement (rare, but possible)
        self.emit(self.visit_expr(node))

    def visit_logical_arg(self, arg):
        """Convert a logical argument (Term, List, etc.) to Python code producing a Term"""
        if isinstance(arg, Term):
            return f"Term({repr(arg.value)}, is_variable={arg.is_variable})"
        elif isinstance(arg, List):
            # Convert Bayan List to Python list of Terms
            elements = [self.visit_logical_arg(e) for e in arg.elements]
            return f"[{', '.join(elements)}]"
        return self.visit_expr(arg)

    def visit_HybridBlock(self, node):
        self.emit("# Start Hybrid Logic Block")
        # Logical facts are now injected into __init__ if in_class
        if not self.in_class:
            for lstmt in node.logical_stmts:
                if isinstance(lstmt, LogicalFact):
                    pred = lstmt.predicate
                    args_list = [self.visit_logical_arg(a) for a in pred.args]
                    self.emit(f"global_logical_engine.add_fact(Fact(Predicate({repr(pred.name)}, [{', '.join(args_list)}])))")
        
        for tstmt in node.traditional_stmts:
            self.visit(tstmt)
        self.emit("# End Hybrid Logic Block")

    def visit_LogicalFact(self, node):
        # If a fact appears outside a hybrid block (top level)
        # We might need a global logical engine
        pass

    def visit_String(self, node):
        self.emit(repr(node.value))

    def visit_Number(self, node):
        self.emit(str(node.value))

    def visit_Boolean(self, node):
        self.emit(str(node.value))

    def visit_ReturnStatement(self, node):
        if node.value:
            self.emit(f"return {self.visit_expr(node.value)}")
        else:
            self.emit("return")

    def visit_PrintStatement(self, node):
        self.emit(f"print({self.visit_expr(node.value)})")

    def visit_IfStatement(self, node):
        self.emit(f"if {self.visit_expr(node.condition)}:")
        self.indent_level += 1
        self.visit(node.then_branch)
        self.indent_level -= 1
        if node.else_branch:
            self.emit("else:")
            self.indent_level += 1
            self.visit(node.else_branch)
            self.indent_level -= 1

    def visit_expr(self, node):
        """Visit an expression and return its Python string representation"""
        if isinstance(node, Number):
            return str(node.value)
        elif isinstance(node, String):
            return repr(node.value)
        elif isinstance(node, Variable):
            return node.name
        elif isinstance(node, BinaryOp):
            return f"({self.visit_expr(node.left)} {node.operator} {self.visit_expr(node.right)})"
        elif isinstance(node, UnaryOp):
            return f"({node.operator} {self.visit_expr(node.operand)})"
        elif isinstance(node, FunctionCall):
            args = [self.visit_expr(a) for a in node.arguments]
            return f"{node.name}({', '.join(args)})"
        elif isinstance(node, MethodCall):
            args = [self.visit_expr(a) for a in node.arguments]
            return f"{self.visit_expr(node.object_expr)}.{node.method_name}({', '.join(args)})"
        elif isinstance(node, AttributeAccess):
            return f"{self.visit_expr(node.object_expr)}.{node.attribute_name}"
        elif isinstance(node, SubscriptAccess):
            return f"{self.visit_expr(node.object_expr)}[{self.visit_expr(node.index_expr)}]"
        elif isinstance(node, SelfReference):
            return "self"
        elif isinstance(node, List):
            elements = [self.visit_expr(e) for e in node.elements]
            return f"[{', '.join(elements)}]"
        elif isinstance(node, ListComprehension):
            cond_str = f" if {self.visit_expr(node.condition)}" if node.condition else ""
            return f"[{self.visit_expr(node.expr)} for {node.var_name} in {self.visit_expr(node.iterable)}{cond_str}]"
        elif isinstance(node, Dict):
            pairs = [f"{self.visit_expr(k)}: {self.visit_expr(v)}" for k, v in node.pairs]
            return f"{{{', '.join(pairs)}}}"
        # ... and so on for all nodes ...
        return f"# [Unsupported Expr: {node.__class__.__name__}]"

    def visit_Variable(self, node):
        # Used when Variable is a statement (e.g. naked expression)
        self.emit(node.name)

    def visit_FunctionCall(self, node):
        args = [self.visit_expr(a) for a in node.arguments]
        self.emit(f"{node.name}({', '.join(args)})")

    def visit_MethodCall(self, node):
        args = [self.visit_expr(a) for a in node.arguments]
        self.emit(f"{self.visit_expr(node.object_expr)}.{node.method_name}({', '.join(args)})")

    def visit_AttributeAccess(self, node):
        self.emit(f"{self.visit_expr(node.object_expr)}.{node.attribute_name}")

    def visit_ImportStatement(self, node):
        alias_str = f" as {node.alias}" if node.alias else ""
        self.emit(f"import {node.module_name}{alias_str}")

    def visit_FromImportStatement(self, node):
        names_str = ", ".join(node.names)
        self.emit(f"from {node.module_name} import {names_str}")

    def visit_PhraseStatement(self, node):
        # Support for the "Phrase" sugar
        self.emit(f"# Phrase: {node.text} [{node.relation}]")
        self.emit(f"self.logical_engine.add_fact(Fact(Predicate('phrase', [Term({repr(node.text)}), Term({repr(node.relation)})])))")

