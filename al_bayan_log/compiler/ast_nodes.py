"""
Abstract Syntax Tree (AST) nodes for Bayan Language
عقد شجرة التجريد للغة بيان
"""

class ASTNode:
    """Base class for all AST nodes with optional source position"""
    line = None
    column = None
    filename = None

    def with_pos(self, line=None, column=None, filename=None):
        self.line = line
        self.column = column
        self.filename = filename
        return self

# ============ Traditional Programming Nodes ============

class Program(ASTNode):
    """Root node of the program"""
    def __init__(self, statements):
        self.statements = statements

    def __repr__(self):
        return f"Program({len(self.statements)} statements)"

class Block(ASTNode):
    """A block of statements"""
    def __init__(self, statements):
        self.statements = statements

    def __repr__(self):
        return f"Block({len(self.statements)} statements)"

class Assignment(ASTNode):
    """Variable assignment: x = 5"""
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"Assignment({self.name}, {self.value})"

class BinaryOp(ASTNode):
    """Binary operation: a + b"""
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right

    def __repr__(self):
        return f"BinaryOp({self.operator}, {self.left}, {self.right})"

class UnaryOp(ASTNode):
    """Unary operation: -x, not x"""
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand

    def __repr__(self):
        return f"UnaryOp({self.operator}, {self.operand})"

class Number(ASTNode):
    """Numeric literal"""
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Number({self.value})"

class String(ASTNode):
    """String literal"""
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"String({repr(self.value)})"

class Boolean(ASTNode):
    """Boolean literal"""
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Boolean({self.value})"

class Variable(ASTNode):
    """Variable reference"""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Variable({self.name})"

class List(ASTNode):
    """List literal: [1, 2, 3]"""
    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"List({len(self.elements)} elements)"

class ListComprehension(ASTNode):
    """List comprehension: [expr for x in iterable if cond]"""
    def __init__(self, expr, var_name, iterable, condition=None):
        self.expr = expr
        self.var_name = var_name
        self.iterable = iterable
        self.condition = condition

    def __repr__(self):
        return f"ListComprehension({self.var_name} in ...)"

class ListPattern(ASTNode):
    """List pattern for Prolog-style matching: [H|T] or [H1, H2|T]"""
    def __init__(self, head_elements, tail):
        self.head_elements = head_elements  # List of head elements
        self.tail = tail  # Tail variable or expression

    def __repr__(self):
        return f"ListPattern({len(self.head_elements)} heads | tail)"

class Slice(ASTNode):
    """Slice expression: list[start:end:step]"""
    def __init__(self, start=None, end=None, step=None):
        self.start = start
        self.end = end
        self.step = step

    def __repr__(self):
        return f"Slice({self.start}:{self.end}:{self.step})"

class Tuple(ASTNode):
    """Tuple literal: (1, 2, 3)"""
    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"Tuple({len(self.elements)} elements)"

class Set(ASTNode):
    """Set literal: {1, 2, 3}"""
    def __init__(self, elements):
        self.elements = elements

    def __repr__(self):
        return f"Set({len(self.elements)} elements)"

class IsExpression(ASTNode):
    """Arithmetic evaluation in logic context: ?X is 5 + 3"""
    def __init__(self, variable, expression):
        self.variable = variable  # Variable to bind result to
        self.expression = expression  # Arithmetic expression to evaluate

    def __repr__(self):
        return f"IsExpression({self.variable} is {self.expression})"

class AsyncFunctionDef(ASTNode):
    """Async function definition: async def name(params): body"""
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

    def __repr__(self):
        return f"AsyncFunctionDef({self.name}, {len(self.params)} params)"

class AwaitExpr(ASTNode):
    """Await expression: await coroutine()"""
    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return f"AwaitExpr({self.expression})"

class YieldExpr(ASTNode):
    """Yield expression: yield value"""
    def __init__(self, value=None):
        self.value = value

    def __repr__(self):
        return f"YieldExpr({self.value})"

class WithStatement(ASTNode):
    """With statement: with expr as var: body"""
    def __init__(self, context_expr, target_var, body):
        self.context_expr = context_expr
        self.target_var = target_var  # Can be None
        self.body = body

    def __repr__(self):
        return f"WithStatement({self.context_expr} as {self.target_var})"

class Cut(ASTNode):
    """Cut operator: ! (prevents backtracking in logic programming)"""
    def __init__(self):
        pass

    def __repr__(self):
        return "Cut(!)"

class Decorator(ASTNode):
    """Decorator: @decorator_name or @decorator(args)"""
    def __init__(self, name, args=None):
        self.name = name
        self.args = args if args is not None else []

    def __repr__(self):
        if self.args:
            return f"Decorator(@{self.name}({len(self.args)} args))"
        return f"Decorator(@{self.name})"

class Dict(ASTNode):
    """Dictionary literal: {key: value}"""
    def __init__(self, pairs):
        self.pairs = pairs  # List of (key, value) tuples

    def __repr__(self):
        return f"Dict({len(self.pairs)} pairs)"

class PhraseStatement(ASTNode):
    """Nominal phrase sugar inside hybrid blocks: e.g., محمد الطبيب. or عصير العنب[of]."""
    def __init__(self, text, relation=None):
        self.text = text
        self.relation = relation

    def __repr__(self):
        return f"PhraseStatement({self.text!r}, relation={self.relation!r})"

class Parameter(ASTNode):
    """Function parameter with optional default value and support for *args/**kwargs"""
    def __init__(self, name, default_value=None, is_varargs=False, is_kwargs=False):
        self.name = name
        self.default_value = default_value
        self.is_varargs = is_varargs  # *args
        self.is_kwargs = is_kwargs    # **kwargs

    def has_default(self):
        """Check if this parameter has a default value"""
        return self.default_value is not None

    def __repr__(self):
        prefix = ""
        if self.is_kwargs:
            prefix = "**"
        elif self.is_varargs:
            prefix = "*"

        if self.has_default():
            return f"Parameter({prefix}{self.name}={self.default_value})"
        return f"Parameter({prefix}{self.name})"

class NamedArgument(ASTNode):
    """Named argument in function call: func(name=value)"""
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"NamedArgument({self.name}={self.value})"

class FunctionCall(ASTNode):
    """Function call: func(arg1, arg2, name=value)"""
    def __init__(self, name, arguments, named_arguments=None):
        self.name = name
        self.arguments = arguments
        self.named_arguments = named_arguments or {}

    def __repr__(self):
        return f"FunctionCall({self.name}, {len(self.arguments)} args, {len(self.named_arguments)} named)"

class FunctionDef(ASTNode):
    """Function definition with support for default parameters and decorators"""
    def __init__(self, name, parameters, body, decorators=None):
        self.name = name
        # parameters can be strings (old format) or Parameter objects (new format)
        self.parameters = parameters
        self.body = body
        self.decorators = decorators if decorators is not None else []

    def __repr__(self):
        dec_str = f", {len(self.decorators)} decorators" if self.decorators else ""
        return f"FunctionDef({self.name}, {len(self.parameters)} params{dec_str})"

class ClassDef(ASTNode):
    """Class definition with support for decorators"""
    def __init__(self, name, base_class, body, base_classes=None, decorators=None):
        self.name = name
        # Backward-compat single base
        self.base_class = base_class
        # New: support multiple bases
        self.base_classes = base_classes
        self.body = body
        self.decorators = decorators if decorators is not None else []

    def __repr__(self):
        dec_str = f", {len(self.decorators)} decorators" if self.decorators else ""
        return f"ClassDef({self.name}{dec_str})"

class ObjectInstance(ASTNode):
    """Object instance"""
    def __init__(self, class_name, arguments):
        self.class_name = class_name
        self.arguments = arguments

    def __repr__(self):
        return f"ObjectInstance({self.class_name})"

class AttributeAccess(ASTNode):
    """Attribute access (obj.attr)"""
    def __init__(self, object_expr, attribute_name):
        self.object_expr = object_expr
        self.attribute_name = attribute_name

    def __repr__(self):
        return f"AttributeAccess({self.attribute_name})"

class MethodCall(ASTNode):
    """Method call (obj.method()) with support for named arguments"""
    def __init__(self, object_expr, method_name, arguments, named_arguments=None):
        self.object_expr = object_expr
        self.method_name = method_name
        self.arguments = arguments
        self.named_arguments = named_arguments or {}

    def __repr__(self):
        return f"MethodCall({self.method_name})"

class SubscriptAccess(ASTNode):
    """Indexing access (obj[index])"""
    def __init__(self, object_expr, index_expr):
        self.object_expr = object_expr
        self.index_expr = index_expr

    def __repr__(self):
        return "SubscriptAccess([])"

class AttributeAssignment(ASTNode):
    """Attribute assignment: obj.attr = value"""
    def __init__(self, object_expr, attribute_name, value):
        self.object_expr = object_expr
        self.attribute_name = attribute_name
        self.value = value

    def __repr__(self):
        return f"AttributeAssignment({self.attribute_name})"

class SubscriptAssignment(ASTNode):
    """Subscript assignment: obj[index] = value"""
    def __init__(self, object_expr, index_expr, value):
        self.object_expr = object_expr
        self.index_expr = index_expr
        self.value = value

    def __repr__(self):
        return "SubscriptAssignment([])"

class SelfReference(ASTNode):
    """Self reference"""
    def __init__(self):
        pass

    def __repr__(self):
        return "SelfReference()"

class SuperCall(ASTNode):
    """Super call for parent class with support for named arguments"""
    def __init__(self, method_name, arguments, named_arguments=None):
        self.method_name = method_name
        self.arguments = arguments
        self.named_arguments = named_arguments or {}

    def __repr__(self):
        return f"SuperCall({self.method_name})"

class ImportStatement(ASTNode):
    """Import statement"""
    def __init__(self, module_name, alias=None):
        self.module_name = module_name
        self.alias = alias

    def __repr__(self):
        return f"ImportStatement({self.module_name})"

class FromImportStatement(ASTNode):
    """From ... import statement"""
    def __init__(self, module_name, names, aliases=None):
        self.module_name = module_name
        self.names = names
        self.aliases = aliases or []

    def __repr__(self):
        return f"FromImportStatement({self.module_name})"

class IfStatement(ASTNode):
    """If statement"""
    def __init__(self, condition, then_branch, else_branch=None):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

    def __repr__(self):
        return f"IfStatement(condition, then, else={self.else_branch is not None})"

class ForLoop(ASTNode):
    """For loop"""
    def __init__(self, variable, iterable, body):
        self.variable = variable
        self.iterable = iterable
        self.body = body

    def __repr__(self):
        return f"ForLoop({self.variable}, iterable, body)"

class WhileLoop(ASTNode):
    """While loop"""
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def __repr__(self):
        return f"WhileLoop(condition, body)"

class ReturnStatement(ASTNode):
    """Return statement"""
    def __init__(self, value=None):
        self.value = value

    def __repr__(self):
        return f"ReturnStatement({self.value})"

class BreakStatement(ASTNode):
    """Break statement"""
    def __repr__(self):
        return "BreakStatement()"

class ContinueStatement(ASTNode):
    """Continue statement"""
    def __repr__(self):
        return "ContinueStatement()"

class PrintStatement(ASTNode):
    """Print statement"""
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"PrintStatement({self.value})"

# ============ Logical Programming Nodes ============

class RaiseStatement(ASTNode):
    """Raise statement"""
    def __init__(self, value=None):
        self.value = value

    def __repr__(self):
        return f"RaiseStatement({self.value})"

class ExceptHandler(ASTNode):
    """Except handler: except [Type] [as alias]: body"""
    def __init__(self, type_name=None, alias=None, body=None):
        self.type_name = type_name
        self.alias = alias
        self.body = body

    def __repr__(self):
        return f"ExceptHandler({self.type_name}, alias={self.alias})"

class TryExceptFinally(ASTNode):
    """Try/Except/Finally statement"""
    def __init__(self, try_block, handlers=None, finally_block=None):
        self.try_block = try_block
        self.handlers = handlers or []
        self.finally_block = finally_block

    def __repr__(self):
        return f"TryExceptFinally(handlers={len(self.handlers)}, finally={self.finally_block is not None})"


class LogicalFact(ASTNode):
    """Logical fact: parent(john, mary)."""
    def __init__(self, predicate):
        self.predicate = predicate

    def __repr__(self):
        return f"LogicalFact({self.predicate})"

class LogicalRule(ASTNode):
    """Logical rule: grandparent(X, Z) :- parent(X, Y), parent(Y, Z)."""
    def __init__(self, head, body):
        self.head = head
        self.body = body  # List of predicates

    def __repr__(self):
        return f"LogicalRule({self.head} :- ...)"

class LogicalQuery(ASTNode):
    """Logical query: ?- parent(X, john)."""
    def __init__(self, goal):
        self.goal = goal

    def __repr__(self):
        return f"LogicalQuery({self.goal})"

class LogicalPredicate(ASTNode):
    """Logical predicate: parent(X, Y)"""
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

    def __repr__(self):
        return f"LogicalPredicate({self.name}/{len(self.arguments)})"

class LogicalVariable(ASTNode):
    """Logical variable: ?X, ?Y"""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"LogicalVariable(?{self.name})"

class LogicalConstant(ASTNode):
    """Logical constant: atom, number, string"""
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"LogicalConstant({self.value})"

class LogicalConjunction(ASTNode):
    """Logical AND: goal1, goal2"""
    def __init__(self, goals):
        self.goals = goals

    def __repr__(self):
        return f"LogicalConjunction({len(self.goals)} goals)"

class LogicalDisjunction(ASTNode):
    """Logical OR: goal1; goal2"""
    def __init__(self, goals):
        self.goals = goals

    def __repr__(self):
        return f"LogicalDisjunction({len(self.goals)} goals)"

class LogicalNegation(ASTNode):
    """Logical NOT: \\+ goal"""
    def __init__(self, goal):
        self.goal = goal

    def __repr__(self):
        return f"LogicalNegation({self.goal})"

# ============ Hybrid Nodes ============

class HybridBlock(ASTNode):
    """Hybrid block combining traditional and logical code"""
    def __init__(self, traditional_stmts, logical_stmts):
        self.traditional_stmts = traditional_stmts
        self.logical_stmts = logical_stmts

    def __repr__(self):
        return f"HybridBlock({len(self.traditional_stmts)} trad, {len(self.logical_stmts)} logical)"

class LogicalIfStatement(ASTNode):
    """If statement with logical condition"""
    def __init__(self, condition, then_branch, else_branch=None):
        self.condition = condition  # Can be a logical query
        self.then_branch = then_branch
        self.else_branch = else_branch

    def __repr__(self):
        return f"LogicalIfStatement(condition, then, else={self.else_branch is not None})"

class QueryExpression(ASTNode):
    """Query expression in traditional code: query parent(?X, john)"""
    def __init__(self, goal):
        self.goal = goal

    def __repr__(self):
        return f"QueryExpression({self.goal})"


# ============ Entity System Nodes ============

class EntityDef(ASTNode):
    """Entity definition: entity <name> { ... } where body is a dict-like structure"""
    def __init__(self, name, body):
        self.name = name
        self.body = body  # AST node (Dict) to be evaluated later

    def __repr__(self):
        return f"EntityDef({self.name})"

class ApplyActionStmt(ASTNode):
    """Apply action: apply <actor>.<action>(<target>, [action_value=...])"""
    def __init__(self, actor_name, action_name, target_expr, named_args=None):
        self.actor_name = actor_name
        self.action_name = action_name
        self.target_expr = target_expr
        self.named_args = named_args or {}

    def __repr__(self):
        return f"ApplyActionStmt({self.actor_name}.{self.action_name}(...))"
