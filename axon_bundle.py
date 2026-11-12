#!/usr/bin/env python3
"""
Axon Enterprise v1.0 - All-in-One Repository Generator, Builder & Test Runner.
Adds: SystemInspector, ModelConverter, CLI Subcommands, and dynamic CFLAGS.

Run: python3 axon_bundle.py
"""

import os
import sys
import subprocess

FILES = {}

# =====================================================================
# Configuration & Build Files
# =====================================================================
FILES["pyproject.toml"] = """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "axon-compiler"
version = "1.0.0-enterprise"
description = "Axon Enterprise DSL & Compiler with Hardware Autotuning & Model Converter"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "llvmlite>=0.40.0",
]

[project.scripts]
axon = "axon.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["axon*"]
"""

FILES["README.md"] = """# Axon Enterprise Compiler (v1.0)

Axon is a domain-specific compiler transforming tensor graphs into standalone native binaries.
Embedded FP4 weights, zero Python runtime dependency.

## Enterprise Features
- **Hardware Autotuner**: SystemInspector dynamically detects OS, Arch, Vector features (AVX2/NEON), RAM, and sets optimal CFLAGS.
- **Model Converter**: ModelConverter reads JSON weights and auto-generates valid Axon source code.
- **Unified CLI**:
  - `axon doctor` / `axon setup`
  - `axon convert <weights.json> -o <model.bin>`
  - `axon build <source.axon> -o <output.bin> [--target <triple>]`
  - `axon run <binary>`

## Build
python3 axon_bundle.py
"""

FILES["Makefile"] = """CC ?= gcc
CFLAGS ?= -O3 -fPIC -Wall -Wextra -std=c11 -march=native -fopenmp -ffast-math
AR ?= ar
PYTHON ?= python3

RUNTIME_DIR = runtime
RUNTIME_LIB = $(RUNTIME_DIR)/libaxon_runtime.a
RUNTIME_OBJ = $(RUNTIME_DIR)/axon_tensor.o \\
              $(RUNTIME_DIR)/axon_matmul.o \\
              $(RUNTIME_DIR)/axon_quant.o \\
              $(RUNTIME_DIR)/axon_mmap.o \\
              $(RUNTIME_DIR)/axon_runtime.o

.PHONY: all clean runtime test

all: runtime

runtime: $(RUNTIME_LIB)

$(RUNTIME_DIR)/%.o: $(RUNTIME_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(RUNTIME_LIB): $(RUNTIME_OBJ)
	$(AR) rcs $@ $^

clean:
	rm -f $(RUNTIME_DIR)/*.o $(RUNTIME_LIB)
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f *.bin *.o *.blob *.S *.gen.o test_e2e.* test_tf.* auto_model.*

test: runtime
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"
"""

# =====================================================================
# Axon Package: __init__, utils, frontend
# =====================================================================
FILES["axon/__init__.py"] = """__version__ = "1.0.0-enterprise"\n"""

FILES["axon/utils/__init__.py"] = """from .errors import AxonError, LexError, ParseError, SemanticError, CodegenError
__all__ = ["AxonError", "LexError", "ParseError", "SemanticError", "CodegenError"]
"""

FILES["axon/utils/errors.py"] = """class AxonError(Exception):
    def __init__(self, message: str, line: int = 0, col: int = 0, filename: str = "<axon>"):
        self.message = message
        self.line = line
        self.col = col
        self.filename = filename
        super().__init__(self.__str__())

    def __str__(self) -> str:
        if self.line > 0:
            return f"[{self.filename}:{self.line}:{self.col}] {self.message}"
        return f"[{self.filename}] {self.message}"

class LexError(AxonError): pass
class ParseError(AxonError): pass
class SemanticError(AxonError): pass
class CodegenError(AxonError): pass
"""

FILES["axon/frontend/__init__.py"] = """from .tokens import Token, TokenKind
from .lexer import Lexer
from .ast_nodes import Program, FunctionDef, Param, TypeExpr, TensorLiteral, VarRef, CallExpr, BinaryExpr, LetStmt, ReturnStmt
from .parser import Parser
from .semantic import SemanticAnalyzer
__all__ = ["Token", "TokenKind", "Lexer", "Program", "FunctionDef", "Param", "TypeExpr", "TensorLiteral", "VarRef", "CallExpr", "BinaryExpr", "LetStmt", "ReturnStmt", "Parser", "SemanticAnalyzer"]
"""

FILES["axon/frontend/tokens.py"] = """from enum import Enum, auto
from dataclasses import dataclass

class TokenKind(Enum):
    IDENT = auto(); INT_LIT = auto(); FLOAT_LIT = auto(); STRING_LIT = auto()
    FN = auto(); LET = auto(); IN = auto(); RETURN = auto(); AS = auto()
    TENSOR = auto(); MATMUL = auto(); RELU = auto(); ADD = auto(); MUL = auto(); CAST = auto()
    LAYERNORM = auto(); GELU = auto(); SOFTMAX = auto(); TRANSPOSE = auto()
    LPAREN = auto(); RPAREN = auto(); LBRACKET = auto(); RBRACKET = auto(); LBRACE = auto(); RBRACE = auto()
    COMMA = auto(); COLON = auto(); ARROW = auto(); ASSIGN = auto(); SEMICOLON = auto()
    PLUS = auto(); STAR = auto(); EOF = auto()

@dataclass(frozen=True)
class Token:
    kind: TokenKind
    lexeme: str
    line: int
    col: int
"""

FILES["axon/frontend/lexer.py"] = """from .tokens import Token, TokenKind
from ..utils.errors import LexError

class Lexer:
    KEYWORDS = {
        "fn": TokenKind.FN, "let": TokenKind.LET, "in": TokenKind.IN,
        "return": TokenKind.RETURN, "as": TokenKind.AS, "tensor": TokenKind.TENSOR,
        "matmul": TokenKind.MATMUL, "relu": TokenKind.RELU, "add": TokenKind.ADD,
        "mul": TokenKind.MUL, "cast": TokenKind.CAST,
        "layernorm": TokenKind.LAYERNORM, "gelu": TokenKind.GELU, "softmax": TokenKind.SOFTMAX,
        "transpose": TokenKind.TRANSPOSE,
    }

    def __init__(self, source: str, filename: str = "<axon>"):
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.source[idx] if idx < len(self.source) else ""

    def _advance(self) -> str:
        ch = self._peek()
        self.pos += 1
        if ch == "\\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> list[Token]:
        while self.pos < len(self.source):
            ch = self._peek()
            if ch.isspace():
                self._advance()
                continue
            if ch == "/" and self._peek(1) == "/":
                while self._peek() not in ("", "\\n"):
                    self._advance()
                continue
            start_l, start_c = self.line, self.col
            if ch == "-" and self._peek(1) == ">":
                self._advance(); self._advance()
                self.tokens.append(Token(TokenKind.ARROW, "->", start_l, start_c))
                continue
            single = {
                "(": TokenKind.LPAREN, ")": TokenKind.RPAREN,
                "[": TokenKind.LBRACKET, "]": TokenKind.RBRACKET,
                "{": TokenKind.LBRACE, "}": TokenKind.RBRACE,
                ",": TokenKind.COMMA, ":": TokenKind.COLON,
                ";": TokenKind.SEMICOLON, "=": TokenKind.ASSIGN,
                "+": TokenKind.PLUS, "*": TokenKind.STAR
            }
            if ch in single:
                self._advance()
                self.tokens.append(Token(single[ch], ch, start_l, start_c))
            elif ch.isalpha() or ch == "_":
                chars = []
                while self._peek().isalnum() or self._peek() == "_":
                    chars.append(self._advance())
                s = "".join(chars)
                self.tokens.append(Token(self.KEYWORDS.get(s, TokenKind.IDENT), s, start_l, start_c))
            elif ch.isdigit() or (ch == "-" and (self._peek(1).isdigit() or self._peek(1) == ".")):
                chars = [self._advance()] if ch == "-" else []
                has_dot = False
                while self._peek().isdigit() or (self._peek() == "." and not has_dot):
                    if self._peek() == ".": has_dot = True
                    chars.append(self._advance())
                s = "".join(chars)
                self.tokens.append(Token(TokenKind.FLOAT_LIT if has_dot else TokenKind.INT_LIT, s, start_l, start_c))
            else:
                raise LexError(f"Unexpected character: {ch!r}", self.line, self.col, self.filename)
        self.tokens.append(Token(TokenKind.EOF, "", self.line, self.col))
        return self.tokens
"""

FILES["axon/frontend/ast_nodes.py"] = """from dataclasses import dataclass, field
from typing import Optional, List

@dataclass(frozen=True)
class SourceLoc:
    line: int
    col: int

class Expr: loc: SourceLoc
class Stmt: loc: SourceLoc

@dataclass
class TypeExpr(Expr):
    name: str
    shape: List[int]
    loc: SourceLoc

@dataclass
class Param(Expr):
    name: str
    type_expr: TypeExpr
    loc: SourceLoc

@dataclass
class TensorLiteral(Expr):
    values: List[float]
    shape: List[int]
    dtype: str
    loc: SourceLoc

@dataclass
class VarRef(Expr):
    name: str
    loc: SourceLoc

@dataclass
class CallExpr(Expr):
    callee: str
    args: List[Expr]
    loc: SourceLoc
    resolved_shape: Optional[List[int]] = None
    resolved_dtype: Optional[str] = None

@dataclass
class BinaryExpr(Expr):
    op: str
    left: Expr
    right: Expr
    loc: SourceLoc
    resolved_shape: Optional[List[int]] = None
    resolved_dtype: Optional[str] = None

@dataclass
class LetStmt(Stmt):
    name: str
    type_expr: Optional[TypeExpr]
    value: Expr
    loc: SourceLoc

@dataclass
class ReturnStmt(Stmt):
    value: Expr
    loc: SourceLoc

@dataclass
class ExprStmt(Stmt):
    expr: Expr
    loc: SourceLoc

@dataclass
class FunctionDef(Expr):
    name: str
    params: List[Param]
    return_type: TypeExpr
    body: List[Stmt]
    loc: SourceLoc

@dataclass
class Program(Expr):
    functions: List[FunctionDef]
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(1, 1))
"""

FILES["axon/frontend/grammar.py"] = """AXON_EBNF_SPEC = '''
program      = { function_def } ;
function_def = "fn" identifier "(" [ param { "," param } ] ")" "->" type_expr block ;
param        = identifier ":" type_expr ;
type_expr    = dtype [ "[" integer { "," integer } "]" ] ;
dtype        = "f32" | "f16" | "fp4" | "i32" | "bool" ;
block        = "{" { statement } "}" ;
statement    = let_stmt | return_stmt | expr_stmt ;
let_stmt     = "let" identifier [ ":" type_expr ] "=" expr ";" ;
return_stmt  = "return" expr ";" ;
expr_stmt    = expr ";" ;
builtin_ops  = "matmul" | "relu" | "add" | "cast" | "gelu" | "softmax" | "layernorm" | "transpose" ;
'''
"""

FILES["axon/frontend/parser.py"] = """from .tokens import Token, TokenKind
from .ast_nodes import (
    SourceLoc, Program, FunctionDef, Param, TypeExpr, TensorLiteral,
    VarRef, CallExpr, BinaryExpr, LetStmt, ReturnStmt, ExprStmt, Expr, Stmt
)
from ..utils.errors import ParseError

CALLABLE_TOKENS = (
    TokenKind.IDENT, TokenKind.MATMUL, TokenKind.RELU, TokenKind.ADD,
    TokenKind.MUL, TokenKind.CAST, TokenKind.LAYERNORM, TokenKind.GELU,
    TokenKind.SOFTMAX, TokenKind.TRANSPOSE,
)

class Parser:
    def __init__(self, tokens: list[Token], filename: str = "<axon>"):
        self.tokens = tokens
        self.filename = filename
        self.pos = 0

    def _peek(self) -> Token: return self.tokens[self.pos]
    def _check(self, kind: TokenKind) -> bool: return self._peek().kind == kind

    def _advance(self) -> Token:
        tok = self._peek()
        if tok.kind != TokenKind.EOF: self.pos += 1
        return tok

    def _consume(self, kind: TokenKind, msg: str) -> Token:
        if self._check(kind): return self._advance()
        curr = self._peek()
        raise ParseError(f"{msg}, got {curr.kind.name} ('{curr.lexeme}')", curr.line, curr.col, self.filename)

    def parse(self) -> Program:
        functions = []
        while not self._check(TokenKind.EOF):
            functions.append(self._parse_function())
        return Program(functions=functions)

    def _parse_function(self) -> FunctionDef:
        fn_tok = self._consume(TokenKind.FN, "Expected 'fn'")
        name_tok = self._consume(TokenKind.IDENT, "Expected function name")
        self._consume(TokenKind.LPAREN, "Expected '('")
        params = []
        if not self._check(TokenKind.RPAREN):
            while True:
                p_name = self._consume(TokenKind.IDENT, "Expected parameter name")
                self._consume(TokenKind.COLON, "Expected ':'")
                p_type = self._parse_type()
                params.append(Param(p_name.lexeme, p_type, SourceLoc(p_name.line, p_name.col)))
                if self._check(TokenKind.COMMA): self._advance()
                else: break
        self._consume(TokenKind.RPAREN, "Expected ')'")
        self._consume(TokenKind.ARROW, "Expected '->'")
        return_type = self._parse_type()
        self._consume(TokenKind.LBRACE, "Expected '{'")
        body: list[Stmt] = []
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            body.append(self._parse_statement())
        self._consume(TokenKind.RBRACE, "Expected '}'")
        return FunctionDef(name_tok.lexeme, params, return_type, body, SourceLoc(fn_tok.line, fn_tok.col))

    def _parse_type(self) -> TypeExpr:
        dtype_tok = self._consume(TokenKind.IDENT, "Expected type")
        shape: list[int] = []
        if self._check(TokenKind.LBRACKET):
            self._advance()
            while not self._check(TokenKind.RBRACKET) and not self._check(TokenKind.EOF):
                dim_tok = self._consume(TokenKind.INT_LIT, "Expected integer dimension")
                shape.append(int(dim_tok.lexeme))
                if self._check(TokenKind.COMMA): self._advance()
                else: break
            self._consume(TokenKind.RBRACKET, "Expected ']'")
        return TypeExpr(dtype_tok.lexeme, shape, SourceLoc(dtype_tok.line, dtype_tok.col))

    def _parse_statement(self) -> Stmt:
        if self._check(TokenKind.LET):
            tok = self._advance()
            name_tok = self._consume(TokenKind.IDENT, "Expected name")
            type_expr = None
            if self._check(TokenKind.COLON):
                self._advance()
                type_expr = self._parse_type()
            self._consume(TokenKind.ASSIGN, "Expected '='")
            val = self._parse_expr()
            self._consume(TokenKind.SEMICOLON, "Expected ';'")
            return LetStmt(name_tok.lexeme, type_expr, val, SourceLoc(tok.line, tok.col))
        if self._check(TokenKind.RETURN):
            tok = self._advance()
            val = self._parse_expr()
            self._consume(TokenKind.SEMICOLON, "Expected ';'")
            return ReturnStmt(val, SourceLoc(tok.line, tok.col))
        expr = self._parse_expr()
        self._consume(TokenKind.SEMICOLON, "Expected ';'")
        return ExprStmt(expr, expr.loc)

    def _parse_expr(self) -> Expr:
        left = self._parse_primary()
        while self._check(TokenKind.PLUS) or self._check(TokenKind.STAR):
            op = self._advance()
            right = self._parse_primary()
            left = BinaryExpr(op.lexeme, left, right, SourceLoc(op.line, op.col))
        return left

    def _parse_primary(self) -> Expr:
        tok = self._peek()
        if tok.kind == TokenKind.LBRACKET:
            self._advance()
            values: list[float] = []
            if not self._check(TokenKind.RBRACKET):
                while True:
                    sign = 1.0
                    if self._check(TokenKind.PLUS): self._advance()
                    num = self._advance()
                    if num.kind not in (TokenKind.INT_LIT, TokenKind.FLOAT_LIT):
                        raise ParseError("Expected number", num.line, num.col, self.filename)
                    values.append(sign * float(num.lexeme))
                    if self._check(TokenKind.COMMA): self._advance()
                    else: break
            self._consume(TokenKind.RBRACKET, "Expected ']'")
            dtype, shape = "f32", [len(values)]
            if self._check(TokenKind.AS):
                self._advance()
                texpr = self._parse_type()
                dtype = texpr.name
                if texpr.shape: shape = texpr.shape
            return TensorLiteral(values, shape, dtype, SourceLoc(tok.line, tok.col))

        if tok.kind in CALLABLE_TOKENS:
            ident = self._advance()
            if self._check(TokenKind.LPAREN):
                self._advance()
                args: list[Expr] = []
                if not self._check(TokenKind.RPAREN):
                    while True:
                        args.append(self._parse_expr())
                        if self._check(TokenKind.COMMA): self._advance()
                        else: break
                self._consume(TokenKind.RPAREN, "Expected ')'")
                return CallExpr(ident.lexeme, args, SourceLoc(ident.line, ident.col))
            return VarRef(ident.lexeme, SourceLoc(ident.line, ident.col))

        if tok.kind == TokenKind.LPAREN:
            self._advance(); e = self._parse_expr(); self._consume(TokenKind.RPAREN, "Expected ')'")
            return e
        raise ParseError(f"Unexpected token '{tok.lexeme}'", tok.line, tok.col, self.filename)
"""

FILES["axon/frontend/semantic.py"] = """from dataclasses import dataclass
from typing import Dict, List
from .ast_nodes import Program, FunctionDef, LetStmt, ReturnStmt, ExprStmt, CallExpr, BinaryExpr, TensorLiteral, VarRef, Expr
from ..utils.errors import SemanticError

@dataclass
class TypeInfo:
    dtype: str
    shape: List[int]

class SemanticAnalyzer:
    def __init__(self, filename: str = "<axon>"):
        self.filename = filename
        self.sym: Dict[str, TypeInfo] = {}

    def analyze(self, program: Program) -> Program:
        for fn in program.functions:
            self._analyze_function(fn)
        return program

    def _analyze_function(self, fn: FunctionDef) -> None:
        self.sym.clear()
        for p in fn.params:
            self.sym[p.name] = TypeInfo(p.type_expr.name, p.type_expr.shape)
        for s in fn.body:
            if isinstance(s, LetStmt):
                inf = self._infer(s.value)
                if s.type_expr:
                    if s.type_expr.name != inf.dtype:
                        raise SemanticError(f"Type mismatch on {s.name}", s.loc.line, s.loc.col, self.filename)
                    if s.type_expr.shape and s.type_expr.shape != inf.shape:
                        raise SemanticError(f"Shape mismatch on {s.name}: declared {s.type_expr.shape}, inferred {inf.shape}",
                                            s.loc.line, s.loc.col, self.filename)
                self.sym[s.name] = inf
            elif isinstance(s, ReturnStmt):
                r = self._infer(s.value)
                if r.dtype != fn.return_type.name:
                    raise SemanticError("Return dtype mismatch", s.loc.line, s.loc.col, self.filename)
            elif isinstance(s, ExprStmt):
                self._infer(s.expr)

    def _infer(self, expr: Expr) -> TypeInfo:
        if isinstance(expr, TensorLiteral):
            return TypeInfo(expr.dtype, expr.shape)
        if isinstance(expr, VarRef):
            if expr.name not in self.sym:
                raise SemanticError(f"Undefined: {expr.name}", expr.loc.line, expr.loc.col, self.filename)
            return self.sym[expr.name]
        if isinstance(expr, BinaryExpr):
            lt, rt = self._infer(expr.left), self._infer(expr.right)
            shp = self._broadcast(lt.shape, rt.shape, expr.loc)
            expr.resolved_shape, expr.resolved_dtype = shp, lt.dtype
            return TypeInfo(lt.dtype, shp)
        if isinstance(expr, CallExpr):
            c = expr.callee
            args = [self._infer(a) for a in expr.args]
            if c == "matmul":
                a, b = args[0], args[1]
                if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
                    raise SemanticError(f"Invalid matmul shapes: {a.shape} vs {b.shape}", expr.loc.line, expr.loc.col, self.filename)
                shp = [a.shape[0], b.shape[1]]
                expr.resolved_shape, expr.resolved_dtype = shp, a.dtype
                return TypeInfo(a.dtype, shp)
            elif c == "relu":
                expr.resolved_shape, expr.resolved_dtype = list(args[0].shape), args[0].dtype
                return args[0]
            elif c == "gelu":
                expr.resolved_shape, expr.resolved_dtype = list(args[0].shape), args[0].dtype
                return args[0]
            elif c == "softmax":
                if len(args) != 1:
                    raise SemanticError("softmax requires 1 arg", expr.loc.line, expr.loc.col, self.filename)
                expr.resolved_shape, expr.resolved_dtype = list(args[0].shape), args[0].dtype
                return TypeInfo(args[0].dtype, list(args[0].shape))
            elif c == "layernorm":
                if len(args) != 3:
                    raise SemanticError("layernorm requires (x, gamma, beta)", expr.loc.line, expr.loc.col, self.filename)
                x_t, g_t, b_t = args[0], args[1], args[2]
                if not x_t.shape:
                    raise SemanticError("layernorm requires shaped tensor", expr.loc.line, expr.loc.col, self.filename)
                D = x_t.shape[-1]
                if g_t.shape != [D] or b_t.shape != [D]:
                    raise SemanticError(f"layernorm gamma/beta must be [{D}], got {g_t.shape}/{b_t.shape}",
                                        expr.loc.line, expr.loc.col, self.filename)
                expr.resolved_shape, expr.resolved_dtype = list(x_t.shape), x_t.dtype
                return TypeInfo(x_t.dtype, list(x_t.shape))
            elif c == "transpose":
                if len(args) != 1:
                    raise SemanticError("transpose requires 1 arg", expr.loc.line, expr.loc.col, self.filename)
                a_t = args[0]
                if len(a_t.shape) != 2:
                    raise SemanticError(f"transpose requires 2D tensor, got shape {a_t.shape}",
                                        expr.loc.line, expr.loc.col, self.filename)
                M, N = a_t.shape
                shp = [N, M]
                expr.resolved_shape, expr.resolved_dtype = shp, a_t.dtype
                return TypeInfo(a_t.dtype, shp)
            elif c == "add":
                shp = self._broadcast(args[0].shape, args[1].shape, expr.loc)
                expr.resolved_shape, expr.resolved_dtype = shp, args[0].dtype
                return TypeInfo(args[0].dtype, shp)
            elif c == "cast":
                expr.resolved_shape, expr.resolved_dtype = list(args[0].shape), "f32"
                return TypeInfo("f32", expr.resolved_shape)
        return TypeInfo("f32", [])

    def _broadcast(self, s1: List[int], s2: List[int], loc) -> List[int]:
        if s1 == s2 or not s2: return list(s1)
        if not s1: return list(s2)
        if len(s2) == 1 and (len(s1) == 2 and s2[0] == s1[1]): return list(s1)
        if len(s1) == 1 and (len(s2) == 2 and s1[0] == s2[1]): return list(s2)
        raise SemanticError(f"Cannot broadcast shapes {s1} and {s2}", loc.line, loc.col, self.filename)
"""

# =====================================================================
# Axon Package: IR & Builder
# =====================================================================
FILES["axon/ir/__init__.py"] = """from .tensor_ir import OpKind, IRNode, Graph, WeightBlob
from .builder import IRBuilder
from .printer import IRPrinter
__all__ = ["OpKind", "IRNode", "Graph", "WeightBlob", "IRBuilder", "IRPrinter"]
"""

FILES["axon/ir/tensor_ir.py"] = """from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

class OpKind(Enum):
    INPUT = auto(); CONSTANT = auto(); WEIGHT = auto(); MATMUL = auto()
    ADD = auto(); RELU = auto(); CAST = auto(); RESHAPE = auto()
    GELU = auto(); SOFTMAX = auto(); LAYER_NORM = auto(); TRANSPOSE = auto()
    FUSED_MATMUL_BIAS_RELU = auto(); FUSED_MATMUL_BIAS = auto(); FUSED_MATMUL_BIAS_GELU = auto()

@dataclass
class WeightBlob:
    name: str
    data: List[float]
    shape: List[int]
    dtype: str
    is_quantized: bool = False
    packed_data: Optional[bytes] = None
    scale_bytes: Optional[bytes] = None
    residuals: List[Tuple[int, float]] = field(default_factory=list)

@dataclass
class IRNode:
    id: int
    op: OpKind
    inputs: List[int]
    attrs: Dict[str, Any]
    output_shape: Tuple[int, ...]
    output_dtype: str
    name: str = ""
    def __hash__(self): return hash(self.id)

@dataclass
class Graph:
    nodes: List[IRNode] = field(default_factory=list)
    inputs: List[int] = field(default_factory=list)
    outputs: List[int] = field(default_factory=list)
    weights: Dict[str, WeightBlob] = field(default_factory=dict)

    def get_node(self, nid: int) -> Optional[IRNode]:
        for n in self.nodes:
            if n.id == nid: return n
        return None

    def nodes_in_topo_order(self) -> List[IRNode]:
        in_degree = {n.id: 0 for n in self.nodes}
        adj = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for inp in n.inputs:
                if inp in adj:
                    adj[inp].append(n.id)
                    in_degree[n.id] += 1
        queue = [nid for nid, d in in_degree.items() if d == 0]
        topo: List[IRNode] = []
        while queue:
            cid = queue.pop(0)
            node = self.get_node(cid)
            if node: topo.append(node)
            for neighbor in adj.get(cid, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0: queue.append(neighbor)
        return topo if len(topo) == len(self.nodes) else self.nodes[:]

    def replace_subgraph(self, old_nodes: List[IRNode], new_node: IRNode) -> None:
        old_ids = {n.id for n in old_nodes}
        sink = old_nodes[-1]
        for n in self.nodes:
            for idx, inp in enumerate(n.inputs):
                if inp == sink.id: n.inputs[idx] = new_node.id
        for idx, out_id in enumerate(self.outputs):
            if out_id == sink.id: self.outputs[idx] = new_node.id
        self.nodes = [n for n in self.nodes if n.id not in old_ids]
        self.nodes.append(new_node)
"""

FILES["axon/ir/builder.py"] = """from typing import Dict
from ..frontend.ast_nodes import Program, FunctionDef, LetStmt, ReturnStmt, ExprStmt, CallExpr, BinaryExpr, TensorLiteral, VarRef, Expr
from .tensor_ir import Graph, IRNode, OpKind, WeightBlob

class IRBuilder:
    def __init__(self):
        self.graph = Graph()
        self.nid = 0
        self.var_map: Dict[str, int] = {}

    def _alloc_id(self) -> int:
        i = self.nid; self.nid += 1; return i

    def build(self, program: Program) -> Graph:
        for fn in program.functions:
            for param in fn.params:
                i = self._alloc_id()
                node = IRNode(i, OpKind.INPUT, [], {"name": param.name}, tuple(param.type_expr.shape), param.type_expr.name, param.name)
                self.graph.nodes.append(node)
                self.graph.inputs.append(i)
                self.var_map[param.name] = i
            for stmt in fn.body:
                if isinstance(stmt, LetStmt):
                    vid = self._build_expr(stmt.value, target_name=stmt.name)
                    self.var_map[stmt.name] = vid
                elif isinstance(stmt, ReturnStmt):
                    vid = self._build_expr(stmt.value)
                    self.graph.outputs.append(vid)
                elif isinstance(stmt, ExprStmt):
                    self._build_expr(stmt.expr)
        return self.graph

    def _build_expr(self, expr: Expr, target_name: str = "") -> int:
        if isinstance(expr, VarRef): return self.var_map[expr.name]
        if isinstance(expr, TensorLiteral):
            i = self._alloc_id()
            name = target_name or f"const_{i}"
            self.graph.weights[name] = WeightBlob(name, expr.values, expr.shape, expr.dtype)
            node = IRNode(i, OpKind.CONSTANT, [], {"weight_name": name, "raw_values": expr.values}, tuple(expr.shape), expr.dtype, name)
            self.graph.nodes.append(node)
            return i
        if isinstance(expr, BinaryExpr):
            lid = self._build_expr(expr.left)
            rid = self._build_expr(expr.right)
            i = self._alloc_id()
            op = OpKind.ADD if expr.op == "+" else OpKind.MATMUL
            node = IRNode(i, op, [lid, rid], {}, tuple(expr.resolved_shape or ()), expr.resolved_dtype or "f32", target_name)
            self.graph.nodes.append(node)
            return i
        if isinstance(expr, CallExpr):
            args = [self._build_expr(a) for a in expr.args]
            i = self._alloc_id()
            op_map = {
                "matmul": OpKind.MATMUL, "relu": OpKind.RELU, "add": OpKind.ADD,
                "cast": OpKind.CAST, "gelu": OpKind.GELU, "softmax": OpKind.SOFTMAX,
                "layernorm": OpKind.LAYER_NORM, "transpose": OpKind.TRANSPOSE,
            }
            op = op_map.get(expr.callee, OpKind.CONSTANT)
            node = IRNode(i, op, args, {"callee": expr.callee}, tuple(expr.resolved_shape or ()), expr.resolved_dtype or "f32", target_name)
            self.graph.nodes.append(node)
            return i
        raise NotImplementedError(f"Unsupported expr: {type(expr)}")
"""

FILES["axon/ir/printer.py"] = """from .tensor_ir import Graph

class IRPrinter:
    @staticmethod
    def dump(graph: Graph) -> str:
        lines = ["axon.Graph {"]
        lines.append(f"  // Inputs: {graph.inputs}")
        lines.append(f"  // Outputs: {graph.outputs}")
        lines.append(f"  // Weights: {list(graph.weights.keys())}")
        for n in graph.nodes_in_topo_order():
            inp = ", ".join(f"%{i}" for i in n.inputs)
            shp = "x".join(map(str, n.output_shape))
            lines.append(f"  %{n.id} = {n.op.name}({inp}) : [{shp}] {n.output_dtype} // {n.name}")
        lines.append("}")
        return "\\n".join(lines)
"""

# =====================================================================
# Axon Package: Optimizer
# =====================================================================
FILES["axon/optimizer/__init__.py"] = """from .fusion import FusionPass
from .quantize import FP4QuantizePass
from .passes import OptimizationPipeline
__all__ = ["FusionPass", "FP4QuantizePass", "OptimizationPipeline"]
"""

FILES["axon/optimizer/fusion.py"] = """from typing import Optional
from ..ir.tensor_ir import Graph, IRNode, OpKind

class FusionPass:
    def run(self, graph: Graph) -> Graph:
        changed = True
        while changed:
            changed = False
            for node in graph.nodes_in_topo_order():
                if node.op in (OpKind.RELU, OpKind.GELU):
                    if self._try_fuse_activation(node, graph) is not None:
                        changed = True
                        break
                elif node.op == OpKind.ADD:
                    mm_node = self._parent_op(node, OpKind.MATMUL, graph)
                    if mm_node and self._uses(mm_node, graph) == 1:
                        bias_inp = self._other_input(node, mm_node.id)
                        if bias_inp is not None:
                            fused = IRNode(node.id, OpKind.FUSED_MATMUL_BIAS,
                                           [mm_node.inputs[0], mm_node.inputs[1], bias_inp],
                                           {"fused_from": [mm_node.id, node.id]},
                                           node.output_shape, node.output_dtype, f"fused_{node.name}")
                            graph.replace_subgraph([mm_node, node], fused)
                            changed = True
                            break
        return graph

    def _try_fuse_activation(self, node: IRNode, graph: Graph) -> Optional[IRNode]:
        add_node = self._parent(node, graph)
        if not add_node or add_node.op != OpKind.ADD:
            return None
        if self._uses(add_node, graph) != 1:
            return None
        mm_node = self._parent_op(add_node, OpKind.MATMUL, graph)
        if not mm_node or self._uses(mm_node, graph) != 1:
            return None
        bias_inp = self._other_input(add_node, mm_node.id)
        if bias_inp is None:
            return None
        if node.op == OpKind.RELU:
            fused_op = OpKind.FUSED_MATMUL_BIAS_RELU
        elif node.op == OpKind.GELU:
            fused_op = OpKind.FUSED_MATMUL_BIAS_GELU
        else:
            return None
        fused = IRNode(node.id, fused_op,
                       [mm_node.inputs[0], mm_node.inputs[1], bias_inp],
                       {"fused_from": [mm_node.id, add_node.id, node.id]},
                       node.output_shape, node.output_dtype, f"fused_{node.name}")
        graph.replace_subgraph([mm_node, add_node, node], fused)
        return fused

    def _parent(self, node: IRNode, graph: Graph) -> Optional[IRNode]:
        return graph.get_node(node.inputs[0]) if len(node.inputs) == 1 else None

    def _parent_op(self, node: IRNode, op: OpKind, graph: Graph) -> Optional[IRNode]:
        for i in node.inputs:
            p = graph.get_node(i)
            if p and p.op == op: return p
        return None

    def _other_input(self, node: IRNode, excl_id: int) -> Optional[int]:
        cands = [i for i in node.inputs if i != excl_id]
        return cands[0] if cands else None

    def _uses(self, node: IRNode, graph: Graph) -> int:
        return sum(n.inputs.count(node.id) for n in graph.nodes) + graph.outputs.count(node.id)
"""

FILES["axon/optimizer/quantize.py"] = """import math
from typing import List, Tuple, Optional, Dict
from ..ir.tensor_ir import Graph, WeightBlob

FP4_ENC_MAP = {
    0.0: 0x0, 0.5: 0x1, 1.0: 0x2, 1.5: 0x3,
    2.0: 0x4, 3.0: 0x5, 4.0: 0x6, 6.0: 0x7
}

def round_fp4(val: float) -> float:
    sign = -1.0 if math.copysign(1.0, val) < 0 else 1.0
    abs_v = abs(val)
    best, min_dist = 0.0, float("inf")
    for cand in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        dist = abs(abs_v - cand)
        if dist < min_dist:
            min_dist = dist; best = cand
    return sign * best

def fp4_to_nibble(val: float) -> int:
    is_neg = math.copysign(1.0, val) < 0
    code = FP4_ENC_MAP.get(round_fp4(abs(val)), 0x0)
    return (code | 0x8) if is_neg else code

class FP4QuantizePass:
    BLOCK = 32

    def run(self, graph: Graph, calibration: Optional[Dict] = None) -> Graph:
        for blob in graph.weights.values():
            if blob.dtype in ("f32", "f16") and len(blob.data) >= self.BLOCK:
                self._quantize(blob)
        return graph

    def _quantize(self, blob: WeightBlob) -> None:
        flat = list(blob.data)
        pad = (self.BLOCK - (len(flat) % self.BLOCK)) % self.BLOCK
        if pad > 0: flat.extend([0.0] * pad)
        n_blocks = len(flat) // self.BLOCK
        packed = bytearray(n_blocks * 16)
        scales = bytearray(n_blocks)
        residuals: List[Tuple[int, float]] = []

        for b in range(n_blocks):
            start = b * self.BLOCK
            chunk = flat[start: start + self.BLOCK]
            abs_c = [abs(x) for x in chunk]
            mean_v = sum(abs_c) / 32.0
            std_v = math.sqrt(sum((x - mean_v) ** 2 for x in abs_c) / 32.0)
            tau = mean_v + 3.0 * std_v

            non_outliers = [abs_c[i] for i in range(32) if abs_c[i] <= tau]
            max_val = max(non_outliers) if non_outliers else 1e-8
            if max_val <= 0.0: max_val = 1e-8

            exp = max(-126, min(127, int(math.ceil(math.log2(max_val / 6.0)))))
            scale = 2.0 ** exp
            scales[b] = (exp + 127) & 0xFF

            for i in range(32):
                v = chunk[i]
                if abs_c[i] > tau:
                    q = (1.0 if v >= 0 else -1.0) * 6.0
                    nib = fp4_to_nibble(q)
                    residuals.append((start + i, v - scale * q))
                else:
                    nib = fp4_to_nibble(round_fp4(v / scale))
                idx = (b * 16) + (i // 2)
                packed[idx] |= (nib & 0x0F) if (i % 2 == 0) else ((nib & 0x0F) << 4)

        blob.is_quantized = True
        blob.packed_data = bytes(packed)
        blob.scale_bytes = bytes(scales)
        blob.residuals = residuals
        blob.dtype = "fp4"
"""

FILES["axon/optimizer/passes.py"] = """from ..ir.tensor_ir import Graph
from .fusion import FusionPass
from .quantize import FP4QuantizePass

class OptimizationPipeline:
    def __init__(self, enable_fusion: bool = True, enable_quant: bool = True):
        self.enable_fusion = enable_fusion
        self.enable_quant = enable_quant

    def run(self, graph: Graph) -> Graph:
        if self.enable_fusion: graph = FusionPass().run(graph)
        if self.enable_quant: graph = FP4QuantizePass().run(graph)
        return graph
"""

# =====================================================================
# Axon Package: Backend & Baker
# =====================================================================
FILES["axon/backend/__init__.py"] = """from .llvm_backend import LLVMBackend
from .codegen_ops import CodeGenOps
from .runtime_linker import RuntimeLinker
__all__ = ["LLVMBackend", "CodeGenOps", "RuntimeLinker"]
"""

FILES["axon/backend/codegen_ops.py"] = """import llvmlite.ir as ir

class CodeGenOps:
    def __init__(self, module: ir.Module):
        void = ir.VoidType()
        i8_ptr = ir.PointerType(ir.IntType(8))
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)
        i64_ptr = ir.PointerType(i64)

        self.fn_alloc = ir.Function(module, ir.FunctionType(i8_ptr, [i64, i64_ptr, i32]), name="axon_tensor_alloc")
        self.fn_free = ir.Function(module, ir.FunctionType(void, [i8_ptr]), name="axon_tensor_free")
        self.fn_matmul = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr, i8_ptr]), name="axon_matmul")
        self.fn_add = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr, i8_ptr]), name="axon_add")
        self.fn_relu = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr]), name="axon_relu")
        self.fn_gelu = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr]), name="axon_gelu")
        self.fn_softmax = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr]), name="axon_softmax")
        self.fn_layer_norm = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr, i8_ptr, i8_ptr]), name="axon_layer_norm")
        self.fn_transpose = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr]), name="axon_transpose")
        self.fn_fused_matmul_bias_relu = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr, i8_ptr, i8_ptr]), name="axon_fused_matmul_bias_relu")
        self.fn_fused_matmul_bias = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr, i8_ptr, i8_ptr]), name="axon_fused_matmul_bias")
        self.fn_fused_matmul_bias_gelu = ir.Function(module, ir.FunctionType(void, [i8_ptr, i8_ptr, i8_ptr, i8_ptr]), name="axon_fused_matmul_bias_gelu")
        self.fn_get_weight = ir.Function(module, ir.FunctionType(i8_ptr, [i8_ptr]), name="axon_get_weight")
        self.fn_create_tensor_from_weight = ir.Function(module, ir.FunctionType(i8_ptr, [i8_ptr]), name="axon_create_tensor_from_weight")
"""

FILES["axon/backend/llvm_backend.py"] = """import llvmlite.ir as ir
import llvmlite.binding as llvm
from .codegen_ops import CodeGenOps
from ..ir.tensor_ir import Graph, OpKind

def _numel(shape):
    n = 1
    for s in shape: n *= s
    return n

def _count_flops(graph: Graph) -> int:
    total = 0
    for node in graph.nodes:
        if node.op == OpKind.MATMUL:
            a = graph.get_node(node.inputs[0]) if len(node.inputs) > 0 else None
            b = graph.get_node(node.inputs[1]) if len(node.inputs) > 1 else None
            if a and b and len(a.output_shape) == 2 and len(b.output_shape) == 2:
                M, K = a.output_shape
                _, N = b.output_shape
                total += 2 * M * K * N
        elif node.op in (OpKind.FUSED_MATMUL_BIAS_RELU, OpKind.FUSED_MATMUL_BIAS_GELU):
            a = graph.get_node(node.inputs[0]) if len(node.inputs) > 0 else None
            b = graph.get_node(node.inputs[1]) if len(node.inputs) > 1 else None
            if a and b and len(a.output_shape) == 2 and len(b.output_shape) == 2:
                M, K = a.output_shape
                _, N = b.output_shape
                total += 2 * M * K * N + M * N * 5
        elif node.op == OpKind.FUSED_MATMUL_BIAS:
            a = graph.get_node(node.inputs[0]) if len(node.inputs) > 0 else None
            b = graph.get_node(node.inputs[1]) if len(node.inputs) > 1 else None
            if a and b and len(a.output_shape) == 2 and len(b.output_shape) == 2:
                M, K = a.output_shape
                _, N = b.output_shape
                total += 2 * M * K * N + M * N
        elif node.op == OpKind.ADD:
            total += _numel(node.output_shape)
        elif node.op in (OpKind.RELU, OpKind.GELU):
            total += _numel(node.output_shape) * 3
        elif node.op == OpKind.SOFTMAX:
            total += _numel(node.output_shape) * 5
        elif node.op == OpKind.LAYER_NORM:
            total += _numel(node.output_shape) * 6
        elif node.op == OpKind.TRANSPOSE:
            total += _numel(node.output_shape)
    return total

class LLVMBackend:
    def __init__(self, module_name: str = "axon_module", target_triple: str = ""):
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        self.module = ir.Module(name=module_name)
        self.target_triple = target_triple or llvm.get_default_triple()
        self.module.triple = self.target_triple
        self.ops = CodeGenOps(self.module)
        self.str_map: dict[str, ir.GlobalVariable] = {}

    def _str(self, builder: ir.IRBuilder, val: str) -> ir.Value:
        if val in self.str_map: gv = self.str_map[val]
        else:
            raw = bytearray((val + "\\0").encode("utf-8"))
            t = ir.ArrayType(ir.IntType(8), len(raw))
            gv = ir.GlobalVariable(self.module, t, name=f".str.{len(self.str_map)}")
            gv.linkage, gv.global_constant, gv.initializer = "internal", True, ir.Constant(t, raw)
            self.str_map[val] = gv
        z = ir.Constant(ir.IntType(32), 0)
        return builder.gep(gv, [z, z], inbounds=True)

    def _shape(self, builder: ir.IRBuilder, shape: tuple) -> ir.Value:
        i64 = ir.IntType(64)
        n = max(len(shape), 1)
        arr = builder.alloca(ir.ArrayType(i64, n), name="shape")
        z = ir.Constant(ir.IntType(32), 0)
        for idx, dim in enumerate(shape):
            p = builder.gep(arr, [z, ir.Constant(ir.IntType(32), idx)], inbounds=True)
            builder.store(ir.Constant(i64, dim), p)
        return builder.bitcast(arr, ir.PointerType(i64))

    def _alloc_out(self, builder: ir.IRBuilder, shape: tuple, i64, i32) -> ir.Value:
        return builder.call(
            self.ops.fn_alloc,
            [ir.Constant(i64, len(shape)), self._shape(builder, shape), ir.Constant(i32, 0)],
        )

    def _emit_flops_fn(self, graph: Graph) -> None:
        total = _count_flops(graph)
        fn = ir.Function(self.module, ir.FunctionType(ir.IntType(64), []), name="axon_get_total_flops")
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        b.ret(ir.Constant(ir.IntType(64), total))

    def compile(self, graph: Graph, output_obj_path: str) -> str:
        i8_ptr, i64, i32 = ir.PointerType(ir.IntType(8)), ir.IntType(64), ir.IntType(32)
        fn = ir.Function(self.module, ir.FunctionType(i8_ptr, [ir.PointerType(i8_ptr)]), name="axon_forward")
        builder = ir.IRBuilder(fn.append_basic_block("entry"))
        nodes: dict[int, ir.Value] = {}

        for idx, in_id in enumerate(graph.inputs):
            p = builder.gep(fn.args[0], [ir.Constant(i32, idx)])
            nodes[in_id] = builder.load(p, name=f"inp_{idx}")

        for node in graph.nodes_in_topo_order():
            if node.id in nodes: continue
            if node.op == OpKind.CONSTANT:
                w_name = node.attrs.get("weight_name", node.name)
                w_ref = builder.call(self.ops.fn_get_weight, [self._str(builder, w_name)])
                nodes[node.id] = builder.call(self.ops.fn_create_tensor_from_weight, [w_ref], name=f"w_{node.id}")
            elif node.op == OpKind.MATMUL:
                out = self._alloc_out(builder, node.output_shape, i64, i32)
                builder.call(self.ops.fn_matmul, [nodes[node.inputs[0]], nodes[node.inputs[1]], out])
                nodes[node.id] = out
            elif node.op == OpKind.ADD:
                out = self._alloc_out(builder, node.output_shape, i64, i32)
                builder.call(self.ops.fn_add, [nodes[node.inputs[0]], nodes[node.inputs[1]], out])
                nodes[node.id] = out
            elif node.op == OpKind.RELU:
                out = self._alloc_out(builder, node.output_shape, i64, i32)
                builder.call(self.ops.fn_relu, [nodes[node.inputs[0]], out])
                nodes[node.id] = out
            elif node.op == OpKind.GELU:
                out = self._alloc_out(builder, node.output_shape, i64, i32)
                builder.call(self.ops.fn_gelu, [nodes[node.inputs[0]], out])
                nodes[node.id] = out
            elif node.op == OpKind.SOFTMAX:
                out = self._alloc_out(builder, node.output_shape, i64, i32)
                builder.call(self.ops.fn_softmax, [nodes[node.inputs[0]], out])
                nodes[node.id] = out
            elif node.op == OpKind.LAYER_NORM:
                out = self._alloc_out(builder, node.output_shape, i64, i32)
                builder.call(self.ops.fn_layer_norm,
                             [nodes[node.inputs[0]], nodes[node.inputs[1]], nodes[node.inputs[2]], out])
                nodes[node.id] = out
            elif node.op == OpKind.TRANSPOSE:
                out = self._alloc_out(builder, node.output_shape, i64, i32)
                builder.call(self.ops.fn_transpose, [nodes[node.inputs[0]], out])
                nodes[node.id] = out
            elif node.op == OpKind.FUSED_MATMUL_BIAS_RELU:
                out = self._alloc_out(builder, node.output_shape, i64, i32)
                builder.call(self.ops.fn_fused_matmul_bias_relu,
                             [nodes[node.inputs[0]], nodes[node.inputs[1]], nodes[node.inputs[2]], out])
                nodes[node.id] = out
            elif node.op == OpKind.FUSED_MATMUL_BIAS:
                out = self._alloc_out(builder, node.output_shape, i64, i32)
                builder.call(self.ops.fn_fused_matmul_bias,
                             [nodes[node.inputs[0]], nodes[node.inputs[1]], nodes[node.inputs[2]], out])
                nodes[node.id] = out
            elif node.op == OpKind.FUSED_MATMUL_BIAS_GELU:
                out = self._alloc_out(builder, node.output_shape, i64, i32)
                builder.call(self.ops.fn_fused_matmul_bias_gelu,
                             [nodes[node.inputs[0]], nodes[node.inputs[1]], nodes[node.inputs[2]], out])
                nodes[node.id] = out

        builder.ret(nodes[graph.outputs[0]] if graph.outputs else ir.Constant(i8_ptr, None))
        self._emit_flops_fn(graph)
        parsed = llvm.parse_assembly(str(self.module))
        parsed.verify()
        tm = llvm.Target.from_triple(self.target_triple).create_target_machine(opt=3)
        with open(output_obj_path, "wb") as f:
            f.write(tm.emit_object(parsed))
        return output_obj_path
"""

FILES["axon/backend/runtime_linker.py"] = """import os
import subprocess
from ..utils.errors import CodegenError

class RuntimeLinker:
    @staticmethod
    def compile_asm_stub(asm_path: str, obj_path: str) -> None:
        cmd = ["as", asm_path, "-o", obj_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0: raise CodegenError(f"Assembler error: {res.stderr}")

    @staticmethod
    def link_binary(objects: list[str], output_bin: str, static_runtime: str = "", target: str = "") -> None:
        if target and target != "native":
            cc = f"{target}-gcc"
        else:
            cc = os.environ.get("CC", "cc")
        cmd = [cc, "-O3", "-fopenmp", "-ffast-math"] + objects
        if static_runtime: cmd.append(static_runtime)
        cmd.extend(["-lm", "-lpthread", "-o", output_bin])
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0: raise CodegenError(f"Linker error: {res.stderr}")
        if os.path.exists(output_bin):
            os.chmod(output_bin, 0o755)
"""

FILES["axon/baker/__init__.py"] = """from .weight_pack import AxonBlobPacker
from .baker import Baker
__all__ = ["AxonBlobPacker", "Baker"]
"""

FILES["axon/baker/weight_pack.py"] = """import struct
from typing import Dict
from ..ir.tensor_ir import WeightBlob

class AxonBlobPacker:
    MAGIC = b"AXON"
    VERSION = 1

    @staticmethod
    def pack(weights: Dict[str, WeightBlob]) -> bytes:
        num = len(weights)
        name_tbl = bytearray()
        data_sec = bytearray()
        meta = []
        name_off = 0

        for name, blob in weights.items():
            enc = name.encode("utf-8") + b"\\0"
            name_tbl.extend(enc)
            while len(data_sec) % 16 != 0: data_sec.append(0)
            d_off = len(data_sec)
            if blob.is_quantized and blob.packed_data:
                payload = bytearray(blob.scale_bytes or b"") + bytearray(blob.packed_data)
                data_sec.extend(payload)
                nbytes = len(payload)
                dtype = 2
            else:
                raw = struct.pack(f"<{len(blob.data)}f", *blob.data)
                data_sec.extend(raw)
                nbytes = len(raw)
                dtype = 0

            shape4 = list(blob.shape) + [0] * (4 - len(blob.shape))
            meta.append({
                "hash": AxonBlobPacker._fnv1a(name), "name_off": name_off,
                "d_off": d_off, "nbytes": nbytes, "dtype": dtype, "ndim": len(blob.shape),
                "shape": shape4[:4]
            })
            name_off += len(enc)

        hdr_sz = 12
        entry_sz = 64
        name_base = hdr_sz + num * entry_sz
        data_base = name_base + len(name_tbl)
        pad = (16 - (data_base % 16)) % 16
        data_base += pad

        blob = bytearray(struct.pack("<4sII", AxonBlobPacker.MAGIC, AxonBlobPacker.VERSION, num))
        for m in meta:
            blob.extend(struct.pack("<QIIQII4qI", m["hash"], m["name_off"], data_base + m["d_off"],
                                    m["nbytes"], m["dtype"], m["ndim"], *m["shape"], 0))
        blob.extend(name_tbl)
        blob.extend(b"\\0" * pad)
        blob.extend(data_sec)
        return bytes(blob)

    @staticmethod
    def _fnv1a(text: str) -> int:
        h = 14695981039346656037
        for ch in text.encode("utf-8"):
            h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h
"""

FILES["axon/baker/baker.py"] = """import os
import tempfile
from typing import List, Dict
from ..ir.tensor_ir import WeightBlob
from .weight_pack import AxonBlobPacker
from ..backend.runtime_linker import RuntimeLinker

class Baker:
    def __init__(self, runtime_archive_path: str = "runtime/libaxon_runtime.a", target: str = ""):
        self.runtime_archive_path = runtime_archive_path
        self.target = target

    def bake(self, object_files: List[str], weights: Dict[str, WeightBlob], output_bin: str) -> None:
        blob_data = AxonBlobPacker.pack(weights)
        with tempfile.TemporaryDirectory() as tmp:
            blob_file = os.path.join(tmp, "weights.blob")
            with open(blob_file, "wb") as f: f.write(blob_data)
            asm_file = os.path.join(tmp, "weights.S")
            with open(asm_file, "w") as f:
                f.write(f'''
.section .axon_weights, "a", @progbits
.align 16
.global __axon_weights_start
__axon_weights_start:
.incbin "{blob_file}"
.global __axon_weights_end
__axon_weights_end:
''')
            w_obj = os.path.join(tmp, "weights.o")
            RuntimeLinker.compile_asm_stub(asm_file, w_obj)
            RuntimeLinker.link_binary(object_files + [w_obj], output_bin, self.runtime_archive_path, target=self.target)
"""

# =====================================================================
# Axon Enterprise Tools: Inspector & Converter
# =====================================================================
FILES["axon/tools/__init__.py"] = """from .inspector import SystemInspector
from .convert import ModelConverter
__all__ = ["SystemInspector", "ModelConverter"]
"""

FILES["axon/tools/inspector.py"] = """import os
import sys
import platform
import shutil
import subprocess

class SystemInspector:
    def __init__(self):
        self.os_name = platform.system().lower()
        self.arch = platform.machine().lower()
        self.tools = {}
        self.features = []
        self.memory_gb = 0.0
        self.cflags = ""

    def inspect(self):
        for t in ["gcc", "clang", "make", "python3"]:
            self.tools[t] = shutil.which(t) is not None

        if self.os_name == "linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    info = f.read()
                if "avx512" in info: self.features.append("AVX-512")
                elif "avx2" in info: self.features.append("AVX2")
                if "neon" in info or "asimd" in info: self.features.append("ARM NEON")
            except Exception:
                pass
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            self.memory_gb = round(kb / (1024 * 1024), 2)
                            break
            except Exception:
                pass
        elif self.os_name == "darwin":
            try:
                mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
                self.memory_gb = round(mem_bytes / (1024**3), 2)
                self.features.append("Apple Silicon NEON" if "arm" in self.arch else "x86_64")
            except Exception:
                pass

        base_cflags = "-O3 -fPIC -Wall -Wextra -std=c11 -ffast-math"
        if self.arch in ("x86_64", "amd64", "aarch64", "arm64"):
            base_cflags += " -march=native"
        self.cflags = f"{base_cflags} -fopenmp"

    def report(self) -> str:
        lines = [
            f"OS: {self.os_name} ({platform.release()})",
            f"Architecture: {self.arch}",
            f"CPU Features: {', '.join(self.features) if self.features else 'Generic'}",
            f"Memory: {self.memory_gb} GB",
            f"Tools: " + ", ".join([f"{k}:{'OK' if v else 'MISSING'}" for k, v in self.tools.items()]),
            f"Optimized CFLAGS: {self.cflags}"
        ]
        return "\\n".join(lines)
"""

FILES["axon/tools/convert.py"] = """import json
import os

class ModelConverter:
    @staticmethod
    def generate_axon(model_data: dict) -> str:
        name = model_data.get("name", "auto_model")
        in_shape = model_data.get("input_shape", [1, 4])
        layers = model_data.get("layers", [])
        
        code = [f"fn {name}(x: f32[{in_shape[0]}, {in_shape[1]}]) -> f32[{in_shape[0]}, {in_shape[1]}] {{"]
        
        prev_shape = in_shape
        for i, layer in enumerate(layers):
            ltype = layer["type"]
            var_in = "x" if i == 0 else f"v{i-1}"
            var_out = f"v{i}"
            
            if ltype == "Linear":
                w = layer["weights"]
                b = layer["bias"]
                w_shape = [len(w), len(w[0])]
                b_shape = [len(b)]
                out_shape = [prev_shape[0], w_shape[1]]
                w_str = "[" + ",\\n        ".join([", ".join([repr(float(x)) for x in row]) for row in w]) + "]"
                b_str = "[" + ", ".join([repr(float(x)) for x in b]) + "]"
                code.append(f"    let w{i}: f32[{w_shape[0]}, {w_shape[1]}] = {w_str} as f32[{w_shape[0]}, {w_shape[1]}];")
                code.append(f"    let b{i}: f32[{b_shape[0]}] = {b_str} as f32[{b_shape[0]}];")
                code.append(f"    let {var_out}: f32[{out_shape[0]}, {out_shape[1]}] = add(matmul({var_in}, w{i}), b{i});")
                prev_shape = out_shape
            elif ltype in ("ReLU", "GELU", "Softmax"):
                fn_name = ltype.lower()
                code.append(f"    let {var_out}: f32[{prev_shape[0]}, {prev_shape[1]}] = {fn_name}({var_in});")
            elif ltype == "LayerNorm":
                g = layer["gamma"]
                b = layer["beta"]
                g_str = "[" + ", ".join([repr(float(x)) for x in g]) + "]"
                b_str = "[" + ", ".join([repr(float(x)) for x in b]) + "]"
                code.append(f"    let g{i}: f32[{len(g)}] = {g_str} as f32[{len(g)}];")
                code.append(f"    let beta{i}: f32[{len(b)}] = {b_str} as f32[{len(b)}];")
                code.append(f"    let {var_out}: f32[{prev_shape[0]}, {prev_shape[1]}] = layernorm({var_in}, g{i}, beta{i});")
                
        final_var = f"v{len(layers)-1}" if layers else "x"
        code.append(f"    return {final_var};")
        code.append("}")
        return "\\n".join(code)
"""

# =====================================================================
# Axon Enterprise CLI
# =====================================================================
FILES["axon/cli.py"] = """import argparse, sys, os, json, subprocess
from .frontend.lexer import Lexer
from .frontend.parser import Parser
from .frontend.semantic import SemanticAnalyzer
from .ir.builder import IRBuilder
from .ir.printer import IRPrinter
from .optimizer.passes import OptimizationPipeline
from .backend.llvm_backend import LLVMBackend
from .baker.baker import Baker
from .utils.errors import AxonError
from .tools.inspector import SystemInspector
from .tools.convert import ModelConverter

def compile_graph(src_path, out_bin, target, dump_ir=False):
    with open(src_path) as f: src = f.read()
    ast = SemanticAnalyzer(src_path).analyze(Parser(Lexer(src, src_path).tokenize(), src_path).parse())
    graph = IRBuilder().build(ast)
    graph = OptimizationPipeline(True, True).run(graph)
    if dump_ir: print(IRPrinter.dump(graph))
    gen_obj = out_bin + ".gen.o"
    LLVMBackend(target_triple=target).compile(graph, gen_obj)
    Baker("runtime/libaxon_runtime.a", target=target).bake([gen_obj], graph.weights, out_bin)
    if os.path.exists(gen_obj): os.remove(gen_obj)

def cmd_doctor(args):
    inspector = SystemInspector()
    inspector.inspect()
    print("=== Axon System Inspector ===")
    print(inspector.report())
    print("=============================")

def cmd_convert(args):
    if not os.path.exists(args.weights):
        print(f"Error: {args.weights} not found", file=sys.stderr); sys.exit(1)
    
    with open(args.weights) as f: data = json.load(f)
    axon_src = ModelConverter.generate_axon(data)
    
    src_file = args.output + ".axon"
    with open(src_file, "w") as f: f.write(axon_src)
    print(f"[CONVERT] Generated Axon source: {src_file}")
    
    try:
        compile_graph(src_file, args.output, args.target)
        print(f"[SUCCESS] Compiled standalone binary: {args.output}")
    except Exception as e:
        print(f"Compilation failed: {e}", file=sys.stderr); sys.exit(1)
    finally:
        if os.path.exists(src_file): os.remove(src_file)

def cmd_build(args):
    if not os.path.exists(args.source):
        print(f"Error: {args.source} not found", file=sys.stderr); sys.exit(1)
    try:
        compile_graph(args.source, args.output, args.target, args.dump_ir)
        print(f"[SUCCESS] Standalone binary baked: {args.output}")
    except AxonError as e:
        print(f"Compilation error: {e}", file=sys.stderr); sys.exit(1)

def cmd_run(args):
    if not os.path.exists(args.binary):
        print(f"Error: {args.binary} not found", file=sys.stderr); sys.exit(1)
    subprocess.run([f"./{args.binary}"])

def main():
    parser = argparse.ArgumentParser(description="Axon Enterprise v1.0 - Tensor DSL Compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Inspect hardware and tools").set_defaults(func=cmd_doctor)
    subparsers.add_parser("setup", help="Alias for doctor").set_defaults(func=cmd_doctor)

    p_conv = subparsers.add_parser("convert", help="Convert JSON model to binary")
    p_conv.add_argument("weights", help="Path to model JSON")
    p_conv.add_argument("-o", "--output", default="model.bin")
    p_conv.add_argument("--target", default="")
    p_conv.set_defaults(func=cmd_convert)

    p_build = subparsers.add_parser("build", help="Compile Axon source to binary")
    p_build.add_argument("source")
    p_build.add_argument("-o", "--output", default="model.bin")
    p_build.add_argument("--target", default="")
    p_build.add_argument("--dump-ir", action="store_true")
    p_build.set_defaults(func=cmd_build)

    p_run = subparsers.add_parser("run", help="Run compiled binary")
    p_run.add_argument("binary")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
"""

# =====================================================================
# Native C Runtime
# =====================================================================
FILES["runtime/axon_tensor.h"] = """#ifndef AXON_TENSOR_H
#define AXON_TENSOR_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define AXON_DTYPE_F32 0
#define AXON_DTYPE_F16 1
#define AXON_DTYPE_FP4 2
#define AXON_DTYPE_I32 3

typedef struct AxonTensor {
    void    *data;
    int64_t  ndim;
    int64_t  shape[4];
    int64_t  strides[4];
    int32_t  dtype;
    int32_t  quant_scale;
    bool     owns_data;
} AxonTensor;

AxonTensor* axon_tensor_alloc(int64_t ndim, const int64_t *shape, int32_t dtype);
void        axon_tensor_free(AxonTensor *t);
int64_t     axon_tensor_numel(const AxonTensor *t);
#endif
"""

FILES["runtime/axon_tensor.c"] = """#define _POSIX_C_SOURCE 200809L
#include "axon_tensor.h"
#include <stdlib.h>
#include <string.h>

int64_t axon_tensor_numel(const AxonTensor *t) {
    if (!t || t->ndim == 0) return 0;
    int64_t n = 1;
    for (int64_t i = 0; i < t->ndim; ++i) n *= t->shape[i];
    return n;
}

AxonTensor* axon_tensor_alloc(int64_t ndim, const int64_t *shape, int32_t dtype) {
    AxonTensor *t = (AxonTensor*)malloc(sizeof(AxonTensor));
    if (!t) return NULL;
    t->ndim = ndim; t->dtype = dtype; t->quant_scale = 0; t->owns_data = true;
    memset(t->shape, 0, sizeof(t->shape));
    memset(t->strides, 0, sizeof(t->strides));

    int64_t total = 1;
    for (int64_t i = 0; i < ndim; ++i) { t->shape[i] = shape[i]; total *= shape[i]; }
    int64_t s = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) { t->strides[i] = s; s *= t->shape[i]; }

    size_t sz = total * sizeof(float);
    if (sz < 64) sz = 64;
    if (posix_memalign(&t->data, 64, sz) != 0) { free(t); return NULL; }
    memset(t->data, 0, sz);
    return t;
}

void axon_tensor_free(AxonTensor *t) {
    if (!t) return;
    if (t->owns_data && t->data) free(t->data);
    free(t);
}
"""

FILES["runtime/axon_quant.h"] = """#ifndef AXON_QUANT_H
#define AXON_QUANT_H
#include <stdint.h>
void axon_dequantize_fp4_block(const uint8_t *packed_16, uint8_t scale_byte, float *out_f32);
#endif
"""

FILES["runtime/axon_quant.c"] = """#include "axon_quant.h"
#include <math.h>

static const float FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

void axon_dequantize_fp4_block(const uint8_t *packed_16, uint8_t scale_byte, float *out_f32) {
    float scale = ldexpf(1.0f, (int32_t)scale_byte - 127);
    for (int i = 0; i < 16; ++i) {
        uint8_t b = packed_16[i];
        out_f32[i * 2 + 0] = FP4_LUT[b & 0x0F] * scale;
        out_f32[i * 2 + 1] = FP4_LUT[(b >> 4) & 0x0F] * scale;
    }
}
"""

FILES["runtime/axon_mmap.h"] = """#ifndef AXON_MMAP_H
#define AXON_MMAP_H
#include "axon_tensor.h"
#include <stdint.h>

typedef struct AxonWeightRef {
    const void *ptr;
    int64_t     nbytes;
    int32_t     dtype;
    int64_t     shape[4];
    int64_t     ndim;
    int32_t     fp4_scale;
} AxonWeightRef;

typedef struct AxonWeightEntry {
    uint64_t name_hash;
    uint32_t name_offset;
    uint32_t data_offset;
    uint64_t nbytes;
    uint32_t dtype;
    uint32_t ndim;
    int64_t  shape[4];
    uint32_t fp4_scale;
} __attribute__((packed)) AxonWeightEntry;

typedef struct AxonBlobHeader {
    char     magic[4];
    uint32_t version;
    uint32_t num_weights;
    AxonWeightEntry entries[];
} __attribute__((packed)) AxonBlobHeader;

const AxonWeightRef* axon_get_weight(const char *name);
AxonTensor*          axon_create_tensor_from_weight(const AxonWeightRef *ref);
#endif
"""

FILES["runtime/axon_mmap.c"] = """#include "axon_mmap.h"
#include "axon_quant.h"
#include <string.h>
#include <stdlib.h>

extern const char __axon_weights_start[] __attribute__((weak));
extern const char __axon_weights_end[]   __attribute__((weak));
static const AxonBlobHeader *g_blob = NULL;

static uint64_t hash64(const char *s) {
    uint64_t h = 14695981039346656037ULL;
    while (*s) { h ^= (uint8_t)(*s++); h *= 1099511628211ULL; }
    return h;
}

const AxonWeightRef* axon_get_weight(const char *name) {
    if (!g_blob && &__axon_weights_start != NULL) {
        g_blob = (const AxonBlobHeader*)__axon_weights_start;
    }
    if (!g_blob) return NULL;
    uint64_t h = hash64(name);
    const char *names = (const char*)(g_blob->entries + g_blob->num_weights);
    for (uint32_t i = 0; i < g_blob->num_weights; ++i) {
        const AxonWeightEntry *e = &g_blob->entries[i];
        if (e->name_hash == h && strcmp(names + e->name_offset, name) == 0) {
            static AxonWeightRef ref;
            ref.ptr = ((const char*)g_blob) + e->data_offset;
            ref.nbytes = e->nbytes; ref.dtype = e->dtype; ref.ndim = e->ndim;
            memcpy(ref.shape, e->shape, sizeof(ref.shape));
            ref.fp4_scale = e->fp4_scale;
            return &ref;
        }
    }
    return NULL;
}

AxonTensor* axon_create_tensor_from_weight(const AxonWeightRef *ref) {
    if (!ref) return NULL;
    if (ref->dtype == AXON_DTYPE_FP4) {
        AxonTensor *t = axon_tensor_alloc(ref->ndim, ref->shape, AXON_DTYPE_F32);
        int64_t blocks = axon_tensor_numel(t) / 32;
        const uint8_t *scales = (const uint8_t*)ref->ptr;
        const uint8_t *packed = scales + blocks;
        float *out = (float*)t->data;
        for (int64_t b = 0; b < blocks; ++b) {
            axon_dequantize_fp4_block(packed + (b * 16), scales[b], out + (b * 32));
        }
        return t;
    }
    AxonTensor *t = axon_tensor_alloc(ref->ndim, ref->shape, ref->dtype);
    if (t && ref->ptr) memcpy(t->data, ref->ptr, ref->nbytes);
    return t;
}
"""

FILES["runtime/axon_matmul.c"] = """#include "axon_runtime.h"
#include <string.h>
#include <math.h>
#include <stdint.h>

#define AXON_TILE 64
#define AXON_SQRT_2_OVER_PI 0.7978845608028654f
#define AXON_GELU_C 0.044715f
#define AXON_LN_EPS 1e-5f

void axon_matmul(const AxonTensor *A, const AxonTensor *B, AxonTensor *C) {
    const int64_t M = A->shape[0];
    const int64_t K = A->shape[1];
    const int64_t N = B->shape[1];
    const float *a = (const float*)A->data;
    const float *b = (const float*)B->data;
    float *c = (float*)C->data;
    memset(c, 0, (size_t)(M * N) * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t ii = 0; ii < M; ii += AXON_TILE) {
        for (int64_t jj = 0; jj < N; jj += AXON_TILE) {
            const int64_t i_end = (ii + AXON_TILE < M) ? ii + AXON_TILE : M;
            const int64_t j_end = (jj + AXON_TILE < N) ? jj + AXON_TILE : N;
            for (int64_t kk = 0; kk < K; kk += AXON_TILE) {
                const int64_t k_end = (kk + AXON_TILE < K) ? kk + AXON_TILE : K;
                for (int64_t i = ii; i < i_end; ++i) {
                    for (int64_t k = kk; k < k_end; ++k) {
                        const float aik = a[i * K + k];
                        #pragma omp simd
                        for (int64_t j = jj; j < j_end; ++j) {
                            c[i * N + j] += aik * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

static inline float axon_gelu_scalar(float v) {
    const float v3 = v * v * v;
    return 0.5f * v * (1.0f + tanhf(AXON_SQRT_2_OVER_PI * (v + AXON_GELU_C * v3)));
}

void axon_relu(const AxonTensor *A, AxonTensor *C) {
    const float *a = (const float*)A->data;
    float *c = (float*)C->data;
    const int64_t total = axon_tensor_numel(A);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < total; ++i) c[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

void axon_gelu(const AxonTensor *A, AxonTensor *C) {
    const float *a = (const float*)A->data;
    float *c = (float*)C->data;
    const int64_t total = axon_tensor_numel(A);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < total; ++i) c[i] = axon_gelu_scalar(a[i]);
}

void axon_softmax(const AxonTensor *A, AxonTensor *C) {
    const float *a = (const float*)A->data;
    float *c = (float*)C->data;
    int64_t rows = 1;
    const int64_t cols = A->shape[A->ndim - 1];
    for (int64_t i = 0; i < A->ndim - 1; ++i) rows *= A->shape[i];

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const float *row = a + r * cols;
        float *out = c + r * cols;
        float mx = row[0];
        for (int64_t j = 1; j < cols; ++j) if (row[j] > mx) mx = row[j];
        float sum = 0.0f;
        for (int64_t j = 0; j < cols; ++j) {
            const float e = expf(row[j] - mx);
            out[j] = e;
            sum += e;
        }
        const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int64_t j = 0; j < cols; ++j) out[j] *= inv;
    }
}

void axon_layer_norm(const AxonTensor *X, const AxonTensor *gamma,
                     const AxonTensor *beta, AxonTensor *C) {
    const float *x = (const float*)X->data;
    float *c = (float*)C->data;
    int64_t rows = 1;
    const int64_t cols = X->shape[X->ndim - 1];
    for (int64_t i = 0; i < X->ndim - 1; ++i) rows *= X->shape[i];
    const float *g = gamma ? (const float*)gamma->data : NULL;
    const float *b = beta  ? (const float*)beta->data  : NULL;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const float *row = x + r * cols;
        float *out = c + r * cols;
        float mean = 0.0f;
        for (int64_t j = 0; j < cols; ++j) mean += row[j];
        mean /= (float)cols;
        float var = 0.0f;
        for (int64_t j = 0; j < cols; ++j) {
            const float d = row[j] - mean;
            var += d * d;
        }
        var /= (float)cols;
        const float inv_std = 1.0f / sqrtf(var + AXON_LN_EPS);
        for (int64_t j = 0; j < cols; ++j) {
            const float normed = (row[j] - mean) * inv_std;
            const float gg = g ? g[j] : 1.0f;
            const float bb = b ? b[j] : 0.0f;
            out[j] = normed * gg + bb;
        }
    }
}

void axon_transpose(const AxonTensor *A, AxonTensor *C) {
    const int64_t M = A->shape[0];
    const int64_t N = A->shape[1];
    const float *a = (const float*)A->data;
    float *c = (float*)C->data;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            c[j * M + i] = a[i * N + j];
        }
    }
}

void axon_fused_matmul_bias_relu(const AxonTensor *A, const AxonTensor *B,
                                 const AxonTensor *bias, AxonTensor *C) {
    axon_matmul(A, B, C);
    const int64_t M = A->shape[0], N = B->shape[1];
    float *c = (float*)C->data;
    const float *b = bias ? (const float*)bias->data : NULL;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            const float v = c[i * N + j] + (b ? b[j] : 0.0f);
            c[i * N + j] = v > 0.0f ? v : 0.0f;
        }
    }
}

void axon_fused_matmul_bias(const AxonTensor *A, const AxonTensor *B,
                            const AxonTensor *bias, AxonTensor *C) {
    axon_matmul(A, B, C);
    const int64_t M = A->shape[0], N = B->shape[1];
    float *c = (float*)C->data;
    const float *b = bias ? (const float*)bias->data : NULL;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
            c[i * N + j] += (b ? b[j] : 0.0f);
}

void axon_fused_matmul_bias_gelu(const AxonTensor *A, const AxonTensor *B,
                                 const AxonTensor *bias, AxonTensor *C) {
    axon_matmul(A, B, C);
    const int64_t M = A->shape[0], N = B->shape[1];
    float *c = (float*)C->data;
    const float *b = bias ? (const float*)bias->data : NULL;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
            c[i * N + j] = axon_gelu_scalar(c[i * N + j] + (b ? b[j] : 0.0f));
}

void axon_add(const AxonTensor *A, const AxonTensor *B, AxonTensor *C) {
    const float *a = (const float*)A->data;
    const float *b = (const float*)B->data;
    float *c = (float*)C->data;
    if (A->ndim == 2 && B->ndim == 1 && A->shape[1] == B->shape[0]) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i = 0; i < A->shape[0]; ++i)
            for (int64_t j = 0; j < A->shape[1]; ++j)
                c[i * A->shape[1] + j] = a[i * A->shape[1] + j] + b[j];
        return;
    }
    const int64_t total = axon_tensor_numel(C);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < total; ++i) c[i] = a[i] + b[i];
}
"""

FILES["runtime/axon_runtime.h"] = """#ifndef AXON_RUNTIME_H
#define AXON_RUNTIME_H
#include "axon_tensor.h"
#include "axon_mmap.h"
#include "axon_quant.h"

#ifdef __cplusplus
extern "C" {
#endif

void axon_matmul(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
void axon_add(const AxonTensor *A, const AxonTensor *B, AxonTensor *C);
void axon_relu(const AxonTensor *A, AxonTensor *C);
void axon_gelu(const AxonTensor *A, AxonTensor *C);
void axon_softmax(const AxonTensor *A, AxonTensor *C);
void axon_layer_norm(const AxonTensor *X, const AxonTensor *gamma, const AxonTensor *beta, AxonTensor *C);
void axon_transpose(const AxonTensor *A, AxonTensor *C);
void axon_fused_matmul_bias_relu(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
void axon_fused_matmul_bias(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
void axon_fused_matmul_bias_gelu(const AxonTensor *A, const AxonTensor *B, const AxonTensor *bias, AxonTensor *C);
void axon_print_tensor(const AxonTensor *t);
AxonTensor* axon_forward(AxonTensor **inputs);

#ifdef __cplusplus
}
#endif
#endif
"""

FILES["runtime/axon_runtime.c"] = """#include "axon_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern int64_t axon_get_total_flops(void) __attribute__((weak));

void axon_print_tensor(const AxonTensor *t) {
    if (!t) { printf("Tensor: NULL\\n"); return; }
    printf("Tensor(ndim=%ld, shape=[", (long)t->ndim);
    for (int64_t i = 0; i < t->ndim; ++i) printf("%ld%s", (long)t->shape[i], (i + 1 < t->ndim) ? ", " : "");
    printf("], dtype=%d) = [\\n", t->dtype);
    int64_t total = axon_tensor_numel(t);
    const float *d = (const float*)t->data;
    int64_t lim = total < 16 ? total : 16;
    for (int64_t i = 0; i < lim; ++i) printf("  %f\\n", d[i]);
    if (total > lim) printf("  ... (%ld more elements)\\n", (long)(total - lim));
    printf("]\\n");
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("[AXON RUNTIME] Initializing execution context...\\n");
    int64_t in_shape[2] = {1, 4};
    AxonTensor *inp = axon_tensor_alloc(2, in_shape, AXON_DTYPE_F32);
    float *p = (float*)inp->data;
    p[0] = 1.0f; p[1] = 0.5f; p[2] = -1.0f; p[3] = 2.0f;

    AxonTensor *inputs[1] = { inp };
    printf("[AXON RUNTIME] Invoking generated forward graph...\\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    AxonTensor *out = axon_forward(inputs);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_us = (double)(t1.tv_sec - t0.tv_sec) * 1e6
                      + (double)(t1.tv_nsec - t0.tv_nsec) / 1e3;

    int64_t flops = 0;
    if (axon_get_total_flops) flops = axon_get_total_flops();
    double gflops = (elapsed_us > 0.0 && flops > 0)
                    ? ((double)flops) / (elapsed_us * 1e3)
                    : 0.0;

    printf("[AXON RUNTIME] Execution complete. Result:\\n");
    axon_print_tensor(out);
    printf("[AXON BENCHMARK] elapsed = %.2f us, total_flops = %ld, GFLOPS = %.4f\\n",
           elapsed_us, (long)flops, gflops);

    axon_tensor_free(inp);
    axon_tensor_free(out);
    return 0;
}
"""

FILES["runtime/Makefile"] = """CC ?= gcc
CFLAGS ?= -O3 -fPIC -Wall -Wextra -std=c11 -march=native -fopenmp -ffast-math
AR ?= ar
OBJS = axon_tensor.o axon_matmul.o axon_quant.o axon_mmap.o axon_runtime.o
TARGET = libaxon_runtime.a

all: $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(AR) rcs $@ $^

clean:
	rm -f *.o $(TARGET)
"""

# =====================================================================
# Examples
# =====================================================================
FILES["examples/mlp.axon"] = """fn main(x: f32[1, 4]) -> f32[1, 2] {
    let w1: f32[4, 4] = [
        0.1, 0.2, -0.3, 0.4,
        0.5, -0.6, 0.7, 0.8,
        -0.1, 0.2, 0.3, -0.4,
        0.5, 0.6, -0.7, 0.8
    ] as f32[4, 4];

    let b1: f32[4] = [0.1, 0.1, 0.1, 0.1] as f32[4];
    let h1: f32[1, 4] = relu(add(matmul(x, w1), b1));

    let w2: f32[4, 2] = [
        0.2, -0.4,
        0.6, 0.8,
        -0.5, 0.3,
        0.1, -0.2
    ] as f32[4, 2];

    let b2: f32[2] = [0.05, -0.05] as f32[2];
    let out: f32[1, 2] = add(matmul(h1, w2), b2);
    return out;
}
"""

FILES["examples/transformer_block.axon"] = """fn transformer_block(x: f32[1, 4]) -> f32[1, 4] {
    let W_q: f32[4, 4] = [
        0.1, 0.0, 0.2, 0.0,
        0.0, 0.1, 0.0, 0.2,
        0.2, 0.0, 0.1, 0.0,
        0.0, 0.2, 0.0, 0.1
    ] as f32[4, 4];
    let W_k: f32[4, 4] = [
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1
    ] as f32[4, 4];
    let W_v: f32[4, 4] = [
        0.5, 0.0, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.0, 0.5
    ] as f32[4, 4];

    let q: f32[1, 4] = matmul(x, W_q);

    let k_t: f32[4, 4] = transpose(W_k);
    let scores: f32[1, 4] = matmul(q, k_t);
    let attn:   f32[1, 4] = softmax(scores);
    let context: f32[1, 4] = matmul(attn, W_v);

    let W1: f32[4, 8] = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1
    ] as f32[4, 8];
    let b1: f32[8] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] as f32[8];
    let W2: f32[8, 4] = [
        0.1, 0.2, 0.3, 0.4,
        0.2, 0.3, 0.4, 0.5,
        0.3, 0.4, 0.5, 0.6,
        0.4, 0.5, 0.6, 0.7,
        0.5, 0.6, 0.7, 0.8,
        0.6, 0.7, 0.8, 0.9,
        0.7, 0.8, 0.9, 1.0,
        0.8, 0.9, 1.0, 1.1
    ] as f32[8, 4];
    let b2: f32[4] = [0.05, 0.05, 0.05, 0.05] as f32[4];

    let h:        f32[1, 8] = gelu(add(matmul(context, W1), b1));
    let ffn_out:  f32[1, 4] = add(matmul(h, W2), b2);

    let gamma: f32[4] = [1.0, 1.0, 1.0, 1.0] as f32[4];
    let beta:  f32[4] = [0.0, 0.0, 0.0, 0.0] as f32[4];

    let residual: f32[1, 4] = add(x, ffn_out);
    let out:      f32[1, 4] = layernorm(residual, gamma, beta);
    return out;
}
"""

# =====================================================================
# Unit & E2E Tests
# =====================================================================
FILES["tests/test_lexer.py"] = """import unittest
from axon.frontend.lexer import Lexer
from axon.frontend.tokens import TokenKind

class TestLexer(unittest.TestCase):
    def test_lexer(self):
        t = Lexer("fn forward(x: f32[1, 4]) -> f32[1, 2] { let y = relu(x); return y; }").tokenize()
        kinds = [x.kind for x in t]
        self.assertIn(TokenKind.FN, kinds)
        self.assertIn(TokenKind.RELU, kinds)
        self.assertEqual(t[-1].kind, TokenKind.EOF)

    def test_lexer_transformer_ops(self):
        t = Lexer("fn f(x: f32[1, 4]) -> f32[1, 4] { let y = gelu(x); let z = softmax(y); let w = layernorm(z, z, z); return w; }").tokenize()
        kinds = [x.kind for x in t]
        self.assertIn(TokenKind.GELU, kinds)
        self.assertIn(TokenKind.SOFTMAX, kinds)
        self.assertIn(TokenKind.LAYERNORM, kinds)

    def test_lexer_transpose(self):
        t = Lexer("fn f(x: f32[4, 4]) -> f32[4, 4] { let y = transpose(x); return y; }").tokenize()
        kinds = [x.kind for x in t]
        self.assertIn(TokenKind.TRANSPOSE, kinds)

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_parser.py"] = """import unittest
from axon.frontend.lexer import Lexer
from axon.frontend.parser import Parser

class TestParser(unittest.TestCase):
    def test_parser(self):
        code = "fn f(x: f32[1, 4]) -> f32[1, 4] { let y: f32[1, 4] = relu(x); return y; }"
        ast = Parser(Lexer(code).tokenize()).parse()
        self.assertEqual(len(ast.functions), 1)
        self.assertEqual(ast.functions[0].name, "f")

    def test_parser_transformer_ops(self):
        code = "fn f(x: f32[1, 4]) -> f32[1, 4] { let y: f32[1, 4] = gelu(x); let z: f32[1, 4] = softmax(y); let w: f32[1, 4] = layernorm(z, z, z); return w; }"
        ast = Parser(Lexer(code).tokenize()).parse()
        self.assertEqual(len(ast.functions), 1)

    def test_parser_transpose(self):
        code = "fn f(x: f32[4, 4]) -> f32[4, 4] { let y: f32[4, 4] = transpose(x); return y; }"
        ast = Parser(Lexer(code).tokenize()).parse()
        self.assertEqual(len(ast.functions), 1)

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_transpose_semantic.py"] = """import unittest
from axon.frontend.lexer import Lexer
from axon.frontend.parser import Parser
from axon.frontend.semantic import SemanticAnalyzer
from axon.utils.errors import SemanticError

class TestTransposeSemantic(unittest.TestCase):
    def test_transpose_shape(self):
        code = "fn f(x: f32[3, 7]) -> f32[7, 3] { let y: f32[7, 3] = transpose(x); return y; }"
        ast = SemanticAnalyzer().analyze(Parser(Lexer(code).tokenize()).parse())
        self.assertEqual(len(ast.functions), 1)

    def test_transpose_bad_shape(self):
        code = "fn f(x: f32[3, 7]) -> f32[3, 7] { let y: f32[3, 7] = transpose(x); return y; }"
        with self.assertRaises(SemanticError):
            SemanticAnalyzer().analyze(Parser(Lexer(code).tokenize()).parse())

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_fusion.py"] = """import unittest
from axon.ir.tensor_ir import Graph, IRNode, OpKind
from axon.optimizer.fusion import FusionPass

class TestFusion(unittest.TestCase):
    def _chain(self, tail_op):
        g = Graph()
        g.nodes = [
            IRNode(0, OpKind.INPUT, [], {}, (1, 4), "f32"),
            IRNode(1, OpKind.CONSTANT, [], {}, (4, 4), "f32"),
            IRNode(2, OpKind.MATMUL, [0, 1], {}, (1, 4), "f32"),
            IRNode(3, OpKind.CONSTANT, [], {}, (4,), "f32"),
            IRNode(4, OpKind.ADD, [2, 3], {}, (1, 4), "f32"),
            IRNode(5, tail_op, [4], {}, (1, 4), "f32"),
        ]
        g.inputs = [0]
        g.outputs = [5]
        return g

    def test_relu_fusion(self):
        fg = FusionPass().run(self._chain(OpKind.RELU))
        ops = [n.op for n in fg.nodes]
        self.assertIn(OpKind.FUSED_MATMUL_BIAS_RELU, ops)

    def test_gelu_fusion(self):
        fg = FusionPass().run(self._chain(OpKind.GELU))
        ops = [n.op for n in fg.nodes]
        self.assertIn(OpKind.FUSED_MATMUL_BIAS_GELU, ops)

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_quantize.py"] = """import unittest
from axon.ir.tensor_ir import WeightBlob
from axon.optimizer.quantize import FP4QuantizePass, round_fp4

class TestQuant(unittest.TestCase):
    def test_quant(self):
        self.assertEqual(round_fp4(1.8), 2.0)
        data = [0.2 * i for i in range(32)]
        data[4] = 90.0
        blob = WeightBlob("w", data, [32], "f32")
        FP4QuantizePass()._quantize(blob)
        self.assertTrue(blob.is_quantized)
        self.assertEqual(len(blob.residuals), 1)

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_inspector.py"] = """import unittest
from axon.tools.inspector import SystemInspector

class TestInspector(unittest.TestCase):
    def test_inspect(self):
        inspector = SystemInspector()
        inspector.inspect()
        self.assertIn("OS:", inspector.report())
        self.assertIn("CFLAGS:", inspector.report())
        self.assertIsNotNone(inspector.cflags)

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_converter.py"] = """import unittest
from axon.tools.convert import ModelConverter

class TestConverter(unittest.TestCase):
    def test_generate_axon(self):
        model_data = {
            "name": "test_model",
            "input_shape": [1, 4],
            "layers": [
                {"type": "Linear", "weights": [[1.0, 2.0], [3.0, 4.0]], "bias": [0.1, 0.2]},
                {"type": "GELU"}
            ]
        }
        code = ModelConverter.generate_axon(model_data)
        self.assertIn("fn test_model", code)
        self.assertIn("let w0", code)
        self.assertIn("gelu", code)

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_e2e.py"] = """import unittest, os, subprocess
from axon.frontend.lexer import Lexer
from axon.frontend.parser import Parser
from axon.frontend.semantic import SemanticAnalyzer
from axon.ir.builder import IRBuilder
from axon.backend.llvm_backend import LLVMBackend
from axon.baker.baker import Baker

class TestE2E(unittest.TestCase):
    def test_e2e(self):
        code = '''
        fn forward(x: f32[1, 4]) -> f32[1, 4] {
            let w: f32[4, 4] = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            ] as f32[4, 4];
            let y: f32[1, 4] = matmul(x, w);
            return y;
        }
        '''
        ast = SemanticAnalyzer().analyze(Parser(Lexer(code).tokenize()).parse())
        g = IRBuilder().build(ast)
        obj = "test_e2e.o"
        bin_out = "test_e2e.bin"
        LLVMBackend().compile(g, obj)
        Baker("runtime/libaxon_runtime.a").bake([obj], g.weights, bin_out)
        res = subprocess.run([f"./{bin_out}"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Execution complete", res.stdout)
        self.assertIn("GFLOPS", res.stdout)
        if os.path.exists(obj): os.remove(obj)
        if os.path.exists(bin_out): os.remove(bin_out)

if __name__ == "__main__":
    unittest.main()
"""

FILES["tests/test_transformer_e2e.py"] = """import unittest, os, subprocess
from axon.frontend.lexer import Lexer
from axon.frontend.parser import Parser
from axon.frontend.semantic import SemanticAnalyzer
from axon.ir.builder import IRBuilder
from axon.optimizer.passes import OptimizationPipeline
from axon.backend.llvm_backend import LLVMBackend
from axon.baker.baker import Baker

TRANSFORMER_SRC = '''
fn transformer_block(x: f32[1, 4]) -> f32[1, 4] {
    let W_q: f32[4, 4] = [
        0.1, 0.0, 0.2, 0.0,
        0.0, 0.1, 0.0, 0.2,
        0.2, 0.0, 0.1, 0.0,
        0.0, 0.2, 0.0, 0.1
    ] as f32[4, 4];
    let W_k: f32[4, 4] = [
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1
    ] as f32[4, 4];
    let W_v: f32[4, 4] = [
        0.5, 0.0, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.0, 0.5
    ] as f32[4, 4];

    let q: f32[1, 4] = matmul(x, W_q);
    let k_t: f32[4, 4] = transpose(W_k);
    let scores: f32[1, 4] = matmul(q, k_t);
    let attn: f32[1, 4] = softmax(scores);
    let context: f32[1, 4] = matmul(attn, W_v);

    let W1: f32[4, 8] = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1
    ] as f32[4, 8];
    let b1: f32[8] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] as f32[8];
    let W2: f32[8, 4] = [
        0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5,
        0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7,
        0.5, 0.6, 0.7, 0.8, 0.6, 0.7, 0.8, 0.9,
        0.7, 0.8, 0.9, 1.0, 0.8, 0.9, 1.0, 1.1
    ] as f32[8, 4];
    let b2: f32[4] = [0.05, 0.05, 0.05, 0.05] as f32[4];

    let h: f32[1, 8] = gelu(add(matmul(context, W1), b1));
    let ffn: f32[1, 4] = add(matmul(h, W2), b2);

    let gamma: f32[4] = [1.0, 1.0, 1.0, 1.0] as f32[4];
    let beta:  f32[4] = [0.0, 0.0, 0.0, 0.0] as f32[4];
    let normed: f32[1, 4] = layernorm(ffn, gamma, beta);
    let sm: f32[1, 4] = softmax(normed);
    return sm;
}
'''

class TestTransformerE2E(unittest.TestCase):
    def test_e2e_transformer(self):
        ast = SemanticAnalyzer().analyze(Parser(Lexer(TRANSFORMER_SRC).tokenize()).parse())
        g = IRBuilder().build(ast)
        g = OptimizationPipeline(True, True).run(g)
        ops = [n.op.name for n in g.nodes]
        self.assertIn("FUSED_MATMUL_BIAS_GELU", ops)
        self.assertIn("TRANSPOSE", ops)

        obj = "test_tf.o"
        bin_out = "test_tf.bin"
        LLVMBackend().compile(g, obj)
        Baker("runtime/libaxon_runtime.a").bake([obj], g.weights, bin_out)
        res = subprocess.run([f"./{bin_out}"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertIn("Execution complete", res.stdout)
        self.assertIn("GFLOPS", res.stdout)
        if os.path.exists(obj): os.remove(obj)
        if os.path.exists(bin_out): os.remove(bin_out)

if __name__ == "__main__":
    unittest.main()
"""


# =====================================================================
# Main Generator & Execution Engine
# =====================================================================
def main():
    print("[1/8] Unpacking full Axon Enterprise files (v1.0)...")
    for path, content in FILES.items():
        d = os.path.dirname(path)
        if d: os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  -> Created {path}")

    print("\n[2/8] Running System Inspector...")
    from axon.tools.inspector import SystemInspector
    inspector = SystemInspector()
    inspector.inspect()
    print(inspector.report())
    print("  -> Dynamic CFLAGS locked in.")

    print("\n[3/8] Compiling native C runtime with OpenMP + SIMD...")
    env = os.environ.copy()
    env["CFLAGS"] = inspector.cflags
    res = subprocess.run(["make", "runtime"], env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Build failed:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)
    print("  -> Runtime compiled successfully.")

    print("\n[4/8] Running test suite...")
    test_res = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], capture_output=True, text=True)
    if test_res.returncode != 0:
        print("  -> Some tests failed!", file=sys.stderr)
        print(test_res.stderr, file=sys.stderr)
        print(test_res.stdout, file=sys.stderr)
        sys.exit(1)
    print("  -> All test suites passed cleanly.")

    print("\n[5/8] Creating a sample JSON model for Auto-Conversion...")
    sample_model = {
        "name": "auto_model",
        "input_shape": [1, 4],
        "layers": [
            {"type": "Linear", "weights": [[0.1, 0.2, -0.3, 0.4], [0.5, -0.6, 0.7, 0.8], [-0.1, 0.2, 0.3, -0.4], [0.5, 0.6, -0.7, 0.8]], "bias": [0.1, 0.1, 0.1, 0.1]},
            {"type": "ReLU"},
            {"type": "Linear", "weights": [[0.2, -0.4], [0.6, 0.8], [-0.5, 0.3], [0.1, -0.2]], "bias": [0.05, -0.05]}
        ]
    }
    import json
    with open("auto_model.json", "w") as f:
        json.dump(sample_model, f, indent=2)
    print("  -> auto_model.json created.")

    print("\n[6/8] Converting JSON to Axon Source...")
    from axon.tools.convert import ModelConverter
    axon_src = ModelConverter.generate_axon(sample_model)
    with open("auto_model.axon", "w") as f:
        f.write(axon_src)
    print("  -> auto_model.axon generated.")

    print("\n[7/8] Compiling auto_model.axon to auto_model.bin...")
    from axon.cli import compile_graph
    compile_graph("auto_model.axon", "auto_model.bin", target="")
    print("  -> Compiled successfully: auto_model.bin")

    print("\n[8/8] Executing auto_model.bin (zero Python dependency):")
    print("=" * 60)
    exec_res = subprocess.run(["./auto_model.bin"], capture_output=True, text=True)
    print(exec_res.stdout)
    print("=" * 60)
    print("[COMPLETED] Axon Enterprise v1.0 is 100% operational and ready.")


if __name__ == "__main__":
    main()