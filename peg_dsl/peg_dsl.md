# Implementing a DSL using an open source dynamic PEG parser

Building DSLs has been a is a particular focus for me. We've already covered the topic from these angles:

1. [Generating tensor contractions with Circle and TACO
](https://github.com/seanbaxter/circle/blob/master/gems/taco.md)
1. [Reverse-mode automatic differentiation with Circle and Apex](https://github.com/seanbaxter/apex/blob/master/examples/autodiff.md)
1. [RPN as an embedded Circle compiler](https://github.com/seanbaxter/circle/blob/master/gems/rpn.md)

This example uses an opensource parser generator to convert the DSL text into AST, before lowering the AST to code using Circle macros.

[cpp-peglib](https://github.com/yhirose/cpp-peglib) is a dynamic PEG (parse expression grammar) parser. [PEG](https://en.wikipedia.org/wiki/Parsing_expression_grammar) is essentially a way to interpret [EBNF grammars](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) such that the left rule in an alternative is always matched before the right rule to eliminate parse ambiguity. PEG grammars allow the mechanical generation of a recursive descent implementation from an EBNF specification.

## cpp-peglib

cpp-peglib accepts grammars specified as EBNF in a string. The library parses the EBNF grammar into a data structure which serves as a traversable hierarchical representation of the grammar. Semantic actions may be attached to each production. These are invoked on each match, and allow the user of the library to perform some action, like evaluate and accumulate a subexpression or construct an AST node.

cpp-peglib includes an AST feature which automatically attaches AST node-returning semantic actions on each rule. This is not a bespoke AST like the [expression AST in Apex](https://github.com/seanbaxter/apex/blob/master/include/apex/parse.hxx), but it's still quite easy to consume.

[**calc3.cc**](https://github.com/yhirose/cpp-peglib/blob/master/example/calc3.cc)
```cpp
//
//  calc3.cc
//
//  Copyright (c) 2015 Yuji Hirose. All rights reserved.
//  MIT License
//

#include <peglib.h>
#include <iostream>
#include <cstdlib>

using namespace peg;
using namespace std;

int main(int argc, const char** argv)
{
    if (argc < 2 || string("--help") == argv[1]) {
        cout << "usage: calc3 [formula]" << endl;
        return 1;
    }

    function<long (const Ast&)> eval = [&](const Ast& ast) {
        if (ast.name == "NUMBER") {
            return stol(ast.token);
        } else {
            const auto& nodes = ast.nodes;
            auto result = eval(*nodes[0]);
            for (auto i = 1u; i < nodes.size(); i += 2) {
                auto num = eval(*nodes[i + 1]);
                auto ope = nodes[i]->token[0];
                switch (ope) {
                    case '+': result += num; break;
                    case '-': result -= num; break;
                    case '*': result *= num; break;
                    case '/': result /= num; break;
                }
            }
            return result;
        }
    };

    parser parser(R"(
        EXPRESSION       <-  TERM (TERM_OPERATOR TERM)*
        TERM             <-  FACTOR (FACTOR_OPERATOR FACTOR)*
        FACTOR           <-  NUMBER / '(' EXPRESSION ')'

        TERM_OPERATOR    <-  < [-+] >
        FACTOR_OPERATOR  <-  < [/*] >
        NUMBER           <-  < [0-9]+ >

        %whitespace      <-  [ \t\r\n]*
    )");

    parser.enable_ast();

    auto expr = argv[1];
    shared_ptr<Ast> ast;
    if (parser.parse(expr, ast)) {
        ast = AstOptimizer(true).optimize(ast);
        cout << ast_to_s(ast);
        cout << expr << " = " << eval(*ast) << endl;
        return 0;
    }

    cout << "syntax error..." << endl;

    return -1;
}
```

The library's author has provided this simple demonstration of AST generation and AST consumption in cpp-peglib. The heart of the program is the EBNF grammar, which is provided in a string literal. This resembles the kind of input YACC takes, but the interface and parsing technology are very different. cpp-peglib doesn't require a prebuild step to transform the grammar into C++ parsing code; rather, it consumes the grammar at runtime and constructs a data structure for assisting in the parse. The `enable_ast` call attaches AST semantic actions to each of the rules in the grammar. When the `parse` member function is invoked, the example attempts to match the input string (specified at the command line) against the grammar, and builds AST nodes for each successful match.

The second phase evaluates the AST (which is usually a tree-like data structure) by visiting each node, returning values from the terminals (NUMBER) and folding values in the interior nodes (EXPRESSION, TERM and FACTOR rules). Since the AST actions are generic, they don't do any transformations of the token text. The `eval` function converts NUMBER token spellings to integers using the `stol` standard library function. TERM nodes are organized as odd-length arrays of nodes. The even nodes are FACTORs (which may be NUMBERs or other TERM nodes) and the odd nodes match TERM_OPERATOR or FACTOR_OPERATOR. For example, 1 + 2 * 3 / 4 - 5 + 6 has seven nodes for the EXPRESSION rule: 1, +, 2 * 3 / 4, -, 5, + and 6. The multiplicative expression 2 * 3 / 4 has five nodes for the TERM rule: 2, \*, 3, / and 4.

The eval function traverses the nodes array from left-to-right (because these arithmetic operations are left-associative), and accumulates the value into the result object. The odd nodes specify the operators. The even nodes the values.

## Using cpp-peglib for compile-time parsing and DSL implementation

In this example, I'll take calc3.cc, extend the grammer to match identifiers (such as objects and parameters in the C++ code), and call into the parser to implement an embedded DSL. Then, with that embedded DSL, we'll code up some new functions expressed in that language, which get translated into Circle and from that into LLVM IR. We'll use cpp-peglib _without modification_, executing the entire parser pipeline in Circle's integrated interpreter.

[**peg_dsl.cxx**](peg_dsl.cxx)
```cpp
// Define a simple grammar to do a 5-function integer calculation.
@meta peg::parser peg_parser(R"(
  EXPRESSION       <-  TERM (TERM_OPERATOR TERM)*
  TERM             <-  FACTOR (FACTOR_OPERATOR FACTOR)*
  FACTOR           <-  NUMBER / IDENTIFIER / '(' EXPRESSION ')'

  TERM_OPERATOR    <-  < [-+] >
  FACTOR_OPERATOR  <-  < [/*%] >
  NUMBER           <-  < [0-9]+ >
  IDENTIFIER       <-  < [_a-zA-Z] [_0-9a-zA-Z]* >

  %whitespace      <-  [ \t\r\n]*
)");

// peg-cpplib attaches semantic actions to each rule to construct an AST.
@meta peg_parser.enable_ast();
```

The first step is to extend the calc3.cc's grammar with an IDENTIFIER rule and to add the modulus operator % to the list of FACTOR_OPERATORs. Because we're doing compile-time parsing, the EBNF string literal initializes a global object with meta lifetime. We then call `enable_ast` to set AST-yielding semantic actions on each of the rules. When the translation unit is done, the parser object will be destroyed, and no remnant of cpp-peglib will remain in the final executable. 

```cpp
long dsl_function(long x, long y) {
  // This function has a DSL implementation. The DSL text is parsed and 
  // lowered to code when dsl_function is translated. By the time it is called,
  // any remnant of peg-cpplib is gone, and only the LLVM IR or AST remains.
  return x * peg_dsl_eval("5 * x + 3 * y + 2 * x * (y % 3)") + y;
}

int main(int argc, char** argv) {
  if(3 != argc) {
    printf("Usage: peg_dsl [x] [y]\n");
    exit(1);
  }

  int x = atoi(argv[1]);
  int y = atoi(argv[2]);

  int z = dsl_function(x, y);
  printf("result is %d\n", z);

  return 0;
}
```

We can write functions in this calc3 grammar, and evaluate them as normal expressions by calling `peg_dsl_eval`. The identifiers x and y in the DSL match the IDENTIFIER rule. `peg_dsl_eval` is an expression macro, which gets expanded in the scope of the calling expression. When it encounters these IDENTIFIER nodes, it can perform name lookup and match them with the corresponding function parameters. This demonstrates the integration of the hosting C++ and the hosted DSL--we can use the scripting language qualities of Circle to freely pass information between the two systems.

```cpp
template<typename node_t>
@macro auto peg_dsl_fold(const node_t* nodes, size_t count);

@macro auto peg_dsl_eval(const peg::Ast& ast) {
  @meta if(ast.name == "NUMBER") {
    // Put @meta at the start of an expression to force stol's evaluation
    // at compile time, which is when ast->token is available. This will turn
    // the token spelling into an integral constant at compile time.
    return (@meta stol(ast.token));

  } else @meta if(ast.name == "IDENTIFIER") {
    // Evaluate the identifier in the context of the calling scope.
    // This will find the function parameters x and y in dsl_function and
    // yield lvalues of them.
    return @expression(ast.token);

  } else {
    // We have a sequence of nodes that need to be folded. Because this is an
    // automatically-generated AST, we just have an array of nodes where 
    // the even nodes are FACTOR and the odd nodes are OPERATORs.
    // A bespoke AST would left-associate the array into a binary tree for
    // evaluation that more explicitly models precedence.

    @meta const auto& nodes = ast.nodes;
    return peg_dsl_fold(nodes.data(), nodes.size());
  }
}

@macro auto peg_dsl_eval(const char* text) {
  @meta std::shared_ptr<peg::Ast> ast;
  @meta if(peg_parser.parse(text, ast)) {
    // Generate code for the returned AST as an inline expression in the
    // calling expression.
    return peg_dsl_eval(*ast);

  } else
    @meta throw std::runtime_error("syntax error in PEG expression")
}
```

The magic happens from inside `peg_dsl_eval`. The `const char* text` overload is the entry-point for the DSL. It calls the `parse` member function on the meta `peg_parser` object on the user-supplied DSL text. This text must be known at compile time, although it needn't be a string literal. The programmer could cobble a string together using string formatting tools or load it from a file.

The AST overload of `peg_dsl_eval` lowers the AST to real code. There are three basic cases here:

1. We've hit a NUMBER node. This is a textually-coded integer. We want to convert this to an integral constant, so we use [stol](https://en.cppreference.com/w/cpp/string/basic_string/stol) at compile time. We put the @meta token at the start of the expression to force its evaluation at compile time.
1. We've hit an IDENTIFIER node. This needs to map to an actual object or parameter in our C++ code. We'll simply return `@expression(ast.token)`, which causes the token spelling to be lexed, parsed and translated as a C++ expression. The result object is an lvalue of the variable identified in the DSL text.
1. We've hit a non-terminal. These include rules like EXPRESSION, TERM and FACTOR. They consist of sequences of other AST nodes, interspersed by operators with the same level of precedence. We'll need to fold this array together like a binary tree with left-associativity, and without the use of temporary variables for accumulation. Remember that the expression macro is modelling an expression--the only real statement we can have is a _return-statement_, the argument of which is detached and inserted semantically into the calling environment. There's no place inside an expression to declare a new object (although materialization may in effect create objects with temporary lifetimes), so we'll need to be a bit clever in how we lower these non-terminal nodes.

```cpp
template<typename node_t>
@macro auto peg_dsl_fold(const node_t* nodes, size_t count) {
  static_assert(1 & count, "expected odd number of nodes in peg_dsl_fold");

  // We want to left-associate a run of expressions.

  @meta if(1 == count) {
    // If we're at a terminal in the expression, evaluate the FACTOR and
    //  return it.
    return peg_dsl_eval(*nodes[0]);

  } else {
    // Keep descending until we're at a terminal. To left associate, fold
    // everything on the left with the element on the right. For the
    // expression
    //   a * b / c % d * e    this is evaluated as 
    //   (a * b / c % d) * e, 
    // where the part in () gets recursively processed by peg_dsl_fold.

    // Do a left-associative descent.

    // Since the DSL has operators with the same token spellings as C++,
    // we can just use @op to handle them all generically, instead of switching
    // over each token type.
    return @op(
      nodes[count - 2]->token,
      peg_dsl_fold(nodes, count - 2),
      peg_dsl_eval(*nodes[count - 1])
    );
  }
}
```

`peg_dsl_fold` recursively lowers the non-terminal AST nodes. If there is just one input node, it must be a terminal, so we call into `peg_dsl_eval` to generate its subexpression and return the result object. If there's more than one node, we apply a left-associative recursive fold. We'll combine everything on the left with the node on the right, using the right-most operator. This has the same effect as the left-to-right evaluation in calc3, but it can be done without any accumulator object. The resulting expression is translated exactly like its C++ equivalent.

```
$ circle peg_dsl.cxx -filetype=ll -O0 -console

define i64 @_Z12dsl_functionll(i64, i64) {
  %3 = alloca i64, align 8
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  store i64 %0, i64* %3, align 8
  store i64 %1, i64* %4, align 8
  %6 = load i64, i64* %3, align 8
  %7 = load i64, i64* %3, align 8
  %8 = mul nsw i64 5, %7
  %9 = load i64, i64* %4, align 8
  %10 = mul nsw i64 3, %9
  %11 = add nsw i64 %8, %10
  %12 = load i64, i64* %3, align 8
  %13 = mul nsw i64 2, %12
  %14 = load i64, i64* %4, align 8
  %15 = srem i64 %14, 3
  %16 = mul nsw i64 %13, %15
  %17 = add nsw i64 %11, %16
  %18 = mul nsw i64 %6, %17
  %19 = load i64, i64* %4, align 8
  %20 = add nsw i64 %18, %19
  store i64 %20, i64* %5, align 8
  %21 = load i64, i64* %5, align 8
  ret i64 %21
}
```

Template metaprogramming involves massive amounts of indirection, requiring advanced optimization to inline through dozens of layers of cruft. The programmer has to operate on faith that the effect they intended is lowered to something reasonable. 

Circle is different. It allows very efficient code generation. The LLVM IR above is the _unoptimized_ implementation of `dsl_function`. There are no function calls and no temporaries. Function parameters x and y are loaded from the stack whenever referenced, and all arithmetic from then out is performed entirely in register. Circle gives you total control over the subexpressions generated from data-driven expression macros.

Convenient single-header libraries like cpp-peglib are easy to learn and to use. You can quickly add rules to the grammar, add lowering capability to the macros, and build the DSL to help solve your programming challenges.

The best part of "Circle as a scripting language" is that we get tons of capability without having to do any work. You can download an opensource library like cpp-peglib and use it at compile time without any modification. Metaprogramming in C++ usually involves rewriting any tools in the language of constexpr and partial templates. You are hamstrung as a programmer, the complexity grows, and compiler errors become indecipherable.

With Circle, find a tool you like (it can even be a command line tool), or write a fresh one using ordinary C++ constructs, then include it in your project and use the integrated interpreter to access it during source translation. Circle gives you unrestricted access to the host environment (i.e. call any function, execute any shell command), so you gain much of the convenience of a glue language, but unified in a frontend with Standard C++. The result of running your "script" is a C++ binary.
