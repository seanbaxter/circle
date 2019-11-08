// http://www.circle-lang.org/
// A Circle language example using an open source dynamic BNF parser to 
// implement a simple 

// The parser:
// https://github.com/yhirose/cpp-peglib

#include <cstdio>
#include <stdexcept>
#include "peglib.h"

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

@macro auto peg_dsl_eval(const char* text) {
  @meta std::shared_ptr<peg::Ast> ast;
  @meta if(peg_parser.parse(text, ast)) {
    // Generate code for the returned AST as an inline expression in the
    // calling expression.
    return peg_dsl_eval(*ast);

  } else
    @meta throw std::runtime_error("syntax error in PEG expression")
}

////////////////////////////////////////////////////////////////////////////////

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