#include <cmath>
#include <memory>
#include <iostream>
#include <sstream>
#include <stack>

namespace rpn {

enum class kind_t {
  var,       // A variable or number
  op,        // + - * /
  f1,        // unary function
  f2,        // binary function
};

struct token_t {
  kind_t kind;
  std::string text;
};

struct token_name_t {
  kind_t kind;
  const char* string;
};

const token_name_t token_names[] {
  // Supported with @op
  { kind_t::op,    "+"     },
  { kind_t::op,    "-"     },
  { kind_t::op,    "*"     },
  { kind_t::op,    "/"     },

  // Unary functions
  { kind_t::f1,    "abs"   },
  { kind_t::f1,    "exp"   },
  { kind_t::f1,    "log"   },
  { kind_t::f1,    "sqrt"  },
  { kind_t::f1,    "sin"   },
  { kind_t::f1,    "cos"   },
  { kind_t::f1,    "tan"   },
  { kind_t::f1,    "asin"  },
  { kind_t::f1,    "acos"  },
  { kind_t::f1,    "atan"  },

  // Binary functions
  { kind_t::f2,    "atan2" },
  { kind_t::f2,    "pow"   },
};

inline kind_t find_token_kind(const char* text) {
  for(token_name_t name : token_names) {
    if(!strcmp(text, name.string))
      return name.kind;
  }
  return kind_t::var;
}

struct node_t;
typedef std::unique_ptr<node_t> node_ptr_t;

struct node_t {
  kind_t kind;
  std::string text;
  node_ptr_t a, b;
};

inline node_ptr_t parse(const char* text) {
  std::istringstream iss(text);
  std::string token;

  std::stack<node_ptr_t> stack;
  while(iss>> token) {
    // Make operator ^ call pow.
    if("^" == token)
      token = "pow";

    // Match against any of the known functions.
    kind_t kind = find_token_kind(token.c_str());

    node_ptr_t node = std::make_unique<node_t>();
    node->kind = kind;
    node->text = token;
    
    switch(kind) {
      case kind_t::var:
        // Do nothing before pushing the node.
        break;      

      case kind_t::f1:
        if(stack.size() < 1)
          throw std::range_error("RPN formula is invalid");

        node->a = std::move(stack.top()); stack.pop();
        break;

      case kind_t::op:
      case kind_t::f2:
        // Binary operators and functions pop two items from the stack,
        // combine them, and push the result.
        if(stack.size() < 2)
          throw std::range_error("RPN formula is invalid");

        node->b = std::move(stack.top()); stack.pop();
        node->a = std::move(stack.top()); stack.pop();
        break;
    }

    stack.push(std::move(node));
  }

  if(1 != stack.size())
    throw std::range_error("RPN formula is invalid");

  return std::move(stack.top());
}

@mauto eval_node(const node_t* node) {
  @meta+ if(rpn::kind_t::var == node->kind) {
    @emit return @@expression(node->text);

  } else if(rpn::kind_t::op == node->kind) {
    @emit return @op(
      node->text, 
      rpn::eval_node(node->a.get()),
      rpn::eval_node(node->b.get())
    );

  } else if(rpn::kind_t::f1 == node->kind) {
    // Call a unary function.
    @emit return @(node->text)(
      rpn::eval_node(node->a.get())
    );

  } else if(rpn::kind_t::f2 == node->kind) {
    // Call a binary function.
    @emit return @(node->text)(
      rpn::eval_node(node->a.get()),
      rpn::eval_node(node->b.get())
    );
  }
}

// Define an expression macro for evaluating the RPN expression. This
// expands into the expression context from which it is called. It can have
// any number of meta statements, but only one real return statement, and no
// real declarations (because such declarations are prohibited inside
// expressions).
@mauto eval(const char* text) {
  @meta rpn::node_ptr_t node = rpn::parse(text);
  return rpn::eval_node(node.get());
}

} // namespace rpn

double my_function(double x, double y, double z) {
  return rpn::eval("z 1 x / sin y * ^");
}

int main() {
  double result = my_function(.3, .6, .9);
  printf("%f\n", result);
  return 0;
}