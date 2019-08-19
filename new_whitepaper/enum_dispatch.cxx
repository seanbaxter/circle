#include <type_traits>
#include <cassert>
#include <iostream>

struct ast_literal_t;
struct ast_string_t;
struct ast_unary_t;
struct ast_binary_t;
struct ast_call_t;

struct ast_t {
  // Associate the base class's enums with each concrete type.
  enum typename kind_t {
    kind_literal = ast_literal_t,
    kind_string  = ast_string_t,
    kind_unary   = ast_unary_t,
    kind_binary  = ast_binary_t,
    kind_call    = ast_call_t,
  } kind;

  ast_t(kind_t kind) : kind(kind) { }

  template<typename type_t>
  bool isa() const {
    // Find the enumerator in kind_t that has the associated type type_t.
    static_assert(std::is_base_of<ast_t, type_t>::value);
    return @type_enum(kind_t, type_t) == kind;
  }

  // Perform an unconditional downcast from ast_t to a derived type.
  // This is like llvm::cast.
  template<typename type_t>
  type_t* cast() {
    assert(isa<type_t>());
    return static_cast<type_t*>(this);
  }

  // Perform a conditional downcast. This is like llvm::dyn_cast.
  template<typename type_t>
  type_t* dyn_cast() {
    return isa<type_t>() ? cast<type_t>() : nullptr;
  }
};

struct ast_literal_t : ast_t {
  ast_literal_t() : ast_t(kind_literal) { }
};

struct ast_string_t : ast_t {
  ast_string_t() : ast_t(kind_string) { }
};

struct ast_unary_t : ast_t {
  ast_unary_t() : ast_t(kind_unary) { }
};

struct ast_binary_t : ast_t {
  ast_binary_t() : ast_t(kind_binary) { }
};

struct ast_call_t : ast_t {
  ast_call_t() : ast_t(kind_call) { }
};

// Generate code to visit each concrete type from this base type.
void visit_ast(ast_t* ast) {

  template<typename type_t>
  @macro void forward_declare() {
    // Forward declare visit_ast on this type.
    void visit_ast(type_t* derived);
  }

  switch(ast->kind) {
    @meta for enum(auto e : ast_t::kind_t) {
      case e: 
        // Forward declare a function on the derived type in the global ns.
        @macro namespace() forward_declare<@enum_type(e)>();

        // Downcast the AST pointer and call the just-declared function.
        return visit_ast(ast->cast<@enum_type(e)>());
    }
  }
}

// Automatically generate handlers for each concrete type.
// In a real application, these are hand-coded and perform 
// type-specific actions.
@meta for enum(auto e : ast_t::kind_t) {
  void visit_ast(@enum_type(e)* type) {
    std::cout<< "visit_ast("<< @type_name(@enum_type(e))<< "*) called\n";
  }
}

int main() {
  ast_binary_t binary;
  visit_ast(&binary);

  ast_call_t call;
  visit_ast(&call);

  return 0;
}