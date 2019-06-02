#include "util.hxx"
#include <taco.h>
#include <taco/parser/lexer.h>
#include <taco/parser/parser.h>
#include <taco/lower/lower.h>

using namespace taco;

struct format_t {
  const char* tensor_name;
  const char* format;
};

struct options_t {
  std::vector<format_t> formats;
};

inline ir::Stmt lower_taco_kernel(const char* pattern, 
  const options_t& options, const char* func_name) {

  using namespace taco;

  // Options for compilation.
  std::map<std::string, Format> formats;
  std::map<std::string, Datatype> datatypes;
  std::map<std::string, std::vector<int> > dimensions;
  std::map<std::string, TensorBase> tensors;

  for(format_t format : options.formats) {
    std::vector<ModeFormatPack> modePacks;
    int len = strlen(format.format);
    for(int i = 0; i < len; ++i) {
      switch(format.format[i]) {
        case 'd': 
          modePacks.push_back({ ModeFormat::Dense });
          break;

        case 's': 
          modePacks.push_back({ ModeFormat::Sparse });
          break;

        case 'u':
          modePacks.push_back({ ModeFormat::Sparse(ModeFormat::NOT_UNIQUE) });

        case 'c': 
          modePacks.push_back({ ModeFormat::Singleton(ModeFormat::NOT_UNIQUE) });
          break;

        case 'q':
          modePacks.push_back({ ModeFormat::Singleton+ });
          break;
      }
    }

    formats.insert({
      format.tensor_name,
      Format(std::move(modePacks))
    });
  }  

  // Make a simple parser. Need to add a larger default dimension or else the
  // code generator defaults to the small dimension of 5.
  parser::Parser parser(pattern, formats, datatypes, dimensions, tensors, 1000);
  parser.parse();

  // Parse the pattern.
  const TensorBase& tensor = parser.getResultTensor();

  // Lower the parsed tensor to IR.
  std::set<old::Property> computeProperties;
  computeProperties.insert(old::Compute);
  ir::Stmt compute = old::lower(tensor.getAssignment(), func_name, 
    computeProperties, tensor.getAllocSize());

  return compute;
}

////////////////////////////////////////////////////////////////////////////////
// Generate C++ expression from taco ir::Expr tree.
// Expanded expression macros are replaced by their return argument.

template<Datatype::Kind kind>
struct type_from_datatype_t {
  // For a quick and dirty map from TACO's Datatype enum to real types,
  // make a typed enum that associates each enumerator name with its 
  // actual type.
  enum typename class my_enum_t {
    Bool       = bool,
    UInt8      = uint8_t,
    UInt16     = uint16_t,
    UInt32     = uint32_t,
    UInt64     = uint64_t,
    UInt128    = void,        // Don't support these types
    Int8       = int8_t,
    Int16      = int16_t,
    Int32      = int32_t,
    Int64      = int64_t,
    Int128     = void,        // Don't support these types
    Float32    = float,
    Float64    = double,
    Complex64  = std::complex<float>,
    Complex128 = std::complex<double>
  };

  // Use introspection to get the spelling of the Datatype::Kind enumerator
  // @enum_name(kind)

  // Use that to look up the corresponding enumerator in my_enum_t:
  // my_enum_t::@(@enum_name(kind))

  // Then use @type_enum to extract the type from that typed-enum:
  typedef @enum_type(my_enum_t::@(@enum_name(kind))) type_t;
};

// ir::Var uses a very simple type system.
// 1) is_ptr tells you if it's a pointer.
// 2) is_tensor tells you if it's a taco_tensor_t. If not, it's the type
//    in Datatype ExprNode::type.

// struct Var : public ExprNode<Var> {
// public:
//   std::string name;
//   bool is_ptr;
//   bool is_tensor;
// 
//   static Expr make(std::string name, Datatype type, bool is_ptr=false, 
//                    bool is_tensor=false);
// 
//   static const IRNodeType _type_info = IRNodeType::Var;
// };
template<Datatype::Kind kind, bool is_ptr, bool is_tensor>
struct type_from_var_t {
  typedef typename type_from_datatype_t<kind>::type_t type_t;
};

template<Datatype::Kind kind, bool is_tensor>
struct type_from_var_t<kind, true, is_tensor> {
  typedef typename type_from_var_t<kind, false, is_tensor>::type_t* type_t;
};

template<Datatype::Kind kind>
struct type_from_var_t<kind, false, true> {
  typedef taco_tensor_t type_t;
};

// A convenience macro that takes a TACO Var and returns the @mtype holding
// that type. The caller uses @static_type to unpack the @mtype and treat it as
// a type-id.
@macro auto mtype_from_var(const ir::Var* var) {
  return @dynamic_type(
    typename type_from_var_t<
      @meta var->type.getKind(), 
      var->is_ptr, 
      var->is_tensor
    >::type_t
  );
}

////////////////////////////////////////////////////////////////////////////////

@macro auto expr_inject(const ir::Expr& expr) {
  @meta+ if(const ir::Literal* literal = expr.as<ir::Literal>()) {
    @emit return @expression(util::toString(expr));

  } else if(const ir::Var* var = expr.as<ir::Var>()) {
    @emit return @(var->name);

  } else if(const ir::Neg* neg = expr.as<ir::Neg>()) {
    @emit return -expr_inject(add->a);

  } else if(const ir::Sqrt* sqrt = expr.as<ir::Sqrt>()) {
    @emit return sqrt(expr_inject(add->a));

  } else if(const ir::Add* add = expr.as<ir::Add>()) {
    @emit return expr_inject(add->a) + expr_inject(add->b);

  } else if(const ir::Sub* sub = expr.as<ir::Sub>()) {
    @emit return expr_inject(sub->a) - expr_inject(sub->b);

  } else if(const ir::Mul* mul = expr.as<ir::Mul>()) {
    @emit return expr_inject(mul->a) * expr_inject(mul->b);

  } else if(const ir::Div* div = expr.as<ir::Div>()) {
    @emit return expr_inject(div->a) / expr_inject(div->b);

  } else if(const ir::Rem* rem = expr.as<ir::Rem>()) {
    @emit return expr_inject(rem->a) % expr_inject(rem->b);

  } else if(const ir::Min* min = expr.as<ir::Min>()) {
    static_assert(2 == min->operands.size(), "only 2-operand min supported");
    @emit return std::min(
      expr_inject(min->operands[0]), 
      expr_inject(min->operands[1])
    );

  } else if(const ir::Max* max = expr.as<ir::Max>()) {
    @emit return std::max(expr_inject(max->a), expr_inject(max->b));

  } else if(const ir::BitAnd* bit_and = expr.as<ir::BitAnd>()) {
    @emit return expr_inject(bit_and->a) & expr_inject(bit_and->b);

  } else if(const ir::BitOr* bit_or = expr.as<ir::BitOr>()) {
    @emit return expr_inject(bit_or->a) | expr_inject(bit_or->b);

  } else if(const ir::Eq* eq = expr.as<ir::Eq>()) {
    @emit return expr_inject(eq->a) == expr_inject(eq->b);

  } else if(const ir::Neq* neq = expr.as<ir::Neq>()) {
    @emit return expr_inject(neq->a) != expr_inject(neq->b);

  } else if(const ir::Gt* gt = expr.as<ir::Gt>()) {
    @emit return expr_inject(gt->a) > expr_inject(gt->b);

  } else if(const ir::Lt* lt = expr.as<ir::Lt>()) {
    @emit return expr_inject(lt->a) < expr_inject(lt->b);

  } else if(const ir::Gte* gte = expr.as<ir::Gte>()) {
    @emit return expr_inject(gte->a) >= expr_inject(gte->b);

  } else if(const ir::Lte* lte = expr.as<ir::Lte>()) {
    @emit return expr_inject(lte->a) <= expr_inject(lte->b);

  } else if(const ir::And* and_ = expr.as<ir::And>()) {
    @emit return expr_inject(and_->a) && expr_inject(and_->b);

  } else if(const ir::Or* or_ = expr.as<ir::Or>()) {
    @emit return expr_inject(or_->a) || expr_inject(or_->b);

  } else if(const ir::Cast* cast = expr.as<ir::Cast>()) {
    @emit return 
      (typename type_from_datatype_t<@meta cast->type.getKind()>::type_t)
      expr_inject(cast->a);
  
  } else if(const ir::Load* load = expr.as<ir::Load>()) {
    @emit return expr_inject(load->arr)[expr_inject(load->loc)];

  } else if(const ir::GetProperty* prop = expr.as<ir::GetProperty>()) {
    @emit return expr_prop(prop);

  } else {
    @meta std::string error = format("unsupported expr kind '%s'", 
      @enum_name(expr.ptr->type_info()));
    static_assert(false, error);
  }
}

@macro auto expr_prop(const ir::GetProperty* prop) {

  @meta+ if(ir::TensorProperty::Order == prop->property) {
    // taco_tensor_t::order
    @emit return expr_inject(prop->tensor)->order;

  } else if(ir::TensorProperty::Dimension == prop->property) {
    // taco_tensor_t::dimensions[prop->mode]
    @emit return expr_inject(prop->tensor)->dimensions[prop->mode];

  } else if(ir::TensorProperty::ComponentSize == prop->property) {
    // taco_tensor_t::csize
    @emit return expr_inject(prop->tensor)->csize;

  } else if(ir::TensorProperty::ModeOrdering == prop->property) {
    // taco_tensor_t::mode_ordering
    @emit return expr_inject(prop->tensor)->mode_ordering;

  } else if(ir::TensorProperty::Values == prop->property) {
    // taco_tensor_t::vals
    @emit return (double*)(expr_inject(prop->tensor)->vals);

  } else if(ir::TensorProperty::ValuesSize == prop->property) {
    // taco_tensor_t::vals_size
    @emit return expr_inject(prop->tensor)->vals_size;

  } else if(ir::TensorProperty::Indices == prop->property) {
    // taco_tensor_t::indices[prop->mode][prop->index]
    @emit return expr_inject(prop->tensor)->indices[prop->mode][prop->index];

  } else {
    @meta std::string error = format("unknown tensor property: '%s'", 
      @enum_name(prop->property));
    static_assert(false, error);
  }
}


////////////////////////////////////////////////////////////////////////////////
// Generate C++ statements from taco ir::Stmt tree.
// Statement macros are expanded in 

@macro void stmt_inject(const ir::Stmt& stmt) {

  @meta+ if(const ir::Case* case_ = stmt.as<ir::Case>()) {
    // TACO's Case statement isn't a C++ case statement at all. It's a sequence
    // of if/else statements all chained together.
    @macro stmt_case(case_, 0);

  } else if(const ir::Store* store = stmt.as<ir::Store>()) {
    // store->arr[store->loc] = store->data
    @emit expr_inject(store->arr)[expr_inject(store->loc)] = 
      expr_inject(store->data);
  
  } else if(const ir::For* for_ = stmt.as<ir::For>()) {
    const ir::Var* var = for_->var.as<ir::Var>();
    @emit for(
      @static_type(mtype_from_var(var)) @(var->name) = 
        expr_inject(for_->start); 
      @(var->name) < expr_inject(for_->end); 
      @(var->name) += expr_inject(for_->increment)
    ) {
      @macro stmt_inject(for_->contents);
    }

  } else if(const ir::While* while_ = stmt.as<ir::While>()) {
    @emit while(expr_inject(while_->cond)) {
      @macro stmt_inject(while_->contents);
    }

  } else if(const ir::Block* block = stmt.as<ir::Block>()) {
    // Inject each individual statement.
    for(const ir::Stmt& stmt : block->contents)
      @macro stmt_inject(stmt);

  } else if(const ir::Scope* scope = stmt.as<ir::Scope>()) {
    @macro stmt_inject(scope->scopedStmt);    

  } else if(const ir::VarDecl* var_decl = stmt.as<ir::VarDecl>()) {
    const ir::Var* var = var_decl->var.as<ir::Var>();
    @emit @static_type(mtype_from_var(var)) @(var->name) = 
      expr_inject(var_decl->rhs);

  } else if(const ir::Assign* assign = stmt.as<ir::Assign>()) {
    // assign->lhs = assign->rhs.
    @emit expr_inject(assign->lhs) = expr_inject(assign->rhs);

  } else {
    @meta std::string error = format("unsupported stmt kind '%s'", 
      @enum_name(stmt.ptr->type_info()));
    static_assert(false, error);
  }
} 

@macro void stmt_case(const ir::Case* case_, size_t index) {
  if(expr_inject(case_->clauses[index].first)) {
    @macro stmt_inject(case_->clauses[index].second);

  } else {
    // Recursively build the else case.
    @meta if(index + 1 < case_->clauses.size())
      @macro stmt_case(case_, index + 1);
  }
}

////////////////////////////////////////////////////////////////////////////////

@macro void gen_kernel(const ir::Function* function) {
  @meta const char* name = function->name.c_str();
  @meta std::string struct_name = format("%s_t", name);

  @macro void make_member(const std::vector<ir::Expr>& body) {
    @meta for(const ir::Expr& expr : body) {
      @meta const ir::Var* var = expr.as<ir::Var>();
      @static_type(mtype_from_var(var)) @(var->name);
    }
  }
  struct @(struct_name) {
    @macro make_member(function->inputs);
    @macro make_member(function->outputs);
  };

  // Define a function. The parameters are each of the members of the
  // structure defined above.
  void @(name)(@expand_params(@(struct_name))) {
    @meta printf("Creating %s function\n", name);

    // Inject the body. We expect the body to be a Scope, representing the
    // function's braces.
    @macro stmt_inject(function->body);    
  }
}

// Increment the counter with each TACO formula, so we get unique 
// function names.
@meta int kernel_counter = 0;

template<typename... args_t>
@meta void call_taco(@meta const char* pattern, @meta const options_t& options,
  args_t&&... args) {

  // Generate a unique function name for each call_taco. The taco IR
  // will include this name in its data structures.
  @meta std::string name = format("compute_%d", ++kernel_counter);

  // Parse the pattern and lower to IR.
  @meta ir::Stmt stmt = lower_taco_kernel(pattern, options, name.c_str());

  // Generate the function in the taco_kernel namespace.
  @meta const ir::Function* function = stmt.as<ir::Function>();

  // Expand the gen_kernel in namespace taco_kernel. 
  @macro namespace(taco_kernel) gen_kernel(function);

  // Call the function. Pass each argument tensor by address.
  taco_kernel::@(name)(&args...);
}

int main() {
  taco_tensor_t a { }, b { }, c { };

  // Declare each tensor as 1D and sparse.
  @meta options_t options { };
  @meta options.formats.push_back({ "a", "s" });
  @meta options.formats.push_back({ "b", "s" });
  @meta options.formats.push_back({ "c", "s" });

  call_taco("a(i) = b(i) + c(i)", options, a, b, c);

/*
  call_taco("a(i) = b(i) + c(i)", options, a, b, c);
  taco_tensor_t y { }, M { }, x { };
  @meta options_t options { };
  call_taco("y(i) = M(i, j) * x(i)", options, y, M, x);
*/
  return 0;
}