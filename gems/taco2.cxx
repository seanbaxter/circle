#include "util.hxx"
#include <taco.h>
#include <taco/parser/lexer.h>
#include <taco/parser/parser.h>
#include <taco/lower/lower.h>
#include <../../taco/src/codegen/codegen.h>  // This should be in include/taco

inline std::string gen_taco_kernel(const char* pattern, const char* func_name) {
  using namespace taco;

  // Options for compilation that we aren't using.
  std::map<std::string, Format> formats;
  std::map<std::string, Datatype> datatypes;
  std::map<std::string, std::vector<int> > dimensions;
  std::map<std::string, TensorBase> tensors;

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

  // Stream the C code to oss.
  std::ostringstream oss;
  std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(oss, 
    ir::CodeGen::C99Implementation);

  // Compile the IR to C code.
  codegen->compile(compute);

  return oss.str();
}

@meta int kernel_counter = 0;

template<typename... args_t>
@meta void call_taco(@meta const char* pattern, args_t&&... args) {
  // Generate a unique function name for each call_taco.
  @meta std::string name = format("compute_%d", ++kernel_counter);

  // Execute the gen_taco_kernel at compile time.
  @meta std::string code = gen_taco_kernel(pattern, name.c_str());

  // Print the emitted kernel.
  @meta printf("%s\n", code.c_str());

  // Inject the code into namespace taco_kernel.
  @statements namespace(taco_kernel)(code, name);

  // Call the function. Pass each argument tensor by address.
  taco_kernel::@(name)(&args...);
}

int main() {
  taco_tensor_t v { }, M { }, x { }, N { };

  call_taco("y(i) = M(i, j) * x(i)", v, M, x);

  return 0;
}
