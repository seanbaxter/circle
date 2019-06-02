#include "util.hxx"
#include <taco.h>
#include <taco/parser/lexer.h>
#include <taco/parser/parser.h>
#include <taco/lower/lower.h>
#include <../../taco/src/codegen/codegen.h>  // This should be in include/taco

struct format_t {
  const char* tensor_name;
  const char* format;
};

struct options_t {
  std::vector<format_t> formats;
};

inline std::string gen_taco_kernel(const char* pattern, 
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
          break;
          
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
@meta void call_taco(@meta const char* pattern, @meta const options_t& options, 
  args_t&&... args) {

  // Generate a unique function name for each call_taco.
  @meta std::string name = format("compute_%d", ++kernel_counter);

  // Execute the gen_taco_kernel at compile time.
  @meta std::string code = gen_taco_kernel(pattern, options, name.c_str());

  // Print the emitted kernel.
  @meta printf("%s\n", code.c_str());

  // Inject the code into namespace taco_kernel.
  @statements namespace(taco_kernel)(code, name);

  // Call the function. Pass each argument tensor by address.
  taco_kernel::@(name)(&args...);
}

// TACO relies on macros TACO_MIN and TACO_MAX. Define them as alias to
// std::min and std::max
#define TACO_MIN std::min
#define TACO_MAX std::max

int main() {
  taco_tensor_t a { }, b { }, c { };

  // Declare each tensor as 1D and sparse.
  @meta options_t options { };
  @meta options.formats.push_back({ "a", "s" });
  @meta options.formats.push_back({ "b", "s" });
  @meta options.formats.push_back({ "c", "s" });

  call_taco("a(i) = b(i) + c(i)", options, a, b, c);

  return 0;
}
