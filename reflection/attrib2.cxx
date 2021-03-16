typedef uint GLuint;

using type    [[attribute]]    = typename;
using binding [[attribute]]    = int;
using set     [[attribute(0)]] = int;

enum class storage_class_t {
  uniform,
  buffer, 
  readonly
};

// shader_decl is the primary variable template for coining shader 
// interface variables from attributes.
template<auto x>
void shader_decl;

// Partial template for all three storage classes. Use a requires-clause
// to constrain each partial template to one storage class.
template<auto x>
requires(storage_class_t::uniform == @attribute(x, storage_class_t))
[[spirv::uniform(@attribute(x, binding), @attribute(x, set))]]
@tattribute(x, type) shader_decl<x>;

template<auto x>
requires(storage_class_t::buffer == @attribute(x, storage_class_t))
[[spirv::buffer(@attribute(x, binding), @attribute(x, set))]]
@tattribute(x, type) shader_decl<x>;

template<auto x>
requires(storage_class_t::readonly == @attribute(x, storage_class_t))
[[using spirv: buffer(@attribute(x, binding), @attribute(x, set)), readonly]]
@tattribute(x, type) shader_decl<x>;

// Write a shader that takes its data as a template parameter.
template<typename data_t>
[[spirv::vert]] 
void vert_shader() {
  // Use the attributes on data_t::vertices to declare a shader variable,
  // then load from that and copy to vertex output.
  glvert_Output.Position = shader_decl<&data_t::vertices>[glvert_VertexID];
}

// Imagine your pipeline's data fitting in these variables:
struct simple_pipeline_t {
  [[.storage_class_t=uniform, .type=sampler2D, .binding=3, .set=1]]
  GLuint texture0;

  [[.storage_class_t=readonly, .type=vec4[], .binding=1 /*default set is 0*/]]
  GLuint vertices;
};

// Generate a shader for simple_pipeline_t.
template void vert_shader<simple_pipeline_t>() asm("vert");
