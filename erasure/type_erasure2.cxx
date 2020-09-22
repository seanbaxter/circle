// A Circle implementation of the type erasure tactic implemented here:
// https://github.com/TartanLlama/typeclasses/blob/master/typeclass.hpp

#include "../gems/util.hxx"
#include <memory>
#include <cstring>

// model_t is the base class for impl_t. impl_t has the storage for the 
// object of type_t. model_t has a virtual dtor to trigger impl_t's dtor.
// model_t has a virtual clone function to copy-construct an instance of 
// impl_t into heap memory, which is returned via unique_ptr. model_t has
// a pure virtual function for each method in the interface class typeclass.
template<typename typeclass>
struct model_t {
  virtual ~model_t() { }

  virtual std::unique_ptr<model_t> clone() = 0;

  // Loop over each member function on the interface.
  @meta for(int i = 0; i < @method_count(typeclass); ++i) {

    @meta std::string func_name = @method_name(typeclass, i);

    // Declare a "has_" function.
    virtual bool @("has_", func_name)() const = 0;

    // Declare a pure virtual function for each interface method.
    virtual @func_decl(@method_type(typeclass, i), func_name, args) = 0;
  }
};

template<typename typeclass, typename type_t>
struct impl_t : public model_t<typeclass> {

  // Construct the embedded concrete type.
  template<typename... args_t>
  impl_t(args_t&&... args) : concrete(std::forward<args_t>(args)...) { }

  std::unique_ptr<model_t<typeclass> > clone() override {
    // Copy-construct a new instance of impl_t on the heap.
    return std::make_unique<impl_t>(concrete);
  }
 
  // Loop over each member function on the interface.
  @meta for(int i = 0; i < @method_count(typeclass); ++i) {

    @meta std::string func_name = @method_name(typeclass, i);

    static constexpr bool is_valid = @sfinae(
      std::declval<type_t>().@(func_name)(
        std::declval<@method_params(typeclass, i)>()...
      )
    );

    @meta printf("%d: is_valid = %d\n", i, is_valid);

    // Implement the has_XXX function.
    bool @("has_", func_name)() const override {
      return is_valid;
    }

    // Declare an override function with the same signature as the pure virtual
    // function in model_t.
    @func_decl(@method_type(typeclass, i), func_name, args) override {

      @meta if(is_valid || @sfinae(typeclass::required::@(__func__))) {
        // Forward to the correspondingly-named member function in type_t.
        return concrete.@(__func__)(std::forward<decltype(args)>(args)...);

      } else {

        // We could also call __cxa_pure_virtual or std::terminate here.
        throw std::runtime_error(@string(format("%s::%s not implemented", 
          @type_string(type_t), __func__
        )));
      }
    }
  }

  // Our actual data.
  type_t concrete;
};

////////////////////////////////////////////////////////////////////////////////
// var_t is an 8-byte type that serves as the common wrapper for the 
// type-erasure model_t. It implements move 

template<typename typeclass>
struct var_t {
  // Default initializer creates an empty var_t.
  var_t() = default;

  // Allow initialization from a unique_ptr.
  var_t(std::unique_ptr<model_t<typeclass> >&& model) : 
    model(std::move(model)) { }

  // Move ctor/assign by default.
  var_t(var_t&&) = default;
  var_t& operator=(var_t&&) = default;

  // Call clone for copy ctor/assign.
  var_t(const var_t& rhs) {
    if(rhs)
      model = rhs.model->clone();
  }

  var_t& operator=(const var_t& rhs) {
    model.reset();
    if(rhs)
      model = rhs.model->clone();
    return *this;
  }

  // A virtual dtor triggers the dtor in the impl.
  virtual ~var_t() { }

  // The preferred initializer for a var_t. This constructs an impl_t of
  // type_t on the heap, and stores the pointer in a new var_t.
  template<typename type_t, typename... args_t>
  static var_t construct(args_t&&... args) {
    return var_t(std::make_unique<impl_t<typeclass, type_t> >(
      std::forward<args_t>(args)...
    ));
  }

  // Loop over each member function on the interface.
  @meta for(int i = 0; i < @method_count(typeclass); ++i) {

    // Define a has_XXX member function.
    bool @("has_", @method_name(typeclass, i))() const {
      @meta if(@sfinae(typeclass::required::@(@method_name(typeclass, i))))
        return true;
      else
        return model->@(__func__)();
    }

    // Declare a non-virtual forwarding function for each interface method.
    @func_decl(@method_type(typeclass, i), @method_name(typeclass, i), args) {
      // Forward to the model's virtual function.
      return model->@(__func__)(std::forward<decltype(args)>(args)...);
    }
  }

  explicit operator bool() const {
    return (bool)model;
  }

  // This is actually a unique_ptr to an impl type. We store a pointer to 
  // the base type and rely on model_t's virtual dtor to free the object.
  std::unique_ptr<model_t<typeclass> > model;
}; 


////////////////////////////////////////////////////////////////////////////////
// The var_t class template is specialized to include all member functions in
// my_interface. It makes forwarding calls from these to the virtual 
// functions in model_t.

struct my_interface {
  enum class required {
    print         // Only the print method is required.
  };

  void print(const char* text);
  void save(const char* filename, const char* access);
};

// Print the text in forward order.
struct forward_t {
  void print(const char* text) {
    puts(text);
  }

  void save(const char* filename, const char* access) {
    puts("forward_t::save called");
  }
};

// Print the text in reverse order.
struct reverse_t {
  void print(const char* text) {
    int len = strlen(text);
    for(int i = 0; i < len; ++i)
      putchar(text[len - 1 - i]);
    putchar('\n');
  }
};

// Print the text with caps.
struct allcaps_t {
  void print(const char* text) {
    while(char c = *text++)
      putchar(toupper(c));
    putchar('\n');
  }
};

// The typedef helps emphasize that we have a single type that encompasses
// multiple impl types that aren't related by inheritance.
typedef var_t<my_interface> obj_t;

int main() {

  // Construct an object a.
  obj_t a = obj_t::construct<allcaps_t>();
  a.print("Hello a");

  // Copy-construc a to get b.
  obj_t b = a; 
  b.print("Hello b");

  if(b.has_save())
    b.save("my.save", "w");

  // Copy-assign a to get c.
  obj_t c;
  c = b;
  c.print("Hello c");

  // Create a forward object.
  obj_t d = obj_t::construct<forward_t>();
  d.print("Hello d");
  d.save("foo.save", "w");

  // Create a reverse object.
  obj_t e = obj_t::construct<reverse_t>();
  e.print("Hello e");

  // Throws:
  // terminate called after throwing an instance of 'std::runtime_error'
  //   what():  reverse_t::save not implemented
  e.save("bar.save", "w");

  return 0;
}