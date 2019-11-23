#include "../include/format.hxx"

template<typename... types_t>
struct variant_t {
  enum { count = sizeof...(types_t) };

  enum typename class tag_t : unsigned char {
    // Create an enumerator for each type.
    @meta for(int i = 0; i < count; ++i) {
      static_assert(!std::is_reference<types_t...[i]>::value, 
        "variant member cannot be a reference type");
      static_assert(!std::is_const<types_t...[i]>::value,
        "variant member cannot be const");
      static_assert(!std::is_volatile<types_t...[i]>::value,
        "variant member cannot be volatile");

      types_t...[i];
    }

    // The last enumerator indicates no value. No variant member of this type
    // is constructed, which is why it's safe to set it to type void.
    none = void
  };

  //////////////////////////////////////////////////////////////////////////////
  // variant_t ctors and assignment operators.

  // We need to provide an explicit constructor to allow non-trivial types as
  // variant members.
  variant_t();
  ~variant_t();

  // Copy and move ctors.
  variant_t(const variant_t& rhs);
  variant_t(variant_t&& rhs);

  // Copy and move assignment.
  variant_t& operator=(const variant_t& rhs);
  variant_t& operator=(variant_t&& rhs);

  //////////////////////////////////////////////////////////////////////////////
  // Variant member ctors and assignment operators.

  // Constructors for setting specific variant members.
  template<typename type_t>
  variant_t(const type_t& item);

  template<typename type_t>
  variant_t(type_t&& item);

  template<typename type_t>
  type_t& operator=(const type_t& rhs);

  template<typename type_t>
  type_t& operator=(type_t&& rhs);

  //////////////////////////////////////////////////////////////////////////////
  // Variant member access operators.

  // The user has read-only access to the tag.
  tag_t tag() const { return _tag; }

  // operator bool() evaluates to true if any variant member is set. 
  // It doesn't check the value of the variant member.
  explicit operator bool() const {
    return tag_t::none != _tag;
  }

  // Clear the value.
  void reset();

  // Get the value by ordinal. If the variant member is not active, an assert
  // is triggered.
  template<size_t I>
  auto& get();

  template<size_t I>
  const auto& get() const;

  // Return a reference to the specified type. It must be tagged as the 
  // active variant member, or else this function will assert.
  template<typename type_t>
  type_t& get();

  template<typename type_t>
  const type_t& get() const;

  // Return a pointer to the specified type only if it is the active 
  // variant member. Otherwise return nullptr.
  template<typename type_t>
  type_t* safe_get();

  template<typename type_t>
  const type_t* safe_get() const;

  // Invoke the function object for each variant member. 
  template<typename func_t>
  auto visit(func_t func);

private:
  tag_t _tag = tag_t::none;

  // Put everything in a union.
  union {
    // Generate a variant member for each type.
    @meta for(int i = 0; i < sizeof...(types_t); ++i)
      @enum_type(tag_t, i) @(i);
  };

  void copy(const variant_t& rhs);
  void move(variant_t&& rhs);

  template<typename type_t>
  type_t& set_copy(const type_t& rhs);

  template<typename type_t>
  type_t& set_move(type_t&& rhs);
};

////////////////////////////////////////////////////////////////////////////////
// variant_t ctors and assignment operators.

// We need to provide an explicit constructor to allow non-trivial types as
// variant members.
template<typename... types_t>
variant_t<types_t...>::variant_t() { }

template<typename... types_t>
variant_t<types_t...>::~variant_t() { 
  reset();
}

// Copy and move ctors.
template<typename... types_t> 
variant_t<types_t...>::variant_t(const variant_t& rhs) { 
  copy(rhs);
}

template<typename... types_t> 
variant_t<types_t...>::variant_t(variant_t&& rhs) { 
  move(std::move(rhs));
}

// Copy and move assignment.
template<typename... types_t> 
variant_t<types_t...>& variant_t<types_t...>::operator=(const variant_t& rhs) {
  reset();
  copy(rhs);
  return *this;
}

template<typename... types_t> 
variant_t<types_t...>& variant_t<types_t...>::operator=(variant_t&& rhs) {
  reset();
  move(std::move(rhs));
  return *this;
}

template<typename... types_t>
void variant_t<types_t...>::copy(const variant_t& rhs) {
  assert(tag_t::none == _tag);

  switch(rhs.tag()) {
    @meta for(int i = 0; i < count; ++i) {
      case @enum_value(tag_t, i):
        // Copy-construct the lhs's variant member.
        new (&@(i)) @enum_type(tag_t, i)(rhs.@(i));
        break;
    }
  }
  _tag = rhs.tag();
}

template<typename... types_t>
void variant_t<types_t...>::move(variant_t&& rhs) {
  assert(tag_t::none == _tag);

  switch(rhs.tag()) {
    @meta for(int i = 0; i < count; ++i) {
      case @enum_value(tag_t, i):
        // Move-construct the lhs's variant member.
        new (&@(i)) @enum_type(tag_t, i)(std::move(rhs.@(i)));
        break;
    }
  }
  _tag = rhs.tag();

  // Move ctor and move assignment from a variant resets the rhs. 
  // This is a destructive move.
  rhs.reset();
}


////////////////////////////////////////////////////////////////////////////////
// Variant member ctors and assignment operators.

// Constructors for setting specific variant members.
template<typename... types_t> template<typename type_t>
variant_t<types_t...>::variant_t(const type_t& item) {
  set_copy(item);
}

template<typename... types_t> template<typename type_t>
variant_t<types_t...>::variant_t(type_t&& item) {
  set_move(std::move(item));
}

template<typename... types_t> template<typename type_t>
type_t& variant_t<types_t...>::operator=(const type_t& rhs) {
  return set_copy(rhs);
}

template<typename... types_t> template<typename type_t>
type_t& variant_t<types_t...>::operator=(type_t&& rhs) {
  return set_move(std::move(rhs));
}

////////////////////////////////////////////////////////////////////////////////

template<typename... types_t>
void variant_t<types_t...>::reset() {
  // Destruct the variant member corresponding to the tag's value.
  switch(_tag) {
    @meta for(int i = 0; i < count; ++i) {
      case @enum_value(tag_t, i):
        // The pseudo-destructor name call becomes a no-op at substitution 
        // for non-class types. 
        @(i).~@enum_type(tag_t, i)();
        break;
    }

    case tag_t::none:
      break;
  }

  // Reset the tag.
  _tag = tag_t::none;
}

////////////////////////////////////////////////////////////////////////////////

template<typename... types_t> template<size_t I>
auto& variant_t<types_t...>::get() {
  static_assert(I < count, "variant ordinal is out of range");
  assert((size_t)_tag == I);
  return @(I);
}

template<typename... types_t> template<size_t I>
const auto& variant_t<types_t...>::get() const {
  static_assert(I < count, "variant ordinal is out of range");
  assert((size_t)_tag == I);
  return @(I);
}

template<typename...types_t> template<typename type_t>
type_t& variant_t<types_t...>::get() {
  type_t* p = safe_get<type_t>();
  assert(p);
  return *p;
}

template<typename...types_t> template<typename type_t>
const type_t& variant_t<types_t...>::get() const {
  type_t* p = safe_get<type_t>();
  assert(p);
  return *p;
}

template<typename... types_t> template<typename type_t>
type_t* variant_t<types_t...>::safe_get() {

  @meta for(int i = 0; i < count; ++i) {
    @meta if(std::is_constructible<type_t&, @enum_type(tag_t, i)&>::value) {
      // If the variant member's type is the same type or an accessible 
      // base type of the requested type, this is a match. In this case, emit
      // the tag comparison. This is much cheaper than making a switch on the
      // outside, because we only test the tag type for possible candidates.
      if(_tag == @enum_value(tag_t, i))
        return std::addressof(@(i));
    }
  }

  // Return nullptr for no match.
  return nullptr;
}

template<typename... types_t> template<typename type_t>
const type_t* variant_t<types_t...>::safe_get() const {

  @meta for(int i = 0; i < count; ++i) {
    @meta if(std::is_constructible<const type_t&, @enum_type(tag_t, i)&>::value) {
      // If the variant member's type is the same type or an accessible 
      // base type of the requested type, this is a match. In this case, emit
      // the tag comparison. This is much cheaper than making a switch on the
      // outside, because we only test the tag type for possible candidates.
      if(_tag == @enum_value(tag_t, i))
        return std::addressof(@(i));
    }
  }

  // Return nullptr for no match.
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

template<typename... types_t> template<typename type_t>
type_t& variant_t<types_t...>::set_copy(const type_t& rhs) {
  reset();

  @meta bool has_match = false;
  @meta for(int i = 0; i < count && !has_match; ++i) {
    @meta if(std::is_same<type_t, @enum_type(tag_t, i)>::value) {
      @meta has_match = true;

      // Run the copy constructor on the variant member and set the tag.
      new (&@(i)) @enum_type(tag_t, i)(rhs);
      _tag = @enum_value(tag_t, i);

      return @(i);
    }
  }
    
  static_assert(has_match, "variant_t has no compatible variant member");
}

template<typename... types_t> template<typename type_t>
type_t& variant_t<types_t...>::set_move(type_t&& rhs) {
  reset();

  @meta bool has_match = false;
  @meta for(int i = 0; i < count && !has_match; ++i) {
    @meta if(std::is_same<type_t, @enum_type(tag_t, i)>::value) {
      @meta has_match = true;

      // Run the move constructor on the variant member and set the tag.
      new (&@(i)) @enum_type(tag_t, i)(std::move(rhs));
      _tag = @enum_value(tag_t, i);

      return @(i);
    }
  }
  static_assert(has_match, "variant_t has no compatible variant member");
}

////////////////////////////////////////////////////////////////////////////////

template<typename... types_t> template<typename func_t>
auto variant_t<types_t...>::visit(func_t func) {
  switch(_tag) {
    case tag_t::none:
      assert(false);

    @meta for(int i = 0; i < count; ++i)
      case @enum_value(tag_t, i):
        return func(get<i>());
  }
}


////////////////////////////////////////////////////////////////////////////////

struct vec3_t {
  float x, y, z;
};

typedef variant_t<int, double, vec3_t, std::vector<short> > my_variant_t;

void switch_visitor(my_variant_t& v) {
  switch(v.tag()) {
    case typename int:
      cirprint("int: %\n", v.get<int>());
      break;

    case typename double:
      cirprint("double: %\n", v.get<double>());
      break;

    case typename vec3_t:
      cirprint("vec3_t: %\n", v.get<vec3_t>());
      break;

    case typename std::vector<short>:
      cirprint("std::vector<short>: %\n", v.get<std::vector<short> >());
      break;

    case typename void:
      printf("<none>\n");
      break;
  }
}

int main() {

  // Copy-construct the variant to an integer.
  my_variant_t var = 100;
  switch_visitor(var);

  my_variant_t var2 = std::move(var);
  switch_visitor(var);
  switch_visitor(var2);

  var = 3.14;
  switch_visitor(var);

  // Use the index-based accessor. The double value has index 1.
  double d = var.get<1>();
  cirprint("var<1> = %\n", d);

  var = 15;
  switch_visitor(var);

  // Set to a vector.
  std::vector<short>& vec = var = std::vector<short>();
  vec.push_back(100);
  vec.push_back(200);
  switch_visitor(var);

  // Demonstrate the visitor pattern.
  var = vec3_t { 5, 6, 7 };
  var.visit([](auto& member) {
    // The variant member is passed by reference. Remove the reference type
    // to pretty print it.
    typedef typename std::remove_reference<decltype(member)>::type type_t;
    cirprint("%: %\n", @type_string(type_t, true), member);
  });

  return 0;
}