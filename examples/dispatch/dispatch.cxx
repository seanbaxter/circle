#include "../include/tuple.hxx"
#include "../include/enums.hxx"

template<
  size_t I,
  template<typename...> class client_temp, 
  typename... types_t, 
  typename... enums_t, 
  typename... args_t
>
auto dispatch_inner(tuple_t<enums_t...> e, args_t&&... args) {
  if constexpr(I == sizeof...(enums_t)) {
    // Instantiate the client class template.
    return client_temp<types_t...>().go(std::forward<args_t>(args)...);

  } else {
    switch(get<I>(e)) {
      static_assert(std::is_enum<enums_t...[I]>::value);

      // Forward to the next level.
      @meta for enum(auto e2 : enums_t...[I])
        case e2:
          return dispatch_inner<
            I + 1,
            client_temp,
            types_t...,                    // Expand the old types
            @enum_type(e2)                 // Add this as a new type
          >(e, std::forward<args_t>(args)...);
    }
  }
}

template<
  template<typename...> class client_temp,
  typename... enums_t, 
  typename... args_t
>
auto dispatch(tuple_t<enums_t...> e, args_t&&... args) {
  return dispatch_inner<0, client_temp>(e, std::forward<args_t>(args)...);
}

// Application code

struct circle   { double val() const { return 10; } };
struct square   { double val() const { return 20; } };
struct octagon  { double val() const { return 30; } };

enum typename class shapes_t {
  circle = circle,
  square = square,
  octagon = octagon,
};

struct red      { double val() const { return 1; } };
struct green    { double val() const { return 2; } };
struct yellow   { double val() const { return 3; } };

enum typename class colors_t {
  red = red,
  green = green,
  yellow = yellow,
};

struct solid    { double val() const { return .1; } };
struct hatch    { double val() const { return .2; } };
struct halftone { double val() const { return .3; } };

enum typename class fills_t {
  solid = solid,
  hatch = hatch,
  halftone = halftone,
};

template<typename shape_obj_t, typename color_obj_t, 
  typename fill_obj_t>
struct shape_computer_t {
  shape_obj_t shape;
  color_obj_t color;
  fill_obj_t fill;

  @meta printf("Instantiating { %s, %s, %s }\n", @type_name(shape_obj_t),
    @type_name(color_obj_t), @type_name(fill_obj_t));

  double go(double x) { 
    return (x * shape.val() + color.val()) * fill.val(); 
  }
};

int main(int argc, char** argv) {
  if(5 != argc) {
    printf("Usage: dispatch shape-name color-name fill-name x\n");
    exit(1);
  }

  // Meta-generated if-strcmp chain finds an enumerator from a string.
  shapes_t shape = enum_from_name_error<shapes_t>(argv[1]);
  colors_t color = enum_from_name_error<colors_t>(argv[2]);
  fills_t fill = enum_from_name_error<fills_t>(argv[3]);
  double x = atof(argv[4]);

  // Use our tuple to hold the runtime enums.
  tuple_t<shapes_t, colors_t, fills_t> key { shape, color, fill };

  // Provide the enum tuple to select the object to instantiate and the
  // numeric argument.
  double y = dispatch<shape_computer_t>(key, x);

  printf("The dispatch result is %f\n", y);
  return 0;
}