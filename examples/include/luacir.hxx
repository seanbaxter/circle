#pragma once
#include <lua5.3/lua.hpp>
#include <vector>
#include <string>
#include <optional>
#include <cstdio>
#include "util.hxx"
#include "enums.hxx"

class lua_engine_t {
public:
  lua_engine_t();
  ~lua_engine_t();

  void file(const char* filename);
  void script(const char* text);

  template<typename result_t, typename... args_t>
  result_t call(const char* fname, const args_t&... args);

  operator lua_State*() { return state; }

private:

  template<typename arg_t>
  void push(const arg_t& arg);

  template<typename arg_t>
  void push_array(const arg_t* data, size_t count);

  template<typename arg_t>
  void push_object(const arg_t& obj);

  template<typename value_t>
  std::optional<value_t> get_value(const char* name);

  lua_State* state;
};

inline lua_engine_t::lua_engine_t() {
  state = luaL_newstate();
  luaL_openlibs(state);
}

inline lua_engine_t::~lua_engine_t() {
  lua_close(state);
}

inline void lua_engine_t::file(const char* filename) {
  luaL_dofile(state, filename);
}

inline void lua_engine_t::script(const char* text) {
  luaL_dostring(state, text);
}

template<typename arg_t>
void lua_engine_t::push(const arg_t& arg) {
  if constexpr(std::is_enum<arg_t>::value) {
    lua_pushstring(state, name_from_enum(arg));

  } else if constexpr(@sfinae(arg.c_str())) {
    lua_pushstring(state, arg.c_str());

  } else if constexpr(@sfinae(std::string(arg))) {
    lua_pushstring(state, arg);

  } else if constexpr(std::is_integral<arg_t>::value) {
    lua_pushinteger(state, arg);

  } else if constexpr(std::is_floating_point<arg_t>::value) {
    lua_pushnumber(state, arg);

  } else if constexpr(std::is_array<arg_t>::value) {
    push_array(arg, std::extent<arg_t>::value);

  } else if constexpr(@is_class_template(arg_t, std::vector)) {
    push_array(arg.data(), arg.size());

  } else {
    static_assert(std::is_class<arg_t>::value, "expected class type");
    push_object(arg);
  }
}

template<typename arg_t>
void lua_engine_t::push_array(const arg_t* data, size_t count) {
  lua_createtable(state, count, 0);
  for(size_t i = 0; count; ++i) {
    // Push the value.
    push(data[i]);

    // Insert the item at t[i + 1].
    lua_seti(state, -2, i + 1);
  }
}

template<typename arg_t>
void lua_engine_t::push_object(const arg_t& object) {
  lua_createtable(state, 0, @member_count(arg_t));
  @meta for(size_t i = 0; i < @member_count(arg_t); ++i) {
    // Push the data member.
    push(@member_ref(object, i));

    // Insert the item at t[member-name].
    lua_setfield(state, -2, @member_name(arg_t, i));
  }
}

template<typename result_t, typename... args_t>
result_t lua_engine_t::call(const char* fname, const args_t&... args) {
  // Push the function to the stack.
  lua_getglobal(state, fname);

  const size_t num_args = sizeof...(args_t);
  @meta for(int i = 0; i < num_args; ++i)
    // Push each argument to the stack.
    push(args...[i]);

  // Call the function.
  lua_call(state, num_args, LUA_MULTRET);

  if constexpr(!std::is_void<result_t>::value) {
    result_t ret { };
    if(auto value = get_value<result_t>(fname))
      ret = std::move(*value);
    return ret;
  }
}

template<typename value_t>
std::optional<value_t> lua_engine_t::get_value(const char* name) {
  if(lua_isnil(state, -1)) {
    printf("  **No field in result object named '%s'\n", name);
    lua_pop(state, 1);
    return { };
  }

  std::optional<value_t> value { };
  if constexpr(std::is_enum<value_t>::value) {
    if(lua_isstring(state, -1)) {
      const char* s = lua_tostring(state, -1);
      if(auto e = enum_from_name<value_t>(s))
        value = *e;
      else
        printf("  **Unrecognized enum '%s' in field '%s'\n", s, name);

    } else
      printf("  **Expected enum in field '%s'\n", name);
    
    lua_pop(state, 1);

  } else if constexpr(@is_class_template(value_t, std::vector)) {
    typedef typename value_t::value_type inner_t;

    if(lua_istable(state, 1)) {
      int len = lua_rawlen(state, -1);
      std::vector<inner_t> vec;
      vec.reserve(len);

      for(int i = 1; i <= len; ++i) {
        lua_pushinteger(state, i);
        if(LUA_TNIL != lua_gettable(state, -2)) {
          if(auto x = get_value<inner_t>(name))
            vec.push_back(std::move(*x));
        }
      }
      value = std::move(vec);

    } else
      printf("  **Expected array in field '%s'\n", name);

    lua_pop(state, 1);

  } else if constexpr(std::is_integral<value_t>::value) {
    if(lua_isinteger(state, -1))
      value = lua_tointeger(state, -1);
    else
      printf("  **Expected integer in field '%s'\n", name);

    lua_pop(state, 1);

  } else if constexpr(std::is_floating_point<value_t>::value) {
    if(lua_isnumber(state, -1))
      value = lua_tonumber(state, -1);
    else
      printf("  **Expected floating-point in field '%s'\n", name);

    lua_pop(state, 1);

  } else if constexpr(std::is_same<std::string, value_t>::value) {
    if(lua_isstring(state, -1))
      value = lua_tostring(state, -1);
    else
      printf("  **Expected string in field '%s'\n", name);

    lua_pop(state, 1);

  } else {
    static_assert(std::is_class<value_t>::value, "expected class type");

    if(lua_istable(state, 1)) {
      // Loop over each data member.
      value_t obj { };
      @meta for(int i = 0; i < @member_count(value_t); ++i) {{
        const char* name = @member_name(value_t, i);
        lua_pushstring(state, name);
        lua_gettable(state, -2);
        if(auto x = get_value<@member_type(value_t, i)>(name))
          @member_ref(obj, i) = std::move(*x);
      }}

      value = std::move(obj);
      lua_pop(state, 1);

    } else 
      printf("  **Expected table in field '%s'\n", name);
  }

  return value;
}
