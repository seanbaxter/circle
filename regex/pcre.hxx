#pragma once
#include <optional>
#include <memory>
#include <vector>
#include <cstring>

namespace pcre {

struct node_t;
typedef std::unique_ptr<node_t> node_ptr_t;

enum class metachar_func_t {
  none,
  iscntrl, isprint, isspace, isblank, isgraph, ispunct, isalnum, 
  isalpha, isupper, islower, isdigit, isxdigit, isword,
};

struct node_t {
  enum kind_t {
    // terminals
    kind_char,
    kind_range,
    kind_any,
    kind_meta,
    kind_boundary,
    kind_cclass,

    // unary operators
    kind_opt,
    kind_star,
    kind_plus,
    kind_quant,
    kind_capture,

    // binary operators
    kind_seq,
    kind_alt,
  } kind;

  // kind_cclass, kind_meta, kind_boundary
  bool negate = false;

  union {
    char32_t c;                         // kind_char
    struct { char32_t c_min, c_max; };  // kind_range
    metachar_func_t metachar_func;      // kind_meta
    struct { int r_min, r_max; };       // kind_quant
    int capture_index;                  // knid_capture
  };

  node_t(kind_t kind) : kind(kind) { }
  std::vector<node_ptr_t> children;
};

std::pair<node_ptr_t, int> parse_regex(const char* pattern);

void print_ast(const node_t* node, int indent = 0);

// regex [:word:] is alnum or _.
inline int isword(int c) {
  return isalnum(c) || '_' == c;
}

} // namespace pcre



