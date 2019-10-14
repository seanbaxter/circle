#pragma once
#include "pcre.hxx"

namespace pcre {

struct lib_result_t {
  range_t match;
  std::vector<range_t> captures;
};

// This function is automatically generated into a shared object by
// regex_libmaker.cxx
std::optional<lib_result_t> lib_regex_match(const char* pattern_name,
  const char* begin, const char* end);

} // namespace pcre
