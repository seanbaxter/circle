#include <fstream>
#include "json.hpp"
#include "parse.inl"
#include "eval.hxx"
#include "regex_lib.hxx"

struct regex_t {
  std::string name, pattern;
};

static std::vector<regex_t> load_regex_from_file(const char* filename) {
  // Open the file.
  std::ifstream f(filename);
  if(!f.is_open())
    throw std::runtime_error("could not open file " + std::string(filename));

  // Pipe into JSON.
  nlohmann::json j;
  f>> j;

  // Iterate over all items.
  std::vector<regex_t> patterns;

  for(auto& item : j.items()) {
    patterns.push_back({
      item.key(), item.value()
    });
  }

  return patterns;
}

// Let the user specify the json file with -D LIB_REGEX_JSON=filename
@meta std::vector<regex_t> patterns = load_regex_from_file(LIB_REGEX_JSON);

namespace pcre {

std::optional<lib_result_t> lib_regex_match(const char* name,
  const char* begin, const char* end) {
  
  // Match these pattern names.
  @meta for(regex_t& regex : patterns) {
    if(!strcmp(name, @string(regex.name))) {
      // Run the match.
      std::optional<lib_result_t> result;

      // Specialize the match_regex function template on the string loaded
      // from the json.
      @meta printf("Added pattern \"%s\": \"%s\"\n", regex.name.c_str(),
        regex.pattern.c_str());
      if(auto x = match_regex<regex.pattern>(begin, end)) {
        // Copy the results from a static result object to the 
        // lib_result-t.
        result = lib_result_t {
          x->begin, x->end, { x->captures.begin(), x->captures.end() }
        };
      }

      return result;
    }
  }

  throw std::runtime_error("unrecognized pattern name " + std::string(name));
}

}

// If LIB_REGEX_EXE is defined, generate a main function that tests the 
// argument input against all loaded patterns.

#ifdef LIB_REGEX_EXE

int main(int argc, char** argv) {
  // Test the input against each regex pattern.
  if(2 != argc) {
    printf("%s expects a string input argument\n", argv[0]);
    return 1;
  }

  const char* begin = argv[1];
  const char* end = begin + strlen(begin);
  
  @meta for(regex_t& regex : patterns) {
    if(auto match = pcre::lib_regex_match(@string(regex.name), begin, end)) {
      printf("input matches \"s\": \"%s\"\n", @string(regex.name), 
        @string(regex.pattern));
      for(int i = 0; i < match->captures.size(); ++i) {
        // Print each of the captures.
        pcre::range_t range = match->captures[i];
        printf("  %2d: %.*s\n", i, range.end - range.begin, range.begin);
      }
    }
  }

  return 0;
}

#endif
