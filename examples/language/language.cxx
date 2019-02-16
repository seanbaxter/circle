#include "../include/enums.hxx"
#include <json.hpp>
#include <fstream>

@meta std::ifstream json_file("language.json");

using nlohmann::json;
@meta json j;
@meta json_file>> j;

// Your core set of languages.
enum class language_t {
  english,
  japanese,
  korean,
  finnish,
  dutch,
  greek,
};

template<typename enum_t>
@meta void report_error(@meta const char* name, enum_t language) {

  // Verify the item is in the file.
  static_assert(j.count(name), 
    "Error code '" + std::string(name) + "' is not in language.json");
  @meta json& item = j[name];

  // Switch over each language in the enumerator.
  switch(language) {
    @meta for(int i = 0; i < @enum_count(enum_t); ++i) {
      // Emit a case-statement for this language.
      @meta enum_t e = @enum_value(enum_t, i);
      case e: {
        // Turn the enumerator name into a string.
        @meta std::string language_name = @enum_name(enum_t, i);

        // Check that the entry is available for the requested language.
        static_assert(item.count(language_name),
          "There is no '" + language_name + "' entry for '" + name + "'");

        // Get the string.
        @meta std::string s = item[language_name].get<std::string>();

        // Print it at runtime. The std::string exists only at compile time,
        // so turn it into a string literal with @string.
        printf("[ERROR]: %s\n", @string(s));
        break;
      }
    }

    default:
      break;
  }
}

int main(int argc, char** argv) {
  if(2 != argc) {
    printf("Usage: language [lang] where lang = \n"
      "\tenglish\n"
      "\tjapanese\n"
      "\tkorean\n"
      "\tfinnish\n"
      "\tgreek\n"
      "\tdutch\n");
    exit(1);
  }

  // These two errors are supported by all languages.
  if(auto language = enum_from_name<language_t>(argv[1])) {
    report_error("green eggs", *language);
    report_error("feline headgear", *language);
  }

  // This error is only supported by parched languages.
  enum parched_language_t {
    english,
    finnish,
    dutch
  };
  if(auto parched = enum_from_name<parched_language_t>(argv[1])) {
    report_error("obligatory seinfeld reference", *parched); 
  }

  return 0;
}