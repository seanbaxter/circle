#include <cstdarg>
#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <cassert>

inline std::string format(const char* pattern, ...) {
  va_list args;
  va_start(args, pattern);

  va_list args_copy;
  va_copy(args_copy, args);

  int len = std::vsnprintf(nullptr, 0, pattern, args);
  std::string result(len, ' ');
  std::vsnprintf((char*)result.data(), len + 1, pattern, args_copy);

  va_end(args_copy);
  va_end(args);

  return result;
}

// Call std::getline and trim the trailing newline character or sequence.
std::string good_getline(std::istream& is) {
  std::string s;
  std::getline(is, s);

  for(char& c : s) {
    switch(c) {
      case '\r':
      case '\n':
        s.resize(&c - s.c_str());
        return s;
        break;

      default:
        break;
    }
  }
  return s;
}

////////////////////////////////////////////////////////////////////////////////

struct type_info_t {
  struct field_t {
    std::string field_type;
    std::string field_name;
  };
  std::vector<field_t> fields;
};

// Define a structure from a type_info_t known at compile time. This macro
// may inject the struct declaration into any namespace from any scope.
@macro void define_type(const char* name, const type_info_t& type_info) {
  struct @(name) {
    @meta for(auto& field : type_info.fields)
      @type_id(field.field_type) @(field.field_name);
  };
}

@macro void define_csv_type(const char* name, const char* filename) {
  @macro define_type(name, read_csv_schema(filename));
}

std::string make_ident(std::string s) {
  // Turn a string into an identifier.
  if(!s.size())
    { };

  if(isdigit(s[0]))
    s = '_' + s;

  for(char& c : s) {
    if(c != '_' && !isalnum(c))
      c = '_';
  }

  return s;
}

type_info_t read_csv_schema(std::istream& is, bool read_types) {
  type_info_t type_info { };

  // Read the field names.
  std::string s = good_getline(is);

  const char* text = s.c_str();
  while(*text) {
    // Consume the leading ',' delimiter.
    if(type_info.fields.size())
      ++text;

    // Find the next comma or end-of-string.
    const char* end = text;
    while(*end && ',' != end[0]) ++end;

    // Push the identifier field_name.
    type_info.fields.push_back({ { }, make_ident(std::string(text, end)) });

    // Advance to the comma or end-of-string.
    text = end;
  }

  if(read_types) {
    // Infer the field types from the first data line.
    s = good_getline(is);
    text = s.c_str();

    size_t num_fields = type_info.fields.size();
    for(size_t i = 0; i < num_fields; ++i) {
      auto& field = type_info.fields[i];

      if(i && !*text) {
        throw std::runtime_error(format(
          "field %s not found at line %d in CSV file", 
          field.field_name.c_str(), 0
        ));
      }

      // Consume the leading ',' delimiter.
      if(i)
        ++text;

      // Find the next comma or end-of-string.
      const char* end = text;
      while(*end && ',' != end[0]) ++end;

      // Test if the field is a double.
      int chars_read;
      double x;
      int result = sscanf(text, "%lf%n", &x, &chars_read);
      field.field_type = (result && chars_read == end - text) ? 
        "double" : "std::string";

      // Advance to the comma or end-of-string.
      text = end;
    }
  }

  return type_info;
}

type_info_t read_csv_schema(const char* filename) {
  std::ifstream file(filename);
  if(!file.is_open()) {
    throw std::runtime_error(format(
      "cannot open CSV file %s", filename
    ));
  }

  return read_csv_schema(file, true);
}

////////////////////////////////////////////////////////////////////////////////

template<typename type_t>
void verify_schema(const type_info_t& type_info) {
  @meta size_t num_fields = @member_count(type_t);

  // Test the number of fields.
  if(type_info.fields.size() != num_fields) {
    throw std::runtime_error(format(
      "%s has %d fields while schema has %d fields", 
      @type_string(type_t), num_fields, type_info.fields.size()
    ));
  }

  @meta for(size_t i = 0; i < num_fields; ++i) {
    // Test the name of each field.
    const auto& field = type_info.fields[i];

    if(field.field_name != @member_name(type_t, i)) {
      throw std::runtime_error(format(
        "field %d is called %s in %s and %s in schema",
        i, @member_name(type_t, i), @type_string(type_t), field.field_name.c_str()
      ));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

template<typename type_t>
type_t read_csv_line(const char* text, int line) {
  type_t obj { };

  @meta for(size_t i = 0; i < @member_count(type_t); ++i) {
    if(i && !*text) {
      throw std::runtime_error(format(
        "field %s not found at line %d in CSV file", 
        @member_name(type_t, i), line
      ));
    }

    // Consume the leading ',' delimiter.
    if constexpr(i) {
      assert(',' == text[0]);
      ++text;
    }

    // Find the next comma or end-of-string.
    const char* end = text;
    while(*end && ',' != end[0]) ++end;

    // Support strings and doubles.
    if constexpr(std::is_same<double, @member_type(type_t, i)>::value) {
      // Parse a double. Confirm that we've read all characters in the field.
      double x = 0;
      if(text < end) {
        int chars_read;
        int result = sscanf(text, "%lf%n", &x, &chars_read);
        if(!result || chars_read != end - text) {
          throw std::runtime_error(format(
            "field %s at line %d \'%s\' is not a number",
            @member_name(type_t, i), line, std::string(text, end - text).c_str()
          ));
        }
      }

      @member_value(obj, i) = x;

    } else {
      @member_value(obj, i) = std::string(text, end);
    }

    // Advance to the comma or end-of-string.
    text = end;
  }

  return obj;
}

template<typename type_t>
std::vector<type_t> read_csv_file(const char* filename) {
  std::ifstream file(filename);
  if(!file.is_open()) {
    throw std::runtime_error(format(
      "cannot open CSV file %s", filename
    ));
  }

  // Load the schema and verify against the static type.
  type_info_t type_info = read_csv_schema(file, false);
  verify_schema<type_t>(type_info);

  // Load each CSV line.
  std::vector<type_t> vec;
  int line = 1;
  
  while(file.good()) {
    ++line;
    std::string s = good_getline(file);
    if(!s.size())
      break;

    vec.push_back(read_csv_line<type_t>(s.c_str(), line));
  }

  return vec;
}

////////////////////////////////////////////////////////////////////////////////
// Everything above this point is library code. The user expands the
// define_csv_type macro to define the data field's object type. The user
// then calls normal function templates, which deserialize the data file
// into a vector of the specified type by using reflection.  

int main() {
  // Use type providers to define the object type from the CSV schema.
  // This happens at compile time.
  @macro define_csv_type("obj_type_t", "schema.csv");

  // Print the field types and names.
  @meta std::cout<< @type_string(@member_types(obj_type_t))<< " "<< 
    @member_names(obj_type_t)<< "\n" ... ;
 
  // Load the values at runtime. The schema is inferred and checked against
  // the static type info.
  auto data = read_csv_file<obj_type_t>("earthquakes1970-2014.csv");

  printf("Read %d records\n", data.size());

  // Print out 10 random coordinates. Access the members by name. These names
  // are inferred from the schema at compile time.
  for(int i = 0; i < 10; ++i) {
    int index = rand() % data.size();

    // The first line is CSV schema. The first index comes at line 2.
    int line = index + 2;
    double lat = data[index].Latitude;
    double lon = data[index].Longitude;
    printf("line = %4d Latitude = %+10f  Longitude = %+10f\n", line, lat, lon);
  }

  // Print out all the fields from a random record. Use introspection for
  // this.
  int index = rand() % data.size();
  std::cout<< index + 2<< ":\n";
  std::cout<< "  "<< @member_names(obj_type_t)<< ": "<< 
    @member_values(data[index])<< "\n" ...;

  return 0;
}
