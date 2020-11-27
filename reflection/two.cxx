#include <cstdio>
#include <string>

struct record_t {
  std::string first;
  std::string last;
  char32_t middle_initial;
  int year_of_birth;  
};

template<typename type_t>
void print_members() {
  printf("%s:\n", @type_string(type_t));
  printf("%d: %s - %s\n", int..., @member_type_strings(type_t), 
    @member_names(type_t))...;
}

int main() {
  print_members<record_t>();
}