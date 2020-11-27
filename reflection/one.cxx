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
  @meta for(int i = 0; i < @member_count(type_t); ++i) {
    printf("%d: %s - %s\n", i, @member_type_string(type_t, i), 
      @member_name(type_t, i));
  }
}

int main() {
  print_members<record_t>();
}