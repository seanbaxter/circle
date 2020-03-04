#include <vector>
#include <string>
#include <iostream>

template<typename vec_t>
void print_vec(const vec_t& vec) {
  std::cout<< "[ ";
  std::cout<< vec[:-2]<< ", " ...;
  if(vec.size())
    std::cout<< vec.back()<< " ";
  std::cout<< "]\n";
}

// Use a fold expression to confirm it reads the same forward and backward.
bool is_palindrome(int i) {
  std::string s = std::to_string(i);
  return (... && (s[:] == s[::-1]));
}

int main() {
  // Select the first 50 numbers greater than 10000 where they are read like 
  // palindromes in base 10.
  auto vec = [for i : @range(10000:)... if is_palindrome(i) => i ...] | 50;
  print_vec(vec);
}
