#include <tuple>
#include <iostream>

int main() {
  // Structured bindings accept arrays, tuples and generic class objects.
  // Bind a parameter pack to each element in the tuple.
  auto [...pack] = std::make_tuple(1.1, 20, "Hello tuple", 'c');

  // Print like a schlub.
  std::cout<< "pack printed with a loop:\n";
  @meta for(int i = 0; i < sizeof...(pack); ++i)
    std::cout<< "  "<< pack...[i]<< "\n";

  // Print like a champ.
  std::cout<< "pack printed with an expansion:\n";
  std::cout<< "  "<< pack<< "\n" ...;

  // Print the types of the pack.
  std::cout<< "pack types are:\n";
  std::cout<< "  "<< @type_name(decltype(pack), true)<< "\n" ...;

  // Create a new pack with reversed values.
  constexpr size_t count = sizeof...(pack);
  auto bar = std::make_tuple(pack...[count - 1 - __integer_pack(count)]...);
  auto [...reversed] = bar;

  std::cout<< "reversed pack is:\n";
  std::cout<< "  "<< reversed<< "\n" ...;

  return 0;
}