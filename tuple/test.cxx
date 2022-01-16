#include <iostream>
#include <string>
#include "tuple.hxx"

int main() {
  using namespace circle;
  using namespace std::string_literals;

  auto a = make_tuple(100, "Hello tuple", 'x');
  auto b = make_tuple(21.1f, nullptr, 19i16);
  auto c = make_tuple(true, 3.14l);

  auto cat = tuple_cat(std::move(a), std::move(b), c);

  // Print the index, the type of each element, and its value.
  std::cout<< "Visit with Circle packs:\n";
  std::cout<< 
    int...<< ": "<< 
    decltype(cat).tuple_elements.string << " is '"<<
    cat...[:]<< "'\n" ...;

  // Use apply() to visit each element.
  std::cout<< "\nVisit with apply:\n";
  apply([]<typename... Ts>(const Ts&... x) {
    std::cout<< int...<< ": "<< Ts.string<< " is '"<< x<< "'\n" ...;
  }, cat);

  {
    tuple<int, double, void*> tup1;           // trivial
    tuple<int, double, void*> tup2; { }       // zeroing    
    tuple<int, std::string, double> tup3 { }; // default
  }

  {
    // Construct from exact elements.
    tuple<int, std::string, double> tup1(
      500, "Hello world"s, 3.14
    );

    // Convert from elements.
    tuple<int, std::string, double> tup2(
      100i16, "Hello tuple", 6.28f
    );

    // Swap.
    swap(tup1, tup2);

    // Assign.
    tup1 = tup2;

    // Move
    tup2 = std::move(tup1);
  }

  {
    tuple<int, const char*, float> tup1(
      500, "Hello world", 3.14f
    );

    // Converting construct from another tuple.
    tuple<long, std::string, double> tup2 = tup1;
    tuple<long, std::string, double> tup3 = std::move(tup1);

    // Converting copy assign.
    tup2 = tup1;

    // Converting move assign.
    tup3 = std::move(tup1);
  }

  {
    auto pair = std::make_pair(500, "Hello pair");

    // Converting construct from a pair.
    tuple<long, std::string> tup1 = pair;
    tuple<long, std::string> tup2 = std::move(pair);
  }

  {
    // Construct with deduction guides.
    tuple tup1 = 100;
    tuple<const char*> tup2 = "Hello World";
    tuple tup3(100, "Hello world", &tup1);
  }
}