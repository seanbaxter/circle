#pragma feature adl
#include <tuple>

int main() {
  auto tup = std::make_tuple(1, 2.2, "Three");

  // Can call with qualified name lookup.
  auto x1 = std::get<0>(tup);

  // OK. the adl keyword enables the ADL candidate.
  auto x2 = adl get<0>(tup);

  // Error: [adl]: adl candidate called without adl token before unqualified name
  auto x3 = get<0>(tup);
}