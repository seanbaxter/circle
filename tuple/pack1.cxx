#include <iostream>
#include <utility>

template<typename... Ts>
struct tuple {
  // Declare a parameter pack of data members with ...<name> declarator-id.
  [[no_unique_address]] Ts ...m;

  // Declare default, copy and move constructors.
  tuple() : m()... { }
  tuple(const tuple&) = default;
  tuple(tuple&&) = default;

  // Converting constructor. Note the ... after the m pack subobject init.
  template<typename... Ts2>
  requires((... && std::is_constructible_v<Ts, Ts2&&>))
  tuple(Ts2&&... x) : m(x)... { }

  // Specialize a single element. Subobject-initialize that one element,
  // and default construct the rest of them.
  template<size_t I, typename T>
  tuple(std::in_place_index_t<I>, T&& x) :
    m...[I](x), m()... { }

  // Use pack subscript ...[I] to access pack data members.
  template<int I>
  Ts...[I]& get() {
    return m...[I];
  }

  template<typename T>
  T& get() {
    // The requested type must appear exactly once in the tuple.
    static_assert(1 == (... + (T == Ts)));
    constexpr size_t I = T == Ts ...?? int... : -1;
    return m...[I];
  }
};

struct empty1_t { };
struct empty2_t { };
struct empty3_t { };

// Members of the same type do not alias under no_unique_address rules.
static_assert(3 == sizeof(tuple<empty1_t, empty1_t, empty1_t>));

// Members of different types do alias under no_unique_address rules.
static_assert(1 == sizeof(tuple<empty1_t, empty2_t, empty3_t>));

int main() {
  // Use the converting constructor to create a tuple.
  tuple<int, double, const char*> x(10, 3.14, "Hello tuple");
  std::cout<< x.get<0>()<< " "<< x.get<1>()<< " "<< x.get<2>()<< "\n";

  // Initialize only member 1 with 100.0.
  tuple<int, double, float> y(std::in_place_index<1>, 100);
  std::cout<< y.get<0>()<< " "<< y.get<1>()<< " "<< y.get<2>()<< "\n";

  // Print the int element of x and the double element of y.
  std::cout<< x.get<int>()<< " "<< y.get<double>()<< "\n";
}