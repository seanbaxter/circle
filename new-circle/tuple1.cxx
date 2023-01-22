#pragma feature tuple
#include <tuple>
#include <cstdio>

// An empty type tuple is spelled (-).
using T0 = (-);
static_assert(T0 == std::tuple<>);

// A 1-tuple is spelled (type-id,)
using T1 = (int*,);
static_assert(T1 == std::tuple<int*>);

// Higher-level tuples are comma separated.
using T2 = (int*, double[5]);
static_assert(T2 == std::tuple<int*, double[5]>);

int main() {

  // An empty tuple expression is (,).
  std::tuple<> tup0 = (,);

  // A 1-tuple expression is (expr, ).
  std::tuple<int> tup1 = (5,);

  // Higher-level tuple expressions are comma separated.
  std::tuple<int, double> tup2 = (5, 3.14);

  // Revert to the discard operator by casting the first element to void.
  int x = ((void)printf("Hello not a tuple\n"), 5);
  printf("x = %d\n", x);
}

