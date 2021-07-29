#include <iostream>
#include <tuple>

template<typename T>
using ReplicateArgs = T.template<
  .{ T.type_args }(int... + 1) ...
>;

using T1 = std::tuple<int, char*, double, short>;
using T2 = ReplicateArgs<T1>;

@meta std::cout<< T2.string<< "\n";