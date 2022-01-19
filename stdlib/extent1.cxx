#include <type_traits>
#include <limits>
#include <iostream>

constexpr size_t dynamic_extent = size_t.max;

template<typename Type>
concept SizeType = std::is_convertible_v<Type, size_t>;

template<size_t index, size_t Extent>
struct _storage_t {
  // static storage.
  constexpr _storage_t(size_t extent) noexcept { }
  static constexpr size_t extent = Extent;
};

template<size_t index>
struct _storage_t<index, dynamic_extent> {
  // dynamic storage.
  constexpr _storage_t(size_t extent) noexcept : extent(extent) { }
  size_t extent;
};

template<size_t... Extents>
struct extents {
  // Partial static storage.
  [[no_unique_address]] _storage_t<int..., Extents> ...m;
  
  // Count the rank (number of Extents).
  static constexpr size_t rank() noexcept {
    return sizeof... Extents;
  }

  // Count the dynamic rank (number of Extents equal to dynamic_equal).
  static constexpr size_t rank_dynamic() noexcept {
    return (0 + ... + (dynamic_extent == Extents));
  }

  // Dynamic access to extents.
  constexpr size_t extent(size_t i) const noexcept {
    return i == int... ...? m.extent : 0;
  }

  // Construct from one index per extent.
  template<SizeType... IndexTypes>
  requires(sizeof...(IndexTypes) == rank())
  constexpr extents(IndexTypes... exts) noexcept : m(exts)... { }

  // Map index I to index of dynamic extent J.
  template<size_t I>
  static constexpr size_t find_dynamic_index =
    (0 + ... + (dynamic_extent == Extents...[:I]));

  // Construct from one index per *dynamic extent*.
  template<SizeType... IndexTypes>
  requires(
    sizeof...(IndexTypes) != rank() && 
    sizeof...(IndexTypes) == rank_dynamic()
  )
  constexpr extents(IndexTypes... exts) noexcept : m(
    dynamic_extent == Extents ??
      exts...[find_dynamic_index<int...>] :
      Extents
  )... { }
};

int main() {
  using Extents = extents<3, 4, dynamic_extent, dynamic_extent, 7>;

  // Initialize extents with one value per extent.
  Extents e1(3, 4, 5, 6, 7);

  // Initialize extents with one value per *dynamic extent*. The static extents
  // are inherited from the template arguments.
  Extents e2(5, 6);

  for(int i : Extents::rank())
    std::cout<< i<< ": "<< e1.extent(i)<< " - "<< e2.extent(i)<< "\n";
}