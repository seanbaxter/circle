#include <limits>

constexpr size_t dynamic_extent = size_t.max;

template<size_t Index, size_t Extent>
struct _storage_t {
  constexpr _storage_t(size_t value = Extent) noexcept { }
  static constexpr size_t extent = Extent;
};

template<size_t Index>
struct _storage_t<Index, dynamic_extent> {
  constexpr _storage_t(size_t value = 0) noexcept : extent(value) { }
  size_t extent;
};

template<size_t... Extents>
struct extents {
  [[no_unique_address]] _storage_t<int..., Extents> ...m;

  
};
