#pragma feature interface
#include <type_traits>
#include <string>

interface IFace { };

template<typename T> requires(T~is_arithmetic)
impl T : IFace { };

// impl is defined for all arithmetic types.
static_assert(impl<uint8_t, IFace>);
static_assert(impl<int16_t, IFace>);
static_assert(impl<long long, IFace>);
static_assert(impl<long double, IFace>);

// impl is undefined for all other types.
static_assert(!impl<void, IFace>);
static_assert(!impl<const char*, IFace>);
static_assert(!impl<std::string, IFace>);
static_assert(!impl<int[10], IFace>);