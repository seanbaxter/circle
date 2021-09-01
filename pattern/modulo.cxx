#include <type_traits>

template<typename T>
concept Modulo = requires(T a, T b) { a % b; };

// Use multi-token type names in expressions.
static_assert(unsigned int is Modulo);
static_assert(unsigned int[5] is not Modulo);

// Initialize objects with multi-token type names.
int x = unsigned int(5);
int y = signed long{5};

// Array declarators [] are parsed as part of the type, but &, & and *
// are not. These may be operators!

// Note the void && tokens are not parsed as the type void&&.
template<typename T>
concept small_type = T is not void && sizeof(T) <= 4;