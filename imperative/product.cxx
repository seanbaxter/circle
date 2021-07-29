template<typename... Ts>
struct list;

template<typename... Ts>
using product = list<
  for ...indices : { for typename T : Ts => sizeof...(T.type_args) } =>
    list<for i, typename T : Ts => T.type_args...[indices...[i]]>
>;

using L1 = list<char8_t, char16_t, char32_t>;
using L2 = list<float, double>;
using L3 = list<int, long>;
using L4 = list<bool, void>;

using L = product<L1, L2, L3, L4>;

static_assert(L == list<
  list<char8_t,  float,  int,  bool>,
  list<char8_t,  float,  int,  void>,    // the RHS is most rapidly varying.
  list<char8_t,  float,  long, bool>, 
  list<char8_t,  float,  long, void>, 
  list<char8_t,  double, int,  bool>, 
  list<char8_t,  double, int,  void>, 
  list<char8_t,  double, long, bool>, 
  list<char8_t,  double, long, void>, 
  list<char16_t, float,  int,  bool>, 
  list<char16_t, float,  int,  void>, 
  list<char16_t, float,  long, bool>,
  list<char16_t, float,  long, void>, 
  list<char16_t, double, int,  bool>, 
  list<char16_t, double, int,  void>, 
  list<char16_t, double, long, bool>, 
  list<char16_t, double, long, void>, 
  list<char32_t, float,  int,  bool>, 
  list<char32_t, float,  int,  void>, 
  list<char32_t, float,  long, bool>, 
  list<char32_t, float,  long, void>, 
  list<char32_t, double, int,  bool>, 
  list<char32_t, double, int,  void>, 
  list<char32_t, double, long, bool>, 
  list<char32_t, double, long, void>
>);