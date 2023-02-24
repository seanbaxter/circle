#feature on template_brackets
#include <iostream>

// auto function parameters generate "invented" template parameters.
void f1(auto... x) {
  std::cout<< decltype(x)~string<< " " ...;
  std::cout<< "\n";
}

// Be careful because invented parameters are tacked to the end.
// That may be surprising since the order of template params doesn't
// match the order of function params.
template<typename Z, typename W>
void f2(auto x, auto y, Z z, W w) {
  std::cout<< decltype(x)~string<< " "
           << decltype(y)~string<< " "
           << Z~string<< " "
           << W~string<< "\n";
}

void dispatch(auto f) {
  // Use !< > to pass a template-argument-list on an object expression. 
  // This allows explicit template arguments for [over.call.object].
  // For lambdas, it is shorthand for f.template operator()<char16_t, short>.
  f!<char16_t, short>(1, 2, 3, 4);
}

int main() {
  dispatch([]f1);
  dispatch([]f1!wchar_t);
  dispatch([]f2);
  dispatch([]f2!wchar_t);  // f2 has invented parameters, which are at the end.
}
