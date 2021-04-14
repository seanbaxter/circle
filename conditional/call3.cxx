template<int x> int func();

int call(int index) {
  return int...(4) == index ...? func<int...>() : __builtin_unreachable();
}

