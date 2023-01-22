int main() {
  const int mask  = 0x07;
  const int value = 0x03;

  static_assert(3 != mask & value);

  #pragma feature simpler_precedence

  static_assert(3 == mask & value);
}