int add_test1(int x, int y) {
  return x + y;
}
int sub_test1(int x, int y) {
  return x - y;
}
int mul_test1(int x, int y) {
  return x * y;
}

#feature on no_signed_overflow_ub

int add_test2(int x, int y) {
  return x + y;
}
int sub_test2(int x, int y) {
  return x - y;
}
int mul_test2(int x, int y) {
  return x * y;
}
