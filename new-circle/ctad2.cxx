template<typename X, typename Y, typename Z>
struct foo_t {
  X x;
  Y y;
  Z z;
};

int main() { 
  #pragma feature no_aggregate_deduction
  foo_t obj { 1, 2, 3 };
}