#include <cstdio>

int main() {
  struct node_t {
    int x, y;
    node_t* p;
  };

  node_t a { 1, 2, nullptr };
  node_t b { 3, 4, &a };

  @match(b) {
    [.p: ? * [_x, _y]] => printf("p->x = %d, p->y = %d\n", _x, _y);
    [_x, _y, _] => printf("x = %d, y = %d, p = null\n", _x, _y);
  };

  return 0;
}