#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  
  @match(atoi(argv[1])) {
    1                       => printf("It's 1\n");
    < 0                     => printf("It's negative\n");
    > 100                   => printf("More than 100\n");
    2 ... 5                 => printf("A number 2 <= x < 5\n");
    5 ... 10 && !7          => printf("5 <= x < 10 but not 7\n");
    7 || 10 || 13           => printf("7 or 10 or 13\n");
    10 ... 15 || 20 ... 25  => printf("In disjoint ranges\n");
    ! 30 ... 90             => printf("Not between 30 and 90\n");
    _x if(1 & _x)           => printf("%d is an odd number\n", _x);
    < 50 _x if(0 == _x % 4) => printf("Less than 50 but multiple of 4\n");
    _                       => printf("Everything else\n");
  };
  
  return 0;

} 