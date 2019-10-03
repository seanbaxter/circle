#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>

std::vector<double> read_file(const char* name) {

  std::ifstream file(name);
  if(!file.is_open())
    throw std::runtime_error("could not open file " + std::string(name));

  std::vector<double> vec;

  double x;
  while(file>> x)
    vec.push_back(x);

  return vec;
}

double my_function(double x) {
  std::vector<double> coef = read_file("series.txt");

  double x2 = 1;
  double y = 0;
  for(double c : coef) {
    y += x2 * c;
    x2 *= x;
  }

  return y;
}

int main(int argc, char** argv) {
  if(2 != argc) {
    fputs("expected 'x' argument to function\n", stderr);
    return 1;
  }

  double x = atof(argv[1]);
  double y = my_function(x);
  printf("f(%f) = %f\n", x, y);

  return 0;
}