set -x
circle -std=c++20 tuple1.cxx && ./tuple1
circle -std=c++20 variant1.cxx && ./variant1
circle -std=c++20 extent1.cxx && ./extent1
