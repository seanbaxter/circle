set -x

time g++-10     benchmark.cxx -std=c++20 -O2 -DBENCHMARK_SIZE=80 -DUSE_STD_TUPLE    -o benchmark-gcc
./benchmark-gcc

# nvcc doesn't have C++20 support.
time nvcc       benchmark.cxx -std=c++17 -O2 -DBENCHMARK_SIZE=80 -DUSE_STD_TUPLE    -o benchmark-nvcc
./benchmark-nvcc

time nvc++      benchmark.cxx -std=c++20 -O2 -DBENCHMARK_SIZE=80 -DUSE_STD_TUPLE    -o benchmark-nvc++
./benchmark-nvcc

time clang++-12 benchmark.cxx -std=c++20 -O2 -DBENCHMARK_SIZE=80 -DUSE_STD_TUPLE    -o benchmark-clang
./benchmark-clang

time circle     benchmark.cxx -std=c++20 -O2 -DBENCHMARK_SIZE=80 -DUSE_STD_TUPLE    -o benchmark-circle-std
./benchmark-circle-std

time circle     benchmark.cxx -std=c++20 -O2 -DBENCHMARK_SIZE=80 -DUSE_EASY_TUPLE   -o benchmark-circle-easy
./benchmark-circle-easy

time circle     benchmark.cxx -std=c++20 -O2 -DBENCHMARK_SIZE=80 -DUSE_HARD_TUPLE   -o benchmark-circle-hard
./benchmark-circle-hard

time circle     benchmark.cxx -std=c++20 -O2 -DBENCHMARK_SIZE=80 -DUSE_CIRCLE_TUPLE -o benchmark-circle-circle
./benchmark-circle-circle
