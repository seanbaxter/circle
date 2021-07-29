#include <cstdio>
#include <unistd.h>

// A board is uniquely determined by its height, width and cell data.
// The timestep is not part of the board.
template<int height, int width, int... cells>
struct board_t {
  // Convert the cell data to a string constant.
  static const char string[] {
    for row : height =>
      .{ cells...[row * width : (row+1) * width] ? '*' : ' ' ..., '\n' },
    '\0'
  };
};

template<int height, int width, int... cells>
using advance_life = board_t<height, width, 
  // Make a for loop over the neighbor counts.
  for row : height =>
    for col : width =>
      auto count : 
        // For each cell, run a fold expression over its 3x3 support.
        (... + for i : 3 => for j : 3 =>
          (1 != i || 1 != j) &&
          cells...[
            (col + i + width - 1) % width +
            (row + j + height - 1) % height * width
          ]
        ) =>
        // New cell set if count is 2, or count is 3 and cell was already set.
        3 == count || (2 == count && cells...[row * width + col])
>;

const int Height = 10;
const int Width = 24;
using Start = board_t<Height, Width, 
  0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
>;

template<int timestep>
struct life_t {
  // Extract the cells from the previous timestep and advance them.
  using prev = typename life_t<timestep - 1>::board;
  using board = advance_life<
    Height, 
    Width, 
    prev.nontype_args...[2:]...
  >;
};

// Seed timestep 0 with the starting point to avoid infinite
// template recursion.
template<> struct life_t<0> { using board = Start; };

@meta+ for(int timestep : 100) {
  usleep(200000);     // Sleep for 200ms, to target 5fps.
  printf("%3d:\n%s\n", timestep, life_t<timestep>::board::string);
}