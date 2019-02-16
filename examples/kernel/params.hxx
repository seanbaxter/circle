#pragma once
#include <vector>
#include <string>

enum class kernel_flag_t {
  ldg,
  ftz,
  fast_math
};

struct kernel_key_t {
  int sm;
  std::string type;
};

struct params_t {
  int bytes_per_lane;
  int lanes_per_thread;
  std::vector<kernel_flag_t> flags;
};