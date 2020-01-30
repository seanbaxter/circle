#include <vector>
#include <cstdio>

inline std::vector<int> find_primes(int count) {
  std::vector<int> primes;
  primes.reserve(count);

  int cur = 2;
  while(primes.size() < count) {
    // Check if any element in primes divides cur.
    bool divides = (... || (0 == cur % primes[:]));

    // If cur is relatively prime against what has been computed, save it.
    if(!divides)
      primes.push_back(cur);

    // Try the next elemnent.
    ++cur;
  }

  return primes;
}

// Compute the primes at compile time into an std::vector.
@meta std::vector<int> primes_vec = find_primes(47);

// Transfer them into a static array for runtime access.
// Expand 
const int primes[] { primes_vec[:] ... };

// Print the primes at compile time. Any static data member is available
// at compile time as well as runtime, even though it was just created
// dynamically.
@meta for(size_t i = 0; i < std::size(primes); i += 10) {
  @meta printf("%3d ", primes[i : i + 10])...; 
  @meta printf("\n");
}

int main() {
  // Print the primes at runtime.
  for(size_t i = 0; i < std::size(primes); i += 10) { 
    printf("%3d ", primes[i : i + 10])...; 
    printf("\n");
  }
}