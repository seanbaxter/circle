// Make stat available.
#define __USE_EXTERN_INLINES
#include <sys/stat.h>

// Support directory ops.
#include <dirent.h>

#include <string>
#include <vector>
#include <map>
#include <cstdio>

inline std::vector<std::string> get_dir_filenames(std::string dirname) {
  std::vector<std::string> filenames;

  DIR* dir = opendir(dirname.c_str());

  // Loop over all entities in the directory.
  while(dirent* e = readdir(dir)) {
    // Concatenate the dirname and filename.
    std::string filename = dirname + "/" + e->d_name;

    // Match regular files.
    struct stat statbuf;
    if(0 == stat(filename.c_str(), &statbuf) && S_ISREG(statbuf.st_mode))
      filenames.push_back(std::move(filename));
  }

  closedir(dir);
  return filenames;
}

// Relate filenames to file contents.
@meta auto filenames = get_dir_filenames("resources");
@meta puts(filenames[:].c_str())...;

// Populate a runtime std::map relating filenames to contents.
// The @embed operation is performed at compile time, but the map is 
// constructed at runtime.
// The map, being a global, can also be used at compile time by the 
// interpreter. Since the compiler knows the initializer, it JIT-constructs the
// map when ODR-used by the interpreter. The files have already been loaded 
// by this point, and indexed through a deduplicator.

// We can also declare a meta std::map. This has the advantage of not 
// generating any wasteful code, and it may still be used at runtime, but 
// the indices must be compile-time strings.

struct range_t {
  const char* begin;
  size_t size;
};

template<size_t count>
range_t make_range(const char(&array)[count]) {
  return range_t { array, count };
}

std::map<std::string, range_t> file_map = {
  std::make_pair<std::string, range_t>(
    @string(@pack_nontype(filenames)),
    make_range(@embed(char, @pack_nontype(filenames)))
  )...
};

// Print the files loaded at compile time.
@meta+ for(auto& item : file_map) {
  printf("%s (%zu bytes): %.*s\n", item.first.c_str(), 
    item.second.size, item.second.size, item.second.begin);
}


int main() {
  for(auto& item : file_map) {
    printf("%s (%zu bytes): %.*s\n", item.first.c_str(), 
      item.second.size, item.second.size, item.second.begin);
  }
  return 0;
}
