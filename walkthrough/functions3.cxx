#include "json.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <dirent.h>

// Keep an array of the names of functions we injected.
@meta std::vector<std::string> func_names;

// Inject a function given a name and return-statement expression.
@macro void inject_f(const char* name, const char* expr) {
	@meta std::cout<< "Injecting function '"<< name<< "'\n";

	double @(name)(double x, double y) {
		return @expression(expr);
	}

	// Note that we added this function.
	@meta func_names.push_back(name);
}

// Open a JSON from a filename and loop over its items. Inject each item
// with inject_f.
@macro void inject_from_json(const char* filename) {
	// Open the file and read it as JSON.
	@meta std::ifstream f(filename);
	@meta nlohmann::json j;
	@meta f>> j;

	// Loop over all items in the JSON.
	@meta for(auto& item : j.items()) {
		@meta std::string name = item.key();
		@meta std::string value = item.value();

		// Inject each function.
		@macro inject_f(name.c_str(), value.c_str());
	}
}

inline std::string get_extension(const std::string& filename) {  
  return filename.substr(filename.find_last_of(".") + 1);
}

inline bool match_extension(const char* filename, const char* ext) {
  return ext == get_extension(filename);
}

@macro void inject_from_dir(const char* dirname) {
	// Get a cursor into the indicated directory.
	@meta DIR* dir = opendir(dirname);

	// Loop over all files.
	@meta while(dirent* ent = readdir(dir)) {
		// Match .json files.
		@meta if(match_extension(ent->d_name, "json")) {

			// Inject all the functions named in this JSON file.
			@macro inject_from_json(ent->d_name);
		}
	}

	// Close the resource.
	@meta closedir(dir);
}

// Inject a file's worth of functions.
@macro inject_from_dir(".");

// Map a function name to the function lvalue.
typedef double(*fp_t)(double, double);
fp_t get_func_by_name(const char* name) {
	@meta for(const std::string& s : func_names) {
		if(!strcmp(@string(s), name))
			return ::@(s);
	}

	return nullptr;
}

int main(int argc, char** argv) {
	if(4 != argc) {
		fprintf(stderr, "usage is %s [func-name] [x] [y]\n", argv[0]);
		return 1;
	}

	fp_t fp = get_func_by_name(argv[1]);
	if(!fp) {
		fprintf(stderr, "%s is not a recognized function\n", argv[1]);
		return 1;
	}
	
	double x = atof(argv[2]);
	double y = atof(argv[3]);

	double result = fp(x, y);
	printf("result is %f\n", result);
	return 0;
}