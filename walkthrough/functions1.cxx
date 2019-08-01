#include <cstdio>
#include <cmath>

struct func_t {
	const char* name;
	const char* expr;
};

@meta func_t funcs[] {
	{ "F1", "(x + y) / x" },
	{ "F2", "2 * x * sin(y)" }
};

@meta for(func_t f : funcs) {
	double @(f.name)(double x, double y) {
		return @expression(f.expr);
	}
}

typedef double(*fp_t)(double, double);

fp_t get_func_by_name(const char* name) {
	@meta for(func_t f : funcs) {
		if(!strcmp(@string(f.name), name))
			return ::@(f.name);
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