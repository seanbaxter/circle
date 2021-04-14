set -x
circle call1.cxx && ./call1
circle call2.cxx && ./call2
circle call3.cxx -S -emit-llvm && cat call3.ll
circle enum.cxx && ./enum
circle call_first.cxx && ./call_first
circle visit.cxx && ./visit