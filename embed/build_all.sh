set -x
circle gen_file.cxx && ./gen_file
time circle embed1.cxx
time circle embed2.cxx
time circle embed3.cxx
time circle embed4.cxx
time circle embed5.cxx