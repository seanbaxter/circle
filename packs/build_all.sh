set -x
circle enum.cxx && ./enum
circle enum2.cxx && ./enum2
circle integer_pack.cxx && ./integer_pack
circle locus.cxx && ./locus
circle non_type.cxx && ./non_type
circle object.cxx && ./object
circle object2.cxx && ./object2
circle object3.cxx && ./object3
circle pack_type.cxx && ./pack_type