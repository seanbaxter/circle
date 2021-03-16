set -x

circle one.cxx && ./one
circle two.cxx && ./two
circle enum_attrib.cxx && ./enum_attrib
circle enum_attrib2.cxx && ./enum_attrib2
circle access.cxx && ./access
circle enum_to_string.cxx && ./enum_to_string
circle reflect.cxx && ./reflect
circle -c reverse.cxx
circle -shader -emit-spirv -c -E comp shuffle.cxx 
circle attrib.cxx && ./attrib
circle -shader -emit-spirv -c -E vert attrib2.cxx
circle type_name.cxx && ./type_name
circle typed_enum1.cxx && ./typed_enum1
circle typed_enum2.cxx && ./typed_enum2
circle typed_enum3.cxx && ./typed_enum3
circle typed_enum4.cxx && ./typed_enum4
circle for_typename.cxx && ./for_typename
circle json1.cxx && ./json1
circle json2.cxx && ./json2
circle commit.cxx && ./commit