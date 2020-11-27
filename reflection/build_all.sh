set -x

circle one.cxx && ./one
circle two.cxx && ./two
circle access.cxx && ./access
circle enum_to_string.cxx && ./enum_to_string
circle reflect.cxx && ./reflect
circle typed_enum1.cxx && ./typed_enum1
circle typed_enum2.cxx && ./typed_enum2
circle typed_enum3.cxx && ./typed_enum3
circle typed_enum4.cxx && ./typed_enum4
circle typed_enum5.cxx && ./typed_enum5
circle json.cxx && ./json