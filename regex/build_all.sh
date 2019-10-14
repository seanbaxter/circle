set -x

# Build the parser into its own library.
circle -o libparse.so parse.cxx

# Build the regex example using libparse.so.
circle -o regex -M libparse.so regex.cxx

# Build a JSON-driven library.
circle regex_libmaker.cxx -o libregex.so -D LIB_REGEX_JSON=\"sample.json\"

# Build a JSON-driver executable.
circle regex_libmaker.cxx -o regex_test -D LIB_REGEX_EXE -D LIB_REGEX_JSON=\"sample.json\"
