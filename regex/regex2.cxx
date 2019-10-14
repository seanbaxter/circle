#include "eval.hxx"
#include "parse.inl"

int main() {
  const char* text = "gray";
  if(auto match = pcre::match_regex<"gr[ae]y", true>(text)) {
    printf("Input matches pattern\n");

  } else {
    printf("Input doesn't match pattern\n");
  }
  
  return 0;
}