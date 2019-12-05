#include "eval.hxx"

void regex_match(const char* text) {
  auto match = pcre::match_regex<"g(}[:xdigit:]{3})[:xdigit:]*(r*)r[ea]((y|z)*?)", true>(text);

  if(match) {
    printf("Matched characters [%d, %d]\n", match->begin - text, 
      match->end - text);
    for(pcre::range_t capture : match->captures) {
      printf("capture: %.*s\n", capture.end - capture.begin, capture.begin);
    }
  }  
}

int main() {
  regex_match("gfeedfacedeadbeefrrrazyzyz");
  return 0;
}