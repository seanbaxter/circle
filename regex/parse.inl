#pragma once

// If building this file into a library, #define PCRE_LINKAGE as nothing.
// That will set external linkage.

#ifndef PCRE_LINKAGE
#define PCRE_LINKAGE inline
#endif

#include "pcre.hxx"
#include <cassert>
#include <cstdarg>

namespace pcre {

PCRE_LINKAGE std::string format(const char* pattern, ...) {
  va_list args;
  va_start(args, pattern);

  va_list args_copy;
  va_copy(args_copy, args);

  int len = std::vsnprintf(nullptr, 0, pattern, args);
  std::string result(len, ' ');
  std::vsnprintf((char*)result.data(), len + 1, pattern, args_copy);

  va_end(args_copy);
  va_end(args);

  return result;
}

PCRE_LINKAGE void print_ast(const node_t* node, int indent) {
  for(int i = 0; i < indent; ++i)
    printf("  ");

  switch(node->kind) {
    case node_t::kind_char:
      printf("char = %c\n", node->c);
      break;

    case node_t::kind_range:
      printf("range = %c-%c\n", node->c_min, node->c_max);
      break;

    case node_t::kind_any:
      printf("any (.)\n");
      break;

    case node_t::kind_meta:
      printf("[metachar]\n");
      break;

    case node_t::kind_boundary:
      printf("word-boundary\n");
      break;

    case node_t::kind_cclass:
      printf("[character-class]:\n");
      break;

    case node_t::kind_opt:
      printf("opt:\n");
      break;

    case node_t::kind_star:
      printf("star:\n");
      break;

    case node_t::kind_plus:
      printf("plus:\n");
      break;

    case node_t::kind_quant:
      printf("quant: {%d, %d}:\n", node->r_min, node->r_max);
      break;

    case node_t::kind_capture:
      printf("capture (%d):\n", node->capture_index);
      break;

    case node_t::kind_seq:
      printf("seq:\n");
      break;

    case node_t::kind_alt:
      printf("alt:\n");
      break;
  }

  for(const node_ptr_t& child : node->children)
    print_ast(child.get(), indent + 1);
}


struct metachar_kind_name_t {
  metachar_func_t func;
  const char* name;
};

PCRE_LINKAGE const metachar_kind_name_t metachar_kind_names[] {
  { metachar_func_t::iscntrl,     "[:cntrl:]"   },
  { metachar_func_t::isprint,     "[:print:]"   },
  { metachar_func_t::isspace,     "[:space:]"   },
  { metachar_func_t::isblank,     "[:blank:]"   },
  { metachar_func_t::isgraph,     "[:graph:]"   },
  { metachar_func_t::ispunct,     "[:punct:]"   },
  { metachar_func_t::isalnum,     "[:alnum:]"   },
  { metachar_func_t::isalpha,     "[:alpha:]"   },
  { metachar_func_t::isupper,     "[:upper:]"   },
  { metachar_func_t::islower,     "[:lower:]"   },
  { metachar_func_t::isdigit,     "[:digit:]"   },
  { metachar_func_t::isxdigit,    "[:xdigit:]"  },
  { metachar_func_t::isword,      "[:word:]"    },
};

struct result_base_t {
  const char* next;
  node_ptr_t attr;

  result_base_t() : next(nullptr) { }
};

struct result_parse_t : protected result_base_t {
  result_parse_t() : success(false) { }
  result_parse_t(const char* next, node_ptr_t attr) : success(true) { 
    this->next = next;
    this->attr = std::move(attr);
  }

  explicit operator bool() const { 
    return success;
  }

  // Access the attr and next members through this operator. This checks
  // success to confirm we're using the result correctly.
  result_base_t* operator->() {
    assert(success);
    return this;
  }

private:
  bool success;
};

PCRE_LINKAGE result_parse_t make_result(const char* next, node_ptr_t node) {
  return result_parse_t(next, std::move(node));
}

struct grammar_t {
  // character class.
  result_parse_t parse_cclass_item(const char* pattern);
  result_parse_t parse_cclass_char(const char* pattern, bool expect = false);
  result_parse_t parse_escape(const char* pattern);
  result_parse_t parse_metachar(const char* pattern);

  // Productions in highest-to-lowest precedence.
  result_parse_t parse_term(const char* pattern, bool expect);
  result_parse_t parse_duplication(const char* pattern, bool expect = false);
  result_parse_t parse_concat(const char* pattern, bool expect);
  result_parse_t parse_alternation(const char* pattern, bool topmost = false);

  void throw_error(const char* pattern, const char* msg);

  const char* begin = nullptr;
  int capture_count = 0;
};

PCRE_LINKAGE result_parse_t grammar_t::parse_cclass_item(const char* pattern) {
  // Support escapes here?
  result_parse_t a = parse_cclass_char(pattern);
  if(a) {
    pattern = a->next;
    if('-' == *pattern) {
      result_parse_t b = parse_cclass_char(pattern + 1, true);
      a->attr->kind = node_t::kind_range;
      a->attr->c_max = b->attr->c;
      a->next = b->next;

      if(a->attr->c_min > a->attr->c_max)
        throw_error(pattern, "invalid range: max must be greater than min");
    }
  }
  return a;
}

PCRE_LINKAGE result_parse_t grammar_t::parse_cclass_char(const char* pattern, 
  bool expect) {

  result_parse_t result;
  switch(char c = *pattern) {
    case '\0':
    case ']':
      break;

    case '\\': 
      result = parse_escape(pattern);
      break;

    default: {
      auto char_ = std::make_unique<node_t>(node_t::kind_char);
      char_->c = c;
      result = make_result(pattern + 1, std::move(char_));
      break;
    }
  }
  return result;
}

PCRE_LINKAGE result_parse_t grammar_t::parse_escape(const char* pattern) {
  result_parse_t result;
  if('\\' == *pattern++) {
    char32_t c2 = 0;
    switch(char c = *pattern++) {
      // C escapes.
      case 'a': c2 = '\a'; break;
      case 'b': c2 = '\b'; break;
      case 'f': c2 = '\f'; break;
      case 'n': c2 = '\n'; break;
      case 'r': c2 = '\r'; break;
      case 't': c2 = '\t'; break;
      case 'v': c2 = '\v'; break;

      // regex escapes.
      case '.':
      case '*':
      case '+':
      case '?':
      case '|':
      case ')':
      case '(':
      case '[':
      case ']':
      case '{':
      case '}':
      case '/':
      case '\\': c2 = c; break;

      default: 
        throw_error(pattern - 2, "escape not recognized");
        break;
    }

    if(c2) {
      auto char_ = std::make_unique<node_t>(node_t::kind_char);
      char_->c = c2;
      result = make_result(pattern, std::move(char_));
    }
  }
  return result;
}

PCRE_LINKAGE result_parse_t grammar_t::parse_metachar(const char* pattern) {
  result_parse_t result;
  switch(char c = *pattern++) {
    case '\\': {
      char c = *pattern++;
      metachar_func_t func { };
      switch(c) {       
        case 'd':
        case 'D':
          func = metachar_func_t::isdigit;
          break;

        case 'w':
        case 'W':
          func = metachar_func_t::isword;
          break;

        case 's':
        case 'S':
          func = metachar_func_t::isspace;
          break;

        case 'b':
        case 'B': {
          // Word-boundary is handled specially.
          auto boundary = std::make_unique<node_t>(node_t::kind_boundary);
          boundary->negate = isupper(c);
          return make_result(pattern, std::move(boundary));
        }
      }

      if(metachar_func_t::none != func) {
        auto node = std::make_unique<node_t>(node_t::kind_meta);
        node->negate = isupper(c);
        node->metachar_func = func;
        result = make_result(pattern, std::move(node));
      }
      break;
    }

    case '[': {
      --pattern;
      int len1 = strlen(pattern);
      for(auto metachar : metachar_kind_names) {
        int len2 = strlen(metachar.name);
        if(len2 <= len1 && !memcmp(pattern, metachar.name, len2)) {
          auto node = std::make_unique<node_t>(node_t::kind_meta);
          node->negate = false;
          node->metachar_func = metachar.func;
          result = make_result(pattern + len2, std::move(node));
          break;
        }
      }
      break;
    }

    default:
      break;
  }
  return result;
}

PCRE_LINKAGE result_parse_t grammar_t::parse_term(const char* pattern, 
  bool expect) {

  result_parse_t result;
  switch(*pattern) {
    case '\0': 
    case '|':
    case ')':
    case ']':
    case '}':
    case '+':
    case '*':
    case '?':
      throw_error(pattern, "unexpected token in term");
      break;

    case '(': {
      // Go to lowest precedence inside ( ).
      ++pattern;

      if('?' == pattern[0] && ':' == pattern[1]) {
        // This is a non-capture grouping.
        pattern += 2;
        result = parse_alternation(pattern);        

      } else {
        // This is a capture grouping.
        result = parse_alternation(pattern);

        // Put the result into a capture node.
        auto capture = std::make_unique<node_t>(node_t::kind_capture);
        capture->capture_index = capture_count++;
        capture->children.push_back(std::move(result->attr));
        result->attr = std::move(capture);
      }

      pattern = result->next;
      if(')' != *pattern)
        throw_error(pattern, "expected ')' to close group structure");
      result->next = ++pattern;
      break;
    }

    case '[': {
      // Parse character class.
      result = parse_metachar(pattern);
      if(!result) { 
        ++pattern;
        auto cclass = std::make_unique<node_t>(node_t::kind_cclass);
        cclass->negate = false;
        if('^' == *pattern)
          cclass->negate = true, ++pattern;

        if(']' == *pattern) {
          auto char_ = std::make_unique<node_t>(node_t::kind_char);
          char_->c = ']', ++pattern;
          cclass->children.push_back(std::move(char_));
        }

        while(auto item = parse_cclass_item(pattern)) {
          pattern = item->next;
          cclass->children.push_back(std::move(item->attr));
        }

        if(!cclass->children.size())
          throw_error(pattern, "expected character in character class");

        if(']' != *pattern)
          throw_error(pattern, "expected ']' to close character class");
 
        ++pattern;
        result = make_result(pattern, std::move(cclass));
      }

      break;
    }

    case '\\': {
      result = parse_metachar(pattern);
      if(!result) result = parse_escape(pattern);
      break;
    }

    case '.': {
      auto any = std::make_unique<node_t>(node_t::kind_any);
      result = make_result(pattern + 1, std::move(any));
      break;
    }

    default: {
      auto char_ = std::make_unique<node_t>(node_t::kind_char);
      char_->c = *pattern;
      result = make_result(pattern + 1, std::move(char_));
      break;
    }
  }

  return result;   
}


PCRE_LINKAGE result_parse_t grammar_t::parse_duplication(const char* pattern, 
  bool expect) {

  // Parse unary terms and ranges.
  auto term = parse_term(pattern, expect);
  while(term) {
    pattern = term->next;
    switch(char c = *pattern) {
      case '*': {
        auto star = std::make_unique<node_t>(node_t::kind_star);
        star->children.push_back(std::move(term->attr));
        term->attr = std::move(star);
        term->next = ++pattern;
        break;
      }

      case '+': {
        auto plus = std::make_unique<node_t>(node_t::kind_plus);
        plus->children.push_back(std::move(term->attr));
        term->attr = std::move(plus);
        term->next = ++pattern;
        break;
      }

      case '?': {
        auto opt = std::make_unique<node_t>(node_t::kind_opt);
        opt->children.push_back(std::move(term->attr));
        term->attr = std::move(opt);
        term->next = ++pattern;
        break;
      }

      case '{': {
        // Parse quantifier.
        ++pattern;
        int min_count = -1;
        int max_count = -1;
        if(',' != *pattern) {
          // Read the min count.
          int n;
          if(1 != sscanf(pattern, "%d%n", &min_count, &n))
            throw_error(pattern, "expected min-quantifier");
          pattern += n;
        }

        if(',' == *pattern) {
          ++pattern;

          // Read the max count.
          int n;
          if(1 != sscanf(pattern, "%d%n", &max_count, &n))
            throw_error(pattern, "expected max-quantifier");
          pattern += n;

        } else {
          max_count = min_count;
        }

        if('}' != *pattern)
          throw_error(pattern, "expected '}' to end quantifier");
        ++pattern;

        if(-1 == min_count && -1 == max_count)
          throw_error(pattern, "expected min or max quantifier");

        if(min_count > max_count)
          throw_error(pattern, 
            "min quantifier must not be greater than max quantifier");

        auto quant = std::make_unique<node_t>(node_t::kind_quant);
        quant->r_min = min_count;
        quant->r_max = max_count;
        quant->children.push_back(std::move(term->attr));
        term->attr = std::move(quant);
        term->next = pattern;
        break;
      }

      default:
        return std::move(term);
    }
  }

  return term;
}

PCRE_LINKAGE result_parse_t grammar_t::parse_concat(const char* pattern, 
  bool expect) {

  // Match the lhs.
  result_parse_t a = parse_duplication(pattern, expect);
  while(a) {
    // Advance past the lhs.
    pattern = a->next;

    if(auto b = parse_duplication(pattern, false)) {
      pattern = b->next;

      if(node_t::kind_seq != a->attr->kind) {
        auto seq = std::make_unique<node_t>(node_t::kind_seq);
        seq->children.push_back(std::move(a->attr));        
        a->attr = std::move(seq);
      }

      a->attr->children.push_back(std::move(b->attr));
      a->next = pattern;

    } else
      break;
  }

  return std::move(a);
}

PCRE_LINKAGE result_parse_t grammar_t::parse_alternation(const char* pattern,
  bool topmost) {

  result_parse_t a = parse_concat(pattern, true);
  while(a) {
    // Advance past the lhs.
    pattern = a->next;
    if('|' == *pattern) {
      result_parse_t b = parse_concat(pattern + 1, true);

      if(node_t::kind_alt != a->attr->kind) {
        auto alt = std::make_unique<node_t>(node_t::kind_alt);
        alt->children.push_back(std::move(a->attr));
        a->attr = std::move(alt);
      }

      a->attr->children.push_back(std::move(b->attr));
      a->next = b->next;

    } else
      break;
  }

  if(topmost) {
    switch(char c = *pattern) {
      case ')':
        throw_error(pattern, "')' encountered outside of group");
        break;

      case ']':
        throw_error(pattern, "']' encountered outside of character class");
        break;

      default:
        break;
    }
  }

  return a;
}

PCRE_LINKAGE void grammar_t::throw_error(const char* pattern, const char* msg) {
  ptrdiff_t col = pattern - begin;
  std::string s = format("%s\n%*s^\n%s\n", begin, col, "", msg);
  throw std::runtime_error(s);
}

PCRE_LINKAGE std::pair<node_ptr_t, int> parse_regex(const char* pattern) {
  grammar_t g { pattern };
  result_parse_t result = g.parse_alternation(pattern, true);
  if(*result->next)
    g.throw_error(result->next, "cannot complete pattern parse");

  return std::make_pair(std::move(result->attr), g.capture_count); 
}

} // namespace pcre
