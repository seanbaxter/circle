#pragma once
#include "pcre.hxx"
#include <array>

namespace pcre {

typedef const char* it_t;
typedef std::optional<it_t> result_t;

template<typename... types_t>
struct list_t { };

// Match the end of the pattern. This must be the end of input.
template<typename state_t>
inline result_t match(state_t& state, it_t it, list_t<>) {
  if(it != state.end) return { };
  return { it };
}

// Return a successful parse.
struct eps_t { };

template<typename state_t>
inline result_t match(state_t& state, it_t it, list_t<eps_t>) {
  return { it };
}

// Match a single character in the sequence.
template<char c>
struct ch_t { };

template<typename state_t, char c, typename... ts>
result_t match(state_t& state, it_t it, list_t<ch_t<c>, ts...>) {
  if(it == state.end || c != *it) return { };
  return match(state, ++it, list_t<ts...>());
}

// Match a character range.
template<char a, char b>
struct crange_t { };

template<typename state_t, char a, char b, typename... ts>
result_t match(state_t& state, it_t it, list_t<crange_t<a, b>, ts...>) {
  if(it == state.end || a > *it || b < *it) return { };
  return match(state, ++it, list_t<ts...>());
}

// Match any character.
struct any_t { };
template<typename state_t, typename... ts>
result_t match(state_t& state, it_t it, list_t<any_t, ts...>) {
  if(it == state.end) return { };
  return match(state, ++it, list_t<ts...>());
}


// Match a character class. These are functions like isalnum, isalpha, islower.
typedef int(*metachar_ptr_t)(int);

template<bool negate, metachar_ptr_t fp>
struct metachar_t { };

template<typename state_t, bool negate, metachar_ptr_t fp, typename... ts>
result_t match(state_t& state, it_t it, list_t<metachar_t<negate, fp>, ts...>) {
  if(it == state.end || negate == fp(*it)) return { };
  return match(state, ++it, list_t<ts...>());
}

// Match a word boundary.
template<bool negate>
struct boundary_t { };

template<typename state_t, bool negate, typename... ts>
result_t match(state_t& state, it_t it, list_t<boundary_t<negate>, ts...>) {
  // Check the characters it[-1] and it[0].
  // If they have different word status, we're on a word boundary.
  // The beginning/end of the string is a non-word.
  bool left = it > state.begin && isword(it[-1]);
  bool right = it < state.end && isword(it[0]);
  if(negate == (left == right))
    return match(state, it, list_t<ts...>());
  else
    return { };
}

// Match a character class.
template<bool negate, typename... alts>
struct cclass_t { };

template<typename state_t, bool negate, typename... alts, typename... ts>
result_t match(state_t& state, it_t it, 
  list_t<cclass_t<negate, alts...>, ts...>) {
  // Take the first match.
  @meta for(int i = 0; i < sizeof...(alts); ++i) {
    if(auto alt = match(state, it, list_t<alts...[i], eps_t>()))
      return match(state, *alt, list_t<ts...>());
  }

  return { };
}

// Match a sequence.
template<typename... args>
struct seq_t { };

template<typename state_t, typename... args, typename... ts>
result_t match(state_t& state, it_t it, list_t<seq_t<args...>, ts...>) {
  // Unwrap the sequence and insert it to the front of the list.
  return match(state, it, list_t<args..., ts...>());
}

// Match an alternative.
template<typename... args>
struct alt_t { };

template<typename state_t, typename... args, typename... ts>
result_t match(state_t& state, it_t it, list_t<alt_t<args...>, ts...>) {
  // Try each alternative in order.
  @meta for(int i = 0; i < sizeof...(args); ++i) {
    if(auto out = match(state, it, list_t<args...[i], ts...>()))
      return out;
  }
  return { };
}

// Match an optional.
template<typename a>
struct opt_t { };

template<typename state_t, typename a, typename... ts>
result_t match(state_t& state, it_t it, list_t<opt_t<a>, ts...>) {
  // First match with the optional (because it's longer)
  if(auto out = match(state, it, list_t<a, ts...>()))
    return out;
  else 
    // Then match without the optional.
    return match(state, it, list_t<ts...>());
}

// Match 0 or more items.
template<typename a>
struct star_t { };

template<typename state_t, typename a, typename... ts>
result_t match(state_t& state, it_t it, list_t<star_t<a>, ts...>) {
  while(true) {
    if(auto zero = match(state, it, list_t<ts...>())) {
      // We matched the rest of the input.
      return zero;

    } else if(auto one = match(state, it, list_t<a, eps_t>())) {
      // We matched the subject. Advance the iterator and go through the 
      // loop again.
      it = *one;

    } else {
      return { };
    }
  }
}

// Match 1 or more items.
template<typename a>
struct plus_t { };

template<typename state_t, typename a, typename... ts>
result_t match(state_t& state, it_t it, list_t<plus_t<a>, ts...>) {
  // Rewrite a+ as aa*.
  return match(state, it, list_t<a, star_t<a>, ts...>());
}

// Match r_min through r_max items.
template<int r_min, int r_max, typename a>
struct quant_t { };

template<typename state_t, int r_min, int r_max, typename a, typename... ts>
result_t match(state_t& state, it_t it, 
  list_t<quant_t<r_min, r_max, a>, ts...>) {

  int i;
  for(i = 0; i < r_min; ++i) {
    // Advance the minimum number of matches.
    if(auto x = match(state, it, list_t<a, eps_t>()))
      it = *x;
    else
      return { };
  }

  while(true) {
    // Test the rest of the input and return.
    if(auto x = match(state, it, list_t<ts...>()))
      return x;
    else if(i == r_max)
      break;
    else if(auto x = match(state, it, list_t<a, eps_t>()))
      it = *x;
    else
      break;
  }
  
  return { };
}

template<int index, typename a>
struct capture_t { };

template<int index>
struct capture_end_t { };

template<typename state_t, int index, typename a, typename... ts>
result_t match(state_t& state, it_t it, list_t<capture_t<index, a>, ts...>) {
  // Run the regex with capture_end_t inserted after the capture's subject.
  // If successful, mark the start of the capture.
  if(auto x = match(state, it, list_t<a, capture_end_t<index>, ts...>())) {
    state.captures[index].begin = it;
    return x;
  }

  return { };
}

template<typename state_t, int index, typename... ts>
result_t match(state_t& state, it_t it, list_t<capture_end_t<index>, ts...>) {
  // Parse the rest of the list. If it's successful, mark the end of the
  // capture.
  if(auto x = match(state, it, list_t<ts...>())) {
    state.captures[index].end = it;
    return x;
  }

  return { };
}

// Given a compile-time node_t*, return the corresponding mtype.
@macro auto lower_ast(node_t* p) {

  // Evaluate the types of the child nodes.
  @meta std::vector<@mtype> types { 
    lower_ast(@pack_nontype(p->children).get())... 
  };

  @meta+ if(node_t::kind_char == p->kind) {
    @emit return @dynamic_type(ch_t<p->c>);

  } else if(node_t::kind_range == p->kind) {
    @emit return @dynamic_type(crange_t<p->c_min, p->c_max>);

  } else if(node_t::kind_any == p->kind) {
    @emit return @dynamic_type(any_t);

  } else if(node_t::kind_meta == p->kind) {
    // Retrieve the name of the enum, eg isalpha or isxdigit.
    // Wrap in @() to turn the string into an identifier. 
    // Ordinary name lookup finds the function name in this namespace or the
    // global namespace and chooses the overload to match the lhs function 
    // pointer.
    metachar_ptr_t fp = @(@enum_name(p->metachar_func));
    @emit return @dynamic_type(metachar_t<p->negate, fp, @pack_type(types)...>);

  } else if(node_t::kind_boundary == p->kind) {
    @emit return @dynamic_type(boundary_t<p->negate, @pack_type(types)...>);

  } else if(node_t::kind_cclass == p->kind) {
    @emit return @dynamic_type(cclass_t<p->negate, @pack_type(types)...>);

  } else if(node_t::kind_opt == p->kind) {
    @emit return @dynamic_type(opt_t<@pack_type(types)...>);

  } else if(node_t::kind_star == p->kind) {
    @emit return @dynamic_type(star_t<@pack_type(types)...>);

  } else if(node_t::kind_plus == p->kind) {
    @emit return @dynamic_type(plus_t<@pack_type(types)...>);

  } else if(node_t::kind_quant == p->kind) {
    @emit return @dynamic_type(quant_t<p->r_min, p->r_max, @pack_type(types)...>);

  } else if(node_t::kind_capture == p->kind) {
    @emit return @dynamic_type(capture_t<p->capture_index, @pack_type(types)...>);

  } else if(node_t::kind_seq == p->kind) {
    @emit return @dynamic_type(seq_t<@pack_type(types)...>);

  } else if(node_t::kind_alt == p->kind) {
    @emit return @dynamic_type(alt_t<@pack_type(types)...>);
  }
}

struct range_t { 
  it_t begin, end;
};

template<int capture_count>
struct parse_state_t {
  enum { num_capture_groups = capture_count };
  it_t begin, end;
  std::array<range_t, capture_count> captures;
};

template<const char pattern[], bool debug = false>
auto match_regex(const char* begin, const char* end) {
  // First parse the regex pattern at compile time.
  @meta auto parsed = parse_regex(pattern);

  @meta+ if(debug)
    print_ast(parsed.first.get());

  // Construct an expression template type from the AST.
  using type = @static_type(lower_ast(parsed.first.get()));

  // Create and initialize the state. This is required to hold the captures.
  typedef parse_state_t<parsed.second> state_t;
  state_t state { begin, end };

  std::optional<state_t> result;
  if(auto x = match(state, begin, list_t<type>())) {
    // Mark the end point of the parse.
    state.end = *x;

    // Return the captures as part of the result.
    result = state;
  }

  return result;
}

template<const char pattern[], bool debug = false>
auto match_regex(const char* text) {
  return match_regex<pattern, debug>(text, text + strlen(text));
}

} // namespace pcre
