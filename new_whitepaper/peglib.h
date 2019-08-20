//
//  peglib.h
//
//  Copyright (c) 2015-18 Yuji Hirose. All rights reserved.
//  MIT License
//

#ifndef CPPPEGLIB_PEGLIB_H
#define CPPPEGLIB_PEGLIB_H

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// guard for older versions of VC++
#ifdef _MSC_VER
// VS2013 has no constexpr
#if (_MSC_VER == 1800)
#define PEGLIB_NO_CONSTEXPR_SUPPORT
#elif (_MSC_VER >= 1800)
// good to go
#else //(_MSC_VER < 1800)
#error "Requires C+11 support"
#endif
#endif

// define if the compiler doesn't support unicode characters reliably in the
// source code
//#define PEGLIB_NO_UNICODE_CHARS

namespace peg {

/*-----------------------------------------------------------------------------
 *  any
 *---------------------------------------------------------------------------*/

class any
{
public:
    any() : content_(nullptr) {}

    any(const any& rhs) : content_(rhs.clone()) {}

    any(any&& rhs) : content_(rhs.content_) {
        rhs.content_ = nullptr;
    }

    template <typename T>
    any(const T& value) : content_(new holder<T>(value)) {}

    any& operator=(const any& rhs) {
        if (this != &rhs) {
            if (content_) {
                delete content_;
            }
            content_ = rhs.clone();
        }
        return *this;
    }

    any& operator=(any&& rhs) {
        if (this != &rhs) {
            if (content_) {
                delete content_;
            }
            content_ = rhs.content_;
            rhs.content_ = nullptr;
        }
        return *this;
    }

    ~any() {
        delete content_;
    }

    bool is_undefined() const {
        return content_ == nullptr;
    }

    template <
        typename T,
        typename std::enable_if<!std::is_same<T, any>::value, std::nullptr_t>::type = nullptr
    >
    T& get() {
        if (!content_) {
            throw std::bad_cast();
        }
        auto p = dynamic_cast<holder<T>*>(content_);
        assert(p);
        if (!p) {
            throw std::bad_cast();
        }
        return p->value_;
    }

    template <
        typename T,
        typename std::enable_if<std::is_same<T, any>::value, std::nullptr_t>::type = nullptr
    >
    T& get() {
        return *this;
    }

    template <
        typename T,
        typename std::enable_if<!std::is_same<T, any>::value, std::nullptr_t>::type = nullptr
    >
    const T& get() const {
        assert(content_);
        auto p = dynamic_cast<holder<T>*>(content_);
        assert(p);
        if (!p) {
            throw std::bad_cast();
        }
        return p->value_;
    }

    template <
        typename T,
        typename std::enable_if<std::is_same<T, any>::value, std::nullptr_t>::type = nullptr
    >
    const any& get() const {
        return *this;
    }

private:
    struct placeholder {
        virtual ~placeholder() {}
        virtual placeholder* clone() const = 0;
    };

    template <typename T>
    struct holder : placeholder {
        holder(const T& value) : value_(value) {}
        placeholder* clone() const override {
            return new holder(value_);
        }
        T value_;
    };

    placeholder* clone() const {
        return content_ ? content_->clone() : nullptr;
    }

    placeholder* content_;
};

/*-----------------------------------------------------------------------------
 *  scope_exit
 *---------------------------------------------------------------------------*/

// This is based on "http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4189".

template <typename EF>
struct scope_exit
{
    explicit scope_exit(EF&& f)
        : exit_function(std::move(f))
        , execute_on_destruction{true} {}

    scope_exit(scope_exit&& rhs)
        : exit_function(std::move(rhs.exit_function))
        , execute_on_destruction{rhs.execute_on_destruction} {
            rhs.release();
    }

    ~scope_exit() {
        if (execute_on_destruction) {
            this->exit_function();
        }
    }

    void release() {
        this->execute_on_destruction = false;
    }

private:
    scope_exit(const scope_exit&) = delete;
    void operator=(const scope_exit&) = delete;
    scope_exit& operator=(scope_exit&&) = delete;

    EF   exit_function;
    bool execute_on_destruction;
};

template <typename EF>
auto make_scope_exit(EF&& exit_function) -> scope_exit<EF> {
    return scope_exit<typename std::remove_reference<EF>::type>(std::forward<EF>(exit_function));
}

/*-----------------------------------------------------------------------------
 *  UTF8 functions
 *---------------------------------------------------------------------------*/

inline size_t codepoint_length(const char *s8, size_t l) {
  if (l) {
    auto b = static_cast<uint8_t>(s8[0]);
    if ((b & 0x80) == 0) {
      return 1;
    } else if ((b & 0xE0) == 0xC0) {
      return 2;
    } else if ((b & 0xF0) == 0xE0) {
      return 3;
    } else if ((b & 0xF8) == 0xF0) {
      return 4;
    }
  }
  return 0;
}

inline size_t encode_codepoint(char32_t cp, char *buff) {
  if (cp < 0x0080) {
    buff[0] = static_cast<char>(cp & 0x7F);
    return 1;
  } else if (cp < 0x0800) {
    buff[0] = static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
    buff[1] = static_cast<char>(0x80 | (cp & 0x3F));
    return 2;
  } else if (cp < 0xD800) {
    buff[0] = static_cast<char>(0xE0 | ((cp >> 12) & 0xF));
    buff[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    buff[2] = static_cast<char>(0x80 | (cp & 0x3F));
    return 3;
  } else if (cp < 0xE000) {
    // D800 - DFFF is invalid...
    return 0;
  } else if (cp < 0x10000) {
    buff[0] = static_cast<char>(0xE0 | ((cp >> 12) & 0xF));
    buff[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    buff[2] = static_cast<char>(0x80 | (cp & 0x3F));
    return 3;
  } else if (cp < 0x110000) {
    buff[0] = static_cast<char>(0xF0 | ((cp >> 18) & 0x7));
    buff[1] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    buff[2] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    buff[3] = static_cast<char>(0x80 | (cp & 0x3F));
    return 4;
  }
  return 0;
}

inline std::string encode_codepoint(char32_t cp) {
  char buff[4];
  auto l = encode_codepoint(cp, buff);
  return std::string(buff, l);
}

inline bool decode_codepoint(const char *s8, size_t l, size_t &bytes,
                             char32_t &cp) {
  if (l) {
    auto b = static_cast<uint8_t>(s8[0]);
    if ((b & 0x80) == 0) {
      bytes = 1;
      cp = b;
      return true;
    } else if ((b & 0xE0) == 0xC0) {
      if (l >= 2) {
        bytes = 2;
        cp = ((static_cast<char32_t>(s8[0] & 0x1F)) << 6) |
             (static_cast<char32_t>(s8[1] & 0x3F));
        return true;
      }
    } else if ((b & 0xF0) == 0xE0) {
      if (l >= 3) {
        bytes = 3;
        cp = ((static_cast<char32_t>(s8[0] & 0x0F)) << 12) |
             ((static_cast<char32_t>(s8[1] & 0x3F)) << 6) |
             (static_cast<char32_t>(s8[2] & 0x3F));
        return true;
      }
    } else if ((b & 0xF8) == 0xF0) {
      if (l >= 4) {
        bytes = 4;
        cp = ((static_cast<char32_t>(s8[0] & 0x07)) << 18) |
             ((static_cast<char32_t>(s8[1] & 0x3F)) << 12) |
             ((static_cast<char32_t>(s8[2] & 0x3F)) << 6) |
             (static_cast<char32_t>(s8[3] & 0x3F));
        return true;
      }
    }
  }
  return false;
}

inline size_t decode_codepoint(const char *s8, size_t l, char32_t &out) {
  size_t bytes;
  if (decode_codepoint(s8, l, bytes, out)) {
    return bytes;
  }
  return 0;
}

inline char32_t decode_codepoint(const char *s8, size_t l) {
  char32_t out = 0;
  decode_codepoint(s8, l, out);
  return out;
}

inline std::u32string decode(const char *s8, size_t l) {
  std::u32string out;
  size_t i = 0;
  while (i < l) {
    auto beg = i++;
    while (i < l && (s8[i] & 0xc0) == 0x80) {
      i++;
    }
    out += decode_codepoint(&s8[beg], (i - beg));
  }
  return out;
}

/*-----------------------------------------------------------------------------
 *  resolve_escape_sequence
 *---------------------------------------------------------------------------*/

inline bool is_hex(char c, int& v) {
    if ('0' <= c && c <= '9') {
        v = c - '0';
        return true;
    } else if ('a' <= c && c <= 'f') {
        v = c - 'a' + 10;
        return true;
    } else if ('A' <= c && c <= 'F') {
        v = c - 'A' + 10;
        return true;
    }
    return false;
}

inline bool is_digit(char c, int& v) {
    if ('0' <= c && c <= '9') {
        v = c - '0';
        return true;
    }
    return false;
}

inline std::pair<int, size_t> parse_hex_number(const char* s, size_t n, size_t i) {
    int ret = 0;
    int val;
    while (i < n && is_hex(s[i], val)) {
        ret = static_cast<int>(ret * 16 + val);
        i++;
    }
    return std::make_pair(ret, i);
}

inline std::pair<int, size_t> parse_octal_number(const char* s, size_t n, size_t i) {
    int ret = 0;
    int val;
    while (i < n && is_digit(s[i], val)) {
        ret = static_cast<int>(ret * 8 + val);
        i++;
    }
    return std::make_pair(ret, i);
}

inline std::string resolve_escape_sequence(const char* s, size_t n) {
    std::string r;
    r.reserve(n);

    size_t i = 0;
    while (i < n) {
        auto ch = s[i];
        if (ch == '\\') {
            i++;
            switch (s[i]) {
                case 'n':  r += '\n'; i++; break;
                case 'r':  r += '\r'; i++; break;
                case 't':  r += '\t'; i++; break;
                case '\'': r += '\''; i++; break;
                case '"':  r += '"';  i++; break;
                case '[':  r += '[';  i++; break;
                case ']':  r += ']';  i++; break;
                case '\\': r += '\\'; i++; break;
                case 'x':
                case 'u': {
                    char32_t cp;
                    std::tie(cp, i) = parse_hex_number(s, n, i + 1);
                    r += encode_codepoint(cp);
                    break;
                }
                default: {
                    char32_t cp;
                    std::tie(cp, i) = parse_octal_number(s, n, i);
                    r += encode_codepoint(cp);
                    break;
                }
            }
        } else {
            r += ch;
            i++;
        }
    }
    return r;
}

/*-----------------------------------------------------------------------------
 *  PEG
 *---------------------------------------------------------------------------*/

/*
* Line information utility function
*/
inline std::pair<size_t, size_t> line_info(const char* start, const char* cur) {
    auto p = start;
    auto col_ptr = p;
    auto no = 1;

    while (p < cur) {
        if (*p == '\n') {
            no++;
            col_ptr = p + 1;
        }
        p++;
    }

    auto col = p - col_ptr + 1;

    return std::make_pair(no, col);
}

/*
* Semantic values
*/
struct SemanticValues : protected std::vector<any>
{
    // Input text
    const char* path;
    const char* ss;

    // Matched string
    const char* c_str() const { return s_; }
    size_t      length() const { return n_; }

    std::string str() const {
        return std::string(s_, n_);
    }

    // Definition name
    const std::string& name() const { return name_; }

    // Line number and column at which the matched string is
    std::pair<size_t, size_t> line_info() const {
        return peg::line_info(ss, s_);
    }

    // Choice count
    size_t      choice_count() const { return choice_count_; }

    // Choice number (0 based index)
    size_t      choice() const { return choice_; }

    // Tokens
    std::vector<std::pair<const char*, size_t>> tokens;

    std::string token(size_t id = 0) const {
        if (!tokens.empty()) {
            assert(id < tokens.size());
            const auto& tok = tokens[id];
            return std::string(tok.first, tok.second);
        }
        return std::string(s_, n_);
    }

    // Transform the semantic value vector to another vector
    template <typename T>
    auto transform(size_t beg = 0, size_t end = static_cast<size_t>(-1)) const -> vector<T> {
        return this->transform(beg, end, [](const any& v) { return v.get<T>(); });
    }

    SemanticValues() : s_(nullptr), n_(0), choice_count_(0), choice_(0) {}

    using std::vector<any>::iterator;
    using std::vector<any>::const_iterator;
    using std::vector<any>::size;
    using std::vector<any>::empty;
    using std::vector<any>::assign;
    using std::vector<any>::begin;
    using std::vector<any>::end;
    using std::vector<any>::rbegin;
    using std::vector<any>::rend;
    using std::vector<any>::operator[];
    using std::vector<any>::at;
    using std::vector<any>::resize;
    using std::vector<any>::front;
    using std::vector<any>::back;
    using std::vector<any>::push_back;
    using std::vector<any>::pop_back;
    using std::vector<any>::insert;
    using std::vector<any>::erase;
    using std::vector<any>::clear;
    using std::vector<any>::swap;
    using std::vector<any>::emplace;
    using std::vector<any>::emplace_back;

private:
    friend class Context;
    friend class Sequence;
    friend class PrioritizedChoice;
    friend class Holder;

    const char* s_;
    size_t      n_;
    size_t      choice_count_;
    size_t      choice_;
    std::string name_;

    template <typename F>
    auto transform(F f) const -> vector<typename std::remove_const<decltype(f(any()))>::type> {
        vector<typename std::remove_const<decltype(f(any()))>::type> r;
        for (const auto& v: *this) {
            r.emplace_back(f(v));
        }
        return r;
    }

    template <typename F>
    auto transform(size_t beg, size_t end, F f) const -> vector<typename std::remove_const<decltype(f(any()))>::type> {
        vector<typename std::remove_const<decltype(f(any()))>::type> r;
        end = (std::min)(end, size());
        for (size_t i = beg; i < end; i++) {
            r.emplace_back(f((*this)[i]));
        }
        return r;
    }

    void reset() {
        path = nullptr;
        ss = nullptr;
        tokens.clear();

        s_ = nullptr;
        n_ = 0;
        choice_count_ = 0;
        choice_ = 0;
    }
};

/*
 * Semantic action
 */
template <
    typename R, typename F,
    typename std::enable_if<std::is_void<R>::value, std::nullptr_t>::type = nullptr,
    typename... Args>
any call(F fn, Args&&... args) {
    fn(std::forward<Args>(args)...);
    return any();
}

template <
    typename R, typename F,
    typename std::enable_if<std::is_same<typename std::remove_cv<R>::type, any>::value, std::nullptr_t>::type = nullptr,
    typename... Args>
any call(F fn, Args&&... args) {
    return fn(std::forward<Args>(args)...);
}

template <
    typename R, typename F,
    typename std::enable_if<
        !std::is_void<R>::value &&
        !std::is_same<typename std::remove_cv<R>::type, any>::value, std::nullptr_t>::type = nullptr,
    typename... Args>
any call(F fn, Args&&... args) {
    return any(fn(std::forward<Args>(args)...));
}

class Action
{
public:
    Action() = default;

    Action(const Action& rhs) : fn_(rhs.fn_) {}

    template <typename F, typename std::enable_if<!std::is_pointer<F>::value && !std::is_same<F, std::nullptr_t>::value, std::nullptr_t>::type = nullptr>
    Action(F fn) : fn_(make_adaptor(fn, &F::operator())) {}

    template <typename F, typename std::enable_if<std::is_pointer<F>::value, std::nullptr_t>::type = nullptr>
    Action(F fn) : fn_(make_adaptor(fn, fn)) {}

    template <typename F, typename std::enable_if<std::is_same<F, std::nullptr_t>::value, std::nullptr_t>::type = nullptr>
    Action(F /*fn*/) {}

    template <typename F, typename std::enable_if<!std::is_pointer<F>::value && !std::is_same<F, std::nullptr_t>::value, std::nullptr_t>::type = nullptr>
    void operator=(F fn) {
        fn_ = make_adaptor(fn, &F::operator());
    }

    template <typename F, typename std::enable_if<std::is_pointer<F>::value, std::nullptr_t>::type = nullptr>
    void operator=(F fn) {
        fn_ = make_adaptor(fn, fn);
    }

    template <typename F, typename std::enable_if<std::is_same<F, std::nullptr_t>::value, std::nullptr_t>::type = nullptr>
    void operator=(F /*fn*/) {}

    Action& operator=(const Action& rhs) = default;

    operator bool() const {
        return bool(fn_);
    }

    any operator()(SemanticValues& sv, any& dt) const {
        return fn_(sv, dt);
    }

private:
    template <typename R>
    struct TypeAdaptor_sv {
        TypeAdaptor_sv(std::function<R (SemanticValues& sv)> fn)
            : fn_(fn) {}
        any operator()(SemanticValues& sv, any& /*dt*/) {
            return call<R>(fn_, sv);
        }
        std::function<R (SemanticValues& sv)> fn_;
    };

    template <typename R>
    struct TypeAdaptor_csv {
        TypeAdaptor_csv(std::function<R (const SemanticValues& sv)> fn)
            : fn_(fn) {}
        any operator()(SemanticValues& sv, any& /*dt*/) {
            return call<R>(fn_, sv);
        }
        std::function<R (const SemanticValues& sv)> fn_;
    };

    template <typename R>
    struct TypeAdaptor_sv_dt {
        TypeAdaptor_sv_dt(std::function<R (SemanticValues& sv, any& dt)> fn)
            : fn_(fn) {}
        any operator()(SemanticValues& sv, any& dt) {
            return call<R>(fn_, sv, dt);
        }
        std::function<R (SemanticValues& sv, any& dt)> fn_;
    };

    template <typename R>
    struct TypeAdaptor_csv_dt {
        TypeAdaptor_csv_dt(std::function<R (const SemanticValues& sv, any& dt)> fn)
            : fn_(fn) {}
        any operator()(SemanticValues& sv, any& dt) {
            return call<R>(fn_, sv, dt);
        }
        std::function<R (const SemanticValues& sv, any& dt)> fn_;
    };

    typedef std::function<any (SemanticValues& sv, any& dt)> Fty;

    template<typename F, typename R>
    Fty make_adaptor(F fn, R (F::* /*mf*/)(SemanticValues& sv) const) {
        return TypeAdaptor_sv<R>(fn);
    }

    template<typename F, typename R>
    Fty make_adaptor(F fn, R (F::* /*mf*/)(const SemanticValues& sv) const) {
        return TypeAdaptor_csv<R>(fn);
    }

    template<typename F, typename R>
    Fty make_adaptor(F fn, R (F::* /*mf*/)(SemanticValues& sv)) {
        return TypeAdaptor_sv<R>(fn);
    }

    template<typename F, typename R>
    Fty make_adaptor(F fn, R (F::* /*mf*/)(const SemanticValues& sv)) {
        return TypeAdaptor_csv<R>(fn);
    }

    template<typename F, typename R>
    Fty make_adaptor(F fn, R (* /*mf*/)(SemanticValues& sv)) {
        return TypeAdaptor_sv<R>(fn);
    }

    template<typename F, typename R>
    Fty make_adaptor(F fn, R (* /*mf*/)(const SemanticValues& sv)) {
        return TypeAdaptor_csv<R>(fn);
    }

    template<typename F, typename R>
    Fty make_adaptor(F fn, R (F::* /*mf*/)(SemanticValues& sv, any& dt) const) {
        return TypeAdaptor_sv_dt<R>(fn);
    }

    template<typename F, typename R>
    Fty make_adaptor(F fn, R (F::* /*mf*/)(const SemanticValues& sv, any& dt) const) {
        return TypeAdaptor_csv_dt<R>(fn);
    }

    template<typename F, typename R>
    Fty make_adaptor(F fn, R (F::* /*mf*/)(SemanticValues& sv, any& dt)) {
        return TypeAdaptor_sv_dt<R>(fn);
    }

    template<typename F, typename R>
    Fty make_adaptor(F fn, R (F::* /*mf*/)(const SemanticValues& sv, any& dt)) {
        return TypeAdaptor_csv_dt<R>(fn);
    }

    template<typename F, typename R>
    Fty make_adaptor(F fn, R(* /*mf*/)(SemanticValues& sv, any& dt)) {
        return TypeAdaptor_sv_dt<R>(fn);
    }

    template<typename F, typename R>
    Fty make_adaptor(F fn, R(* /*mf*/)(const SemanticValues& sv, any& dt)) {
        return TypeAdaptor_csv_dt<R>(fn);
    }

    Fty fn_;
};

/*
 * Semantic predicate
 */
// Note: 'parse_error' exception class should be be used in sematic action handlers to reject the rule.
struct parse_error {
    parse_error() = default;
    parse_error(const char* s) : s_(s) {}
    const char* what() const { return s_.empty() ? nullptr : s_.c_str(); }
private:
    std::string s_;
};

/*
 * Result
 */
inline bool success(size_t len) {
    return len != static_cast<size_t>(-1);
}

inline bool fail(size_t len) {
    return len == static_cast<size_t>(-1);
}

/*
 * Context
 */
class Context;
class Ope;
class Definition;

typedef std::function<void (const char* name, const char* s, size_t n, const SemanticValues& sv, const Context& c, const any& dt)> Tracer;

class Context
{
public:
    const char*                                  path;
    const char*                                  s;
    const size_t                                 l;

    const char*                                  error_pos;
    const char*                                  message_pos;
    std::string                                  message; // TODO: should be `int`.

    std::vector<std::shared_ptr<SemanticValues>> value_stack;
    size_t                                       value_stack_size;
    std::vector<std::vector<std::shared_ptr<Ope>>> args_stack;

    size_t                                       nest_level;

    bool                                         in_token;

    std::shared_ptr<Ope>                         whitespaceOpe;
    bool                                         in_whitespace;

    std::shared_ptr<Ope>                         wordOpe;

    std::vector<std::unordered_map<std::string, std::string>> capture_scope_stack;

    const size_t                                 def_count;
    const bool                                   enablePackratParsing;
    std::vector<bool>                            cache_registered;
    std::vector<bool>                            cache_success;

    std::map<std::pair<size_t, size_t>, std::tuple<size_t, any>> cache_values;

    std::function<void (const char*, const char*, size_t, const SemanticValues&, const Context&, const any&)> tracer;

    Context(
        const char*          a_path,
        const char*          a_s,
        size_t               a_l,
        size_t               a_def_count,
        std::shared_ptr<Ope> a_whitespaceOpe,
        std::shared_ptr<Ope> a_wordOpe,
        bool                 a_enablePackratParsing,
        Tracer               a_tracer)
        : path(a_path)
        , s(a_s)
        , l(a_l)
        , error_pos(nullptr)
        , message_pos(nullptr)
        , value_stack_size(0)
        , nest_level(0)
        , in_token(false)
        , whitespaceOpe(a_whitespaceOpe)
        , in_whitespace(false)
        , wordOpe(a_wordOpe)
        , def_count(a_def_count)
        , enablePackratParsing(a_enablePackratParsing)
        , cache_registered(enablePackratParsing ? def_count * (l + 1) : 0)
        , cache_success(enablePackratParsing ? def_count * (l + 1) : 0)
        , tracer(a_tracer)
    {
        args_stack.resize(1);
        capture_scope_stack.resize(1);
    }

    template <typename T>
    void packrat(const char* a_s, size_t def_id, size_t& len, any& val, T fn) {
        if (!enablePackratParsing) {
            fn(val);
            return;
        }

        auto col = a_s - s;
        auto idx = def_count * static_cast<size_t>(col) + def_id;

        if (cache_registered[idx]) {
            if (cache_success[idx]) {
                auto key = std::make_pair(col, def_id);
                std::tie(len, val) = cache_values[key];
                return;
            } else {
                len = static_cast<size_t>(-1);
                return;
            }
        } else {
            fn(val);
            cache_registered[idx] = true;
            cache_success[idx] = success(len);
            if (success(len)) {
                auto key = std::make_pair(col, def_id);
                cache_values[key] = std::make_pair(len, val);
            }
            return;
        }
    }

    SemanticValues& push() {
        assert(value_stack_size <= value_stack.size());
        if (value_stack_size == value_stack.size()) {
            value_stack.emplace_back(std::make_shared<SemanticValues>());
        }
        auto& sv = *value_stack[value_stack_size++];
        if (!sv.empty()) {
            sv.clear();
        }
        sv.reset();
        sv.path = path;
        sv.ss = s;
        return sv;
    }

    void pop() {
        value_stack_size--;
    }

    void push_args(const std::vector<std::shared_ptr<Ope>>& args) {
        args_stack.push_back(args);
    }

    void pop_args() {
        args_stack.pop_back();
    }

    const std::vector<std::shared_ptr<Ope>>& top_args() const {
        return args_stack[args_stack.size() - 1];
    }

    void push_capture_scope() {
        capture_scope_stack.resize(capture_scope_stack.size() + 1);
    }

    void pop_capture_scope() {
        capture_scope_stack.pop_back();
    }

    void shift_capture_values() {
        assert(capture_scope_stack.size() >= 2);
        auto it = capture_scope_stack.rbegin();
        auto it_prev = it + 1;
        for (const auto& kv: *it) {
            (*it_prev)[kv.first] = kv.second;
        }
    }

    void set_error_pos(const char* a_s) {
        if (error_pos < a_s) error_pos = a_s;
    }

    void trace(const char* name, const char* a_s, size_t n, SemanticValues& sv, any& dt) const {
        if (tracer) tracer(name, a_s, n, sv, *this, dt);
    }
};

/*
 * Parser operators
 */
class Ope
{
public:
    struct Visitor;

    virtual ~Ope() {}
    virtual size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const = 0;
    virtual void accept(Visitor& v) = 0;
};

class Sequence : public Ope
{
public:
    Sequence(const Sequence& rhs) : opes_(rhs.opes_) {}

#if defined(_MSC_VER) && _MSC_VER < 1900 // Less than Visual Studio 2015
    // NOTE: Compiler Error C2797 on Visual Studio 2013
    // "The C++ compiler in Visual Studio does not implement list
    // initialization inside either a member initializer list or a non-static
    // data member initializer. Before Visual Studio 2013 Update 3, this was
    // silently converted to a function call, which could lead to bad code
    // generation. Visual Studio 2013 Update 3 reports this as an error."
    template <typename... Args>
    Sequence(const Args& ...args) {
        opes_ = std::vector<std::shared_ptr<Ope>>{ static_cast<std::shared_ptr<Ope>>(args)... };
    }
#else
    template <typename... Args>
    Sequence(const Args& ...args) : opes_{ static_cast<std::shared_ptr<Ope>>(args)... } {}
#endif

    Sequence(const std::vector<std::shared_ptr<Ope>>& opes) : opes_(opes) {}
    Sequence(std::vector<std::shared_ptr<Ope>>&& opes) : opes_(opes) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.trace("Sequence", s, n, sv, dt);
        auto& chldsv = c.push();
        size_t i = 0;
        for (const auto& ope : opes_) {
            c.nest_level++;
            auto se = make_scope_exit([&]() { c.nest_level--; });
            const auto& rule = *ope;
            auto len = rule.parse(s + i, n - i, chldsv, c, dt);
            if (fail(len)) {
                return static_cast<size_t>(-1);
            }
            i += len;
        }
        sv.insert(sv.end(), chldsv.begin(), chldsv.end());
        sv.s_ = chldsv.c_str();
        sv.n_ = chldsv.length();
        sv.tokens.insert(sv.tokens.end(), chldsv.tokens.begin(), chldsv.tokens.end());
        return i;
    }

    void accept(Visitor& v) override;

    std::vector<std::shared_ptr<Ope>> opes_;
};

class PrioritizedChoice : public Ope
{
public:
#if defined(_MSC_VER) && _MSC_VER < 1900 // Less than Visual Studio 2015
    // NOTE: Compiler Error C2797 on Visual Studio 2013
    // "The C++ compiler in Visual Studio does not implement list
    // initialization inside either a member initializer list or a non-static
    // data member initializer. Before Visual Studio 2013 Update 3, this was
    // silently converted to a function call, which could lead to bad code
    // generation. Visual Studio 2013 Update 3 reports this as an error."
    template <typename... Args>
    PrioritizedChoice(const Args& ...args) {
        opes_ = std::vector<std::shared_ptr<Ope>>{ static_cast<std::shared_ptr<Ope>>(args)... };
    }
#else
    template <typename... Args>
    PrioritizedChoice(const Args& ...args) : opes_{ static_cast<std::shared_ptr<Ope>>(args)... } {}
#endif

    PrioritizedChoice(const std::vector<std::shared_ptr<Ope>>& opes) : opes_(opes) {}
    PrioritizedChoice(std::vector<std::shared_ptr<Ope>>&& opes) : opes_(opes) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.trace("PrioritizedChoice", s, n, sv, dt);
        size_t id = 0;
        for (const auto& ope : opes_) {
            c.nest_level++;
            auto& chldsv = c.push();
            c.push_capture_scope();
            auto se = make_scope_exit([&]() {
                c.nest_level--;
                c.pop();
                c.pop_capture_scope();
            });
            const auto& rule = *ope;
            auto len = rule.parse(s, n, chldsv, c, dt);
            if (success(len)) {
                sv.insert(sv.end(), chldsv.begin(), chldsv.end());
                sv.s_ = chldsv.c_str();
                sv.n_ = chldsv.length();
                sv.choice_count_ = opes_.size();
                sv.choice_ = id;
                sv.tokens.insert(sv.tokens.end(), chldsv.tokens.begin(), chldsv.tokens.end());

                c.shift_capture_values();
                return len;
            }
            id++;
        }
        return static_cast<size_t>(-1);
    }

    void accept(Visitor& v) override;

    size_t size() const { return opes_.size();  }

    std::vector<std::shared_ptr<Ope>> opes_;
};

class ZeroOrMore : public Ope
{
public:
    ZeroOrMore(const std::shared_ptr<Ope>& ope) : ope_(ope) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.trace("ZeroOrMore", s, n, sv, dt);
        auto save_error_pos = c.error_pos;
        size_t i = 0;
        while (n - i > 0) {
            c.nest_level++;
            c.push_capture_scope();
            auto se = make_scope_exit([&]() {
                c.nest_level--;
                c.pop_capture_scope();
            });
            auto save_sv_size = sv.size();
            auto save_tok_size = sv.tokens.size();
            const auto& rule = *ope_;
            auto len = rule.parse(s + i, n - i, sv, c, dt);
            if (success(len)) {
                c.shift_capture_values();
            } else {
                if (sv.size() != save_sv_size) {
                    sv.erase(sv.begin() + static_cast<std::ptrdiff_t>(save_sv_size));
                }
                if (sv.tokens.size() != save_tok_size) {
                    sv.tokens.erase(sv.tokens.begin() + static_cast<std::ptrdiff_t>(save_tok_size));
                }
                c.error_pos = save_error_pos;
                break;
            }
            i += len;
        }
        return i;
    }

    void accept(Visitor& v) override;

    std::shared_ptr<Ope> ope_;
};

class OneOrMore : public Ope
{
public:
    OneOrMore(const std::shared_ptr<Ope>& ope) : ope_(ope) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.trace("OneOrMore", s, n, sv, dt);
        size_t len = 0;
        {
            c.nest_level++;
            c.push_capture_scope();
            auto se = make_scope_exit([&]() {
                c.nest_level--;
                c.pop_capture_scope();
            });
            const auto& rule = *ope_;
            len = rule.parse(s, n, sv, c, dt);
            if (success(len)) {
                c.shift_capture_values();
            } else {
                return static_cast<size_t>(-1);
            }
        }
        auto save_error_pos = c.error_pos;
        auto i = len;
        while (n - i > 0) {
            c.nest_level++;
            c.push_capture_scope();
            auto se = make_scope_exit([&]() {
                c.nest_level--;
                c.pop_capture_scope();
            });
            auto save_sv_size = sv.size();
            auto save_tok_size = sv.tokens.size();
            const auto& rule = *ope_;
            len = rule.parse(s + i, n - i, sv, c, dt);
            if (success(len)) {
                c.shift_capture_values();
            } else {
                if (sv.size() != save_sv_size) {
                    sv.erase(sv.begin() + static_cast<std::ptrdiff_t>(save_sv_size));
                }
                if (sv.tokens.size() != save_tok_size) {
                    sv.tokens.erase(sv.tokens.begin() + static_cast<std::ptrdiff_t>(save_tok_size));
                }
                c.error_pos = save_error_pos;
                break;
            }
            i += len;
        }
        return i;
    }

    void accept(Visitor& v) override;

    std::shared_ptr<Ope> ope_;
};

class Option : public Ope
{
public:
    Option(const std::shared_ptr<Ope>& ope) : ope_(ope) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.trace("Option", s, n, sv, dt);
        auto save_error_pos = c.error_pos;
        c.nest_level++;
        auto save_sv_size = sv.size();
        auto save_tok_size = sv.tokens.size();
        c.push_capture_scope();
        auto se = make_scope_exit([&]() {
            c.nest_level--;
            c.pop_capture_scope();
        });
        const auto& rule = *ope_;
        auto len = rule.parse(s, n, sv, c, dt);
        if (success(len)) {
            c.shift_capture_values();
            return len;
        } else {
            if (sv.size() != save_sv_size) {
                sv.erase(sv.begin() + static_cast<std::ptrdiff_t>(save_sv_size));
            }
            if (sv.tokens.size() != save_tok_size) {
                sv.tokens.erase(sv.tokens.begin() + static_cast<std::ptrdiff_t>(save_tok_size));
            }
            c.error_pos = save_error_pos;
            return 0;
        }
    }

    void accept(Visitor& v) override;

    std::shared_ptr<Ope> ope_;
};

class AndPredicate : public Ope
{
public:
    AndPredicate(const std::shared_ptr<Ope>& ope) : ope_(ope) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.trace("AndPredicate", s, n, sv, dt);
        c.nest_level++;
        auto& chldsv = c.push();
        c.push_capture_scope();
        auto se = make_scope_exit([&]() {
            c.nest_level--;
            c.pop();
            c.pop_capture_scope();
        });
        const auto& rule = *ope_;
        auto len = rule.parse(s, n, chldsv, c, dt);
        if (success(len)) {
            return 0;
        } else {
            return static_cast<size_t>(-1);
        }
    }

    void accept(Visitor& v) override;

    std::shared_ptr<Ope> ope_;
};

class NotPredicate : public Ope
{
public:
    NotPredicate(const std::shared_ptr<Ope>& ope) : ope_(ope) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.trace("NotPredicate", s, n, sv, dt);
        auto save_error_pos = c.error_pos;
        c.nest_level++;
        auto& chldsv = c.push();
        c.push_capture_scope();
        auto se = make_scope_exit([&]() {
            c.nest_level--;
            c.pop();
            c.pop_capture_scope();
        });
        const auto& rule = *ope_;
        auto len = rule.parse(s, n, chldsv, c, dt);
        if (success(len)) {
            c.set_error_pos(s);
            return static_cast<size_t>(-1);
        } else {
            c.error_pos = save_error_pos;
            return 0;
        }
    }

    void accept(Visitor& v) override;

    std::shared_ptr<Ope> ope_;
};

class LiteralString : public Ope
    , public std::enable_shared_from_this<LiteralString>
{
public:
    LiteralString(const std::string& s)
        : lit_(s)
        , init_is_word_(false)
        , is_word_(false)
        {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override;

    void accept(Visitor& v) override;

    std::string lit_;
    mutable bool init_is_word_;
    mutable bool is_word_;
};

class CharacterClass : public Ope
    , public std::enable_shared_from_this<CharacterClass>
{
public:
    CharacterClass(const std::string& s) {
        auto chars = decode(s.c_str(), s.length());
        auto i = 0u;
        while (i < chars.size()) {
            if (i + 2 < chars.size() && chars[i + 1] == '-') {
                auto cp1 = chars[i];
                auto cp2 = chars[i + 2];
                ranges_.emplace_back(std::make_pair(cp1, cp2));
                i += 3;
            } else {
                auto cp = chars[i];
                ranges_.emplace_back(std::make_pair(cp, cp));
                i += 1;
            }
        }
    }

    CharacterClass(const std::vector<std::pair<char32_t, char32_t>>& ranges) : ranges_(ranges) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.trace("CharacterClass", s, n, sv, dt);

        if (n < 1) {
            c.set_error_pos(s);
            return static_cast<size_t>(-1);
        }

        char32_t cp;
        auto len = decode_codepoint(s, n, cp);

        if (!ranges_.empty()) {
            for (const auto& range: ranges_) {
                if (range.first <= cp && cp <= range.second) {
                    return len;
                }
            }
        }

        c.set_error_pos(s);
        return static_cast<size_t>(-1);
    }

    void accept(Visitor& v) override;

    std::vector<std::pair<char32_t, char32_t>> ranges_;
};

class Character : public Ope
    , public std::enable_shared_from_this<Character>
{
public:
    Character(char ch) : ch_(ch) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.trace("Character", s, n, sv, dt);
        if (n < 1 || s[0] != ch_) {
            c.set_error_pos(s);
            return static_cast<size_t>(-1);
        }
        return 1;
    }

    void accept(Visitor& v) override;

    char ch_;
};

class AnyCharacter : public Ope
    , public std::enable_shared_from_this<AnyCharacter>
{
public:
    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.trace("AnyCharacter", s, n, sv, dt);
        auto len = codepoint_length(s, n);
        if (len < 1) {
            c.set_error_pos(s);
            return static_cast<size_t>(-1);
        }
        return len;
    }

    void accept(Visitor& v) override;
};

class CaptureScope : public Ope
{
public:
    CaptureScope(const std::shared_ptr<Ope>& ope)
        : ope_(ope) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.push_capture_scope();
        auto se = make_scope_exit([&]() {
            c.pop_capture_scope();
        });
        const auto& rule = *ope_;
        auto len = rule.parse(s, n, sv, c, dt);
        return len;
    }

    void accept(Visitor& v) override;

    std::shared_ptr<Ope> ope_;
};

class Capture : public Ope
{
public:
    typedef std::function<void (const char* s, size_t n, Context& c)> MatchAction;

    Capture(const std::shared_ptr<Ope>& ope, MatchAction ma)
        : ope_(ope), match_action_(ma) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        const auto& rule = *ope_;
        auto len = rule.parse(s, n, sv, c, dt);
        if (success(len) && match_action_) {
            match_action_(s, len, c);
        }
        return len;
    }

    void accept(Visitor& v) override;

    std::shared_ptr<Ope> ope_;
    MatchAction          match_action_;
};

class TokenBoundary : public Ope
{
public:
    TokenBoundary(const std::shared_ptr<Ope>& ope) : ope_(ope) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override;

    void accept(Visitor& v) override;

    std::shared_ptr<Ope> ope_;
};

class Ignore : public Ope
{
public:
    Ignore(const std::shared_ptr<Ope>& ope) : ope_(ope) {}

    size_t parse(const char* s, size_t n, SemanticValues& /*sv*/, Context& c, any& dt) const override {
        const auto& rule = *ope_;
        auto& chldsv = c.push();
        auto se = make_scope_exit([&]() {
            c.pop();
        });
        return rule.parse(s, n, chldsv, c, dt);
    }

    void accept(Visitor& v) override;

    std::shared_ptr<Ope> ope_;
};

typedef std::function<size_t (const char* s, size_t n, SemanticValues& sv, any& dt)> Parser;

class User : public Ope
{
public:
    User(Parser fn) : fn_(fn) {}
     size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        c.trace("User", s, n, sv, dt);
        assert(fn_);
        return fn_(s, n, sv, dt);
    }
     void accept(Visitor& v) override;
     std::function<size_t (const char* s, size_t n, SemanticValues& sv, any& dt)> fn_;
};

class WeakHolder : public Ope
{
public:
    WeakHolder(const std::shared_ptr<Ope>& ope) : weak_(ope) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        auto ope = weak_.lock();
        assert(ope);
        const auto& rule = *ope;
        return rule.parse(s, n, sv, c, dt);
    }

    void accept(Visitor& v) override;

    std::weak_ptr<Ope> weak_;
};

class Holder : public Ope
{
public:
    Holder(Definition* outer)
       : outer_(outer) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override;

    void accept(Visitor& v) override;

    any reduce(SemanticValues& sv, any& dt) const;

    std::shared_ptr<Ope> ope_;
    Definition*          outer_;

    friend class Definition;
};

typedef std::unordered_map<std::string, Definition> Grammar;

class Reference : public Ope
    , public std::enable_shared_from_this<Reference>
{
public:
    Reference(
        const Grammar& grammar,
        const std::string& name,
        const char* s,
        bool is_macro,
        const std::vector<std::shared_ptr<Ope>>& args)
        : grammar_(grammar)
        , name_(name)
        , s_(s)
        , is_macro_(is_macro)
        , args_(args)
        , rule_(nullptr)
        , iarg_(0)
        {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override;

    void accept(Visitor& v) override;

    std::shared_ptr<Ope> get_core_operator() const;

    const Grammar&    grammar_;
    const std::string name_;
    const char*       s_;

    const bool is_macro_;
    const std::vector<std::shared_ptr<Ope>> args_;

    Definition* rule_;
    size_t iarg_;
};

class Whitespace : public Ope
{
public:
    Whitespace(const std::shared_ptr<Ope>& ope) : ope_(ope) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override {
        if (c.in_whitespace) {
            return 0;
        }
        c.in_whitespace = true;
        auto se = make_scope_exit([&]() { c.in_whitespace = false; });
        const auto& rule = *ope_;
        return rule.parse(s, n, sv, c, dt);
    }

    void accept(Visitor& v) override;

    std::shared_ptr<Ope> ope_;
};

class BackReference : public Ope
{
public:
    BackReference(const std::string& name) : name_(name) {}

    size_t parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const override;

    void accept(Visitor& v) override;

    std::string name_;
};

/*
 * Factories
 */
template <typename... Args>
std::shared_ptr<Ope> seq(Args&& ...args) {
    return std::make_shared<Sequence>(static_cast<std::shared_ptr<Ope>>(args)...);
}

template <typename... Args>
std::shared_ptr<Ope> cho(Args&& ...args) {
    return std::make_shared<PrioritizedChoice>(static_cast<std::shared_ptr<Ope>>(args)...);
}

inline std::shared_ptr<Ope> zom(const std::shared_ptr<Ope>& ope) {
    return std::make_shared<ZeroOrMore>(ope);
}

inline std::shared_ptr<Ope> oom(const std::shared_ptr<Ope>& ope) {
    return std::make_shared<OneOrMore>(ope);
}

inline std::shared_ptr<Ope> opt(const std::shared_ptr<Ope>& ope) {
    return std::make_shared<Option>(ope);
}

inline std::shared_ptr<Ope> apd(const std::shared_ptr<Ope>& ope) {
    return std::make_shared<AndPredicate>(ope);
}

inline std::shared_ptr<Ope> npd(const std::shared_ptr<Ope>& ope) {
    return std::make_shared<NotPredicate>(ope);
}

inline std::shared_ptr<Ope> lit(const std::string& lit) {
    return std::make_shared<LiteralString>(lit);
}

inline std::shared_ptr<Ope> cls(const std::string& s) {
    return std::make_shared<CharacterClass>(s);
}

inline std::shared_ptr<Ope> cls(const std::vector<std::pair<char32_t, char32_t>>& ranges) {
    return std::make_shared<CharacterClass>(ranges);
}

inline std::shared_ptr<Ope> chr(char dt) {
    return std::make_shared<Character>(dt);
}

inline std::shared_ptr<Ope> dot() {
    return std::make_shared<AnyCharacter>();
}

inline std::shared_ptr<Ope> csc(const std::shared_ptr<Ope>& ope) {
    return std::make_shared<CaptureScope>(ope);
}

inline std::shared_ptr<Ope> cap(const std::shared_ptr<Ope>& ope, Capture::MatchAction ma) {
    return std::make_shared<Capture>(ope, ma);
}

inline std::shared_ptr<Ope> tok(const std::shared_ptr<Ope>& ope) {
    return std::make_shared<TokenBoundary>(ope);
}

inline std::shared_ptr<Ope> ign(const std::shared_ptr<Ope>& ope) {
    return std::make_shared<Ignore>(ope);
}

inline std::shared_ptr<Ope> usr(std::function<size_t (const char* s, size_t n, SemanticValues& sv, any& dt)> fn) {
    return std::make_shared<User>(fn);
}

inline std::shared_ptr<Ope> ref(const Grammar& grammar, const std::string& name, const char* s, bool is_macro, const std::vector<std::shared_ptr<Ope>>& args) {
    return std::make_shared<Reference>(grammar, name, s, is_macro, args);
}

inline std::shared_ptr<Ope> wsp(const std::shared_ptr<Ope>& ope) {
    return std::make_shared<Whitespace>(std::make_shared<Ignore>(ope));
}

inline std::shared_ptr<Ope> bkr(const std::string& name) {
    return std::make_shared<BackReference>(name);
}

/*
 * Visitor
 */
struct Ope::Visitor
{
    virtual ~Visitor() {}
    virtual void visit(Sequence& /*ope*/) {}
    virtual void visit(PrioritizedChoice& /*ope*/) {}
    virtual void visit(ZeroOrMore& /*ope*/) {}
    virtual void visit(OneOrMore& /*ope*/) {}
    virtual void visit(Option& /*ope*/) {}
    virtual void visit(AndPredicate& /*ope*/) {}
    virtual void visit(NotPredicate& /*ope*/) {}
    virtual void visit(LiteralString& /*ope*/) {}
    virtual void visit(CharacterClass& /*ope*/) {}
    virtual void visit(Character& /*ope*/) {}
    virtual void visit(AnyCharacter& /*ope*/) {}
    virtual void visit(CaptureScope& /*ope*/) {}
    virtual void visit(Capture& /*ope*/) {}
    virtual void visit(TokenBoundary& /*ope*/) {}
    virtual void visit(Ignore& /*ope*/) {}
    virtual void visit(User& /*ope*/) {}
    virtual void visit(WeakHolder& /*ope*/) {}
    virtual void visit(Holder& /*ope*/) {}
    virtual void visit(Reference& /*ope*/) {}
    virtual void visit(Whitespace& /*ope*/) {}
    virtual void visit(BackReference& /*ope*/) {}
};

struct AssignIDToDefinition : public Ope::Visitor
{
    void visit(Sequence& ope) override {
        for (auto op: ope.opes_) {
            op->accept(*this);
        }
    }
    void visit(PrioritizedChoice& ope) override {
        for (auto op: ope.opes_) {
            op->accept(*this);
        }
    }
    void visit(ZeroOrMore& ope) override { ope.ope_->accept(*this); }
    void visit(OneOrMore& ope) override { ope.ope_->accept(*this); }
    void visit(Option& ope) override { ope.ope_->accept(*this); }
    void visit(AndPredicate& ope) override { ope.ope_->accept(*this); }
    void visit(NotPredicate& ope) override { ope.ope_->accept(*this); }
    void visit(CaptureScope& ope) override { ope.ope_->accept(*this); }
    void visit(Capture& ope) override { ope.ope_->accept(*this); }
    void visit(TokenBoundary& ope) override { ope.ope_->accept(*this); }
    void visit(Ignore& ope) override { ope.ope_->accept(*this); }
    void visit(WeakHolder& ope) override { ope.weak_.lock()->accept(*this); }
    void visit(Holder& ope) override;
    void visit(Reference& ope) override;
    void visit(Whitespace& ope) override { ope.ope_->accept(*this); }

    std::unordered_map<void*, size_t> ids;
};

struct TokenChecker : public Ope::Visitor
{
    TokenChecker() : has_token_boundary_(false), has_rule_(false) {}

    void visit(Sequence& ope) override {
        for (auto op: ope.opes_) {
            op->accept(*this);
        }
    }
    void visit(PrioritizedChoice& ope) override {
        for (auto op: ope.opes_) {
            op->accept(*this);
        }
    }
    void visit(ZeroOrMore& ope) override { ope.ope_->accept(*this); }
    void visit(OneOrMore& ope) override { ope.ope_->accept(*this); }
    void visit(Option& ope) override { ope.ope_->accept(*this); }
    void visit(CaptureScope& ope) override { ope.ope_->accept(*this); }
    void visit(Capture& ope) override { ope.ope_->accept(*this); }
    void visit(TokenBoundary& /*ope*/) override { has_token_boundary_ = true; }
    void visit(Ignore& ope) override { ope.ope_->accept(*this); }
    void visit(WeakHolder& ope) override { ope.weak_.lock()->accept(*this); }
    void visit(Reference& ope) override;
    void visit(Whitespace& ope) override { ope.ope_->accept(*this); }

    static bool is_token(Ope& ope) {
        TokenChecker vis;
        ope.accept(vis);
        return vis.has_token_boundary_ || !vis.has_rule_;
    }

private:
    bool has_token_boundary_;
    bool has_rule_;
};

struct DetectLeftRecursion : public Ope::Visitor {
    DetectLeftRecursion(const std::string& name)
        : error_s(nullptr), name_(name), done_(false) {}

    void visit(Sequence& ope) override {
        for (auto op: ope.opes_) {
            op->accept(*this);
            if (done_) {
                break;
            } else if (error_s) {
                done_ = true;
                break;
            }
        }
    }
    void visit(PrioritizedChoice& ope) override {
        for (auto op: ope.opes_) {
            op->accept(*this);
            if (error_s) {
                done_ = true;
                break;
            }
        }
    }
    void visit(ZeroOrMore& ope) override { ope.ope_->accept(*this); done_ = false; }
    void visit(OneOrMore& ope) override { ope.ope_->accept(*this); done_ = true; }
    void visit(Option& ope) override { ope.ope_->accept(*this); done_ = false; }
    void visit(AndPredicate& ope) override { ope.ope_->accept(*this); done_ = false; }
    void visit(NotPredicate& ope) override { ope.ope_->accept(*this); done_ = false; }
    void visit(LiteralString& ope) override { done_ = !ope.lit_.empty(); }
    void visit(CharacterClass& /*ope*/) override { done_ = true; }
    void visit(Character& /*ope*/) override { done_ = true; }
    void visit(AnyCharacter& /*ope*/) override { done_ = true; }
    void visit(CaptureScope& ope) override { ope.ope_->accept(*this); }
    void visit(Capture& ope) override { ope.ope_->accept(*this); }
    void visit(TokenBoundary& ope) override { ope.ope_->accept(*this); }
    void visit(Ignore& ope) override { ope.ope_->accept(*this); }
    void visit(User& /*ope*/) override { done_ = true; }
    void visit(WeakHolder& ope) override { ope.weak_.lock()->accept(*this); }
    void visit(Holder& ope) override { ope.ope_->accept(*this); }
    void visit(Reference& ope) override;
    void visit(Whitespace& ope) override { ope.ope_->accept(*this); }
    void visit(BackReference& /*ope*/) override { done_ = true; }

    const char* error_s;

private:
    std::string           name_;
    std::set<std::string> refs_;
    bool                  done_;
};

struct ReferenceChecker : public Ope::Visitor {
    ReferenceChecker(
        const Grammar& grammar,
        const std::vector<std::string>& params)
        : grammar_(grammar), params_(params) {}

    void visit(Sequence& ope) override {
        for (auto op: ope.opes_) {
            op->accept(*this);
        }
    }
    void visit(PrioritizedChoice& ope) override {
        for (auto op: ope.opes_) {
            op->accept(*this);
        }
    }
    void visit(ZeroOrMore& ope) override { ope.ope_->accept(*this); }
    void visit(OneOrMore& ope) override { ope.ope_->accept(*this); }
    void visit(Option& ope) override { ope.ope_->accept(*this); }
    void visit(AndPredicate& ope) override { ope.ope_->accept(*this); }
    void visit(NotPredicate& ope) override { ope.ope_->accept(*this); }
    void visit(CaptureScope& ope) override { ope.ope_->accept(*this); }
    void visit(Capture& ope) override { ope.ope_->accept(*this); }
    void visit(TokenBoundary& ope) override { ope.ope_->accept(*this); }
    void visit(Ignore& ope) override { ope.ope_->accept(*this); }
    void visit(WeakHolder& ope) override { ope.weak_.lock()->accept(*this); }
    void visit(Holder& ope) override { ope.ope_->accept(*this); }
    void visit(Reference& ope) override;
    void visit(Whitespace& ope) override { ope.ope_->accept(*this); }

    std::unordered_map<std::string, const char*> error_s;
    std::unordered_map<std::string, std::string> error_message;

private:
    const Grammar& grammar_;
    const std::vector<std::string>& params_;
};

struct LinkReferences : public Ope::Visitor {
    LinkReferences(
        Grammar& grammar,
        const std::vector<std::string>& params)
        : grammar_(grammar), params_(params) {}

    void visit(Sequence& ope) override {
        for (auto op: ope.opes_) {
            op->accept(*this);
        }
    }
    void visit(PrioritizedChoice& ope) override {
        for (auto op: ope.opes_) {
            op->accept(*this);
        }
    }
    void visit(ZeroOrMore& ope) override { ope.ope_->accept(*this); }
    void visit(OneOrMore& ope) override { ope.ope_->accept(*this); }
    void visit(Option& ope) override { ope.ope_->accept(*this); }
    void visit(AndPredicate& ope) override { ope.ope_->accept(*this); }
    void visit(NotPredicate& ope) override { ope.ope_->accept(*this); }
    void visit(CaptureScope& ope) override { ope.ope_->accept(*this); }
    void visit(Capture& ope) override { ope.ope_->accept(*this); }
    void visit(TokenBoundary& ope) override { ope.ope_->accept(*this); }
    void visit(Ignore& ope) override { ope.ope_->accept(*this); }
    void visit(WeakHolder& ope) override { ope.weak_.lock()->accept(*this); }
    void visit(Holder& ope) override { ope.ope_->accept(*this); }
    void visit(Reference& ope) override;
    void visit(Whitespace& ope) override { ope.ope_->accept(*this); }

private:
    Grammar& grammar_;
    const std::vector<std::string>& params_;
};

struct FindReference : public Ope::Visitor {
    FindReference(
        const std::vector<std::shared_ptr<Ope>>& args,
        const std::vector<std::string>& params)
        : args_(args), params_(params) {}

    void visit(Sequence& ope) override {
        std::vector<std::shared_ptr<Ope>> opes;
        for (auto o: ope.opes_) {
            o->accept(*this);
            opes.push_back(found_ope);
        }
        found_ope = std::make_shared<Sequence>(opes);
    }
    void visit(PrioritizedChoice& ope) override {
        std::vector<std::shared_ptr<Ope>> opes;
        for (auto o: ope.opes_) {
            o->accept(*this);
            opes.push_back(found_ope);
        }
        found_ope = std::make_shared<PrioritizedChoice>(opes);
    }
    void visit(ZeroOrMore& ope) override { ope.ope_->accept(*this); found_ope = zom(found_ope); }
    void visit(OneOrMore& ope) override { ope.ope_->accept(*this); found_ope = oom(found_ope); }
    void visit(Option& ope) override { ope.ope_->accept(*this); found_ope = opt(found_ope); }
    void visit(AndPredicate& ope) override { ope.ope_->accept(*this); found_ope = apd(found_ope); }
    void visit(NotPredicate& ope) override { ope.ope_->accept(*this); found_ope = npd(found_ope); }
    void visit(LiteralString& ope) override { found_ope = ope.shared_from_this(); }
    void visit(CharacterClass& ope) override { found_ope = ope.shared_from_this(); }
    void visit(Character& ope) override { found_ope = ope.shared_from_this(); }
    void visit(AnyCharacter& ope) override { found_ope = ope.shared_from_this(); }
    void visit(CaptureScope& ope) override { ope.ope_->accept(*this); found_ope = csc(found_ope); }
    void visit(Capture& ope) override { ope.ope_->accept(*this); found_ope = cap(found_ope, ope.match_action_); }
    void visit(TokenBoundary& ope) override { ope.ope_->accept(*this); found_ope = tok(found_ope); }
    void visit(Ignore& ope) override { ope.ope_->accept(*this); found_ope = ign(found_ope); }
    void visit(WeakHolder& ope) override { ope.weak_.lock()->accept(*this); }
    void visit(Holder& ope) override { ope.ope_->accept(*this); }
    void visit(Reference& ope) override;
    void visit(Whitespace& ope) override { ope.ope_->accept(*this); found_ope = wsp(found_ope); }

    std::shared_ptr<Ope> found_ope;

private:
    const std::vector<std::shared_ptr<Ope>>& args_;
    const std::vector<std::string>& params_;
};

struct IsPrioritizedChoice : public Ope::Visitor
{
    IsPrioritizedChoice() : is_prioritized_choice_(false) {}

    void visit(PrioritizedChoice& /*ope*/) override {
        is_prioritized_choice_ = true;
    }

    static bool is_prioritized_choice(Ope& ope) {
        IsPrioritizedChoice vis;
        ope.accept(vis);
        return vis.is_prioritized_choice_;
    }

private:
    bool is_prioritized_choice_;
};

/*
 * Keywords
 */
static const char* WHITESPACE_DEFINITION_NAME = "%whitespace";
static const char* WORD_DEFINITION_NAME = "%word";

/*
 * Definition
 */
class Definition
{
public:
    struct Result {
        bool              ret;
        size_t            len;
        const char*       error_pos;
        const char*       message_pos;
        const std::string message;
    };

    Definition()
        : ignoreSemanticValue(false)
        , enablePackratParsing(false)
        , is_macro(false)
        , holder_(std::make_shared<Holder>(this))
        , is_token_(false) {}

    Definition(const Definition& rhs)
        : name(rhs.name)
        , ignoreSemanticValue(false)
        , enablePackratParsing(false)
        , is_macro(false)
        , holder_(rhs.holder_)
        , is_token_(false)
    {
        holder_->outer_ = this;
    }

    Definition(Definition&& rhs)
        : name(std::move(rhs.name))
        , ignoreSemanticValue(rhs.ignoreSemanticValue)
        , whitespaceOpe(rhs.whitespaceOpe)
        , wordOpe(rhs.wordOpe)
        , enablePackratParsing(rhs.enablePackratParsing)
        , is_macro(rhs.is_macro)
        , holder_(std::move(rhs.holder_))
        , is_token_(rhs.is_token_)
    {
        holder_->outer_ = this;
    }

    Definition(const std::shared_ptr<Ope>& ope)
        : ignoreSemanticValue(false)
        , enablePackratParsing(false)
        , is_macro(false)
        , holder_(std::make_shared<Holder>(this))
        , is_token_(false)
    {
        *this <= ope;
    }

    operator std::shared_ptr<Ope>() {
        return std::make_shared<WeakHolder>(holder_);
    }

    Definition& operator<=(const std::shared_ptr<Ope>& ope) {
        holder_->ope_ = ope;
        return *this;
    }

    Result parse(const char* s, size_t n, const char* path = nullptr) const {
        SemanticValues sv;
        any dt;
        return parse_core(s, n, sv, dt, path);
    }

    Result parse(const char* s, const char* path = nullptr) const {
        auto n = strlen(s);
        return parse(s, n, path);
    }

    Result parse(const char* s, size_t n, any& dt, const char* path = nullptr) const {
        SemanticValues sv;
        return parse_core(s, n, sv, dt, path);
    }

    Result parse(const char* s, any& dt, const char* path = nullptr) const {
        auto n = strlen(s);
        return parse(s, n, dt, path);
    }

    template <typename T>
    Result parse_and_get_value(const char* s, size_t n, T& val, const char* path = nullptr) const {
        SemanticValues sv;
        any dt;
        auto r = parse_core(s, n, sv, dt, path);
        if (r.ret && !sv.empty() && !sv.front().is_undefined()) {
            val = sv[0].get<T>();
        }
        return r;
    }

    template <typename T>
    Result parse_and_get_value(const char* s, T& val, const char* path = nullptr) const {
        auto n = strlen(s);
        return parse_and_get_value(s, n, val, path);
    }

    template <typename T>
    Result parse_and_get_value(const char* s, size_t n, any& dt, T& val, const char* path = nullptr) const {
        SemanticValues sv;
        auto r = parse_core(s, n, sv, dt, path);
        if (r.ret && !sv.empty() && !sv.front().is_undefined()) {
            val = sv[0].get<T>();
        }
        return r;
    }

    template <typename T>
    Result parse_and_get_value(const char* s, any& dt, T& val, const char* path = nullptr) const {
        auto n = strlen(s);
        return parse_and_get_value(s, n, dt, val, path);
    }

    Action operator=(Action a) {
        action = a;
        return a;
    }

    template <typename T>
    Definition& operator,(T fn) {
        operator=(fn);
        return *this;
    }

    Definition& operator~() {
        ignoreSemanticValue = true;
        return *this;
    }

    void accept(Ope::Visitor& v) {
        holder_->accept(v);
    }

    std::shared_ptr<Ope> get_core_operator() const {
        return holder_->ope_;
    }

    bool is_token() const {
        std::call_once(is_token_init_, [this]() {
            is_token_ = TokenChecker::is_token(*get_core_operator());
        });
        return is_token_;
    }

    std::string                                                                          name;
    size_t                                                                               id;
    Action                                                                               action;
    std::function<void (const char* s, size_t n, any& dt)>                               enter;
    std::function<void (const char* s, size_t n, size_t matchlen, any& value, any& dt)>  leave;
    std::function<std::string ()>                                                        error_message;
    bool                                                                                 ignoreSemanticValue;
    std::shared_ptr<Ope>                                                                 whitespaceOpe;
    std::shared_ptr<Ope>                                                                 wordOpe;
    bool                                                                                 enablePackratParsing;
    bool                                                                                 is_macro;
    std::vector<std::string>                                                             params;
    Tracer                                                                               tracer;

private:
    friend class Reference;

    Definition& operator=(const Definition& rhs);
    Definition& operator=(Definition&& rhs);

    Result parse_core(const char* s, size_t n, SemanticValues& sv, any& dt, const char* path) const {
        std::shared_ptr<Ope> ope = holder_;

        AssignIDToDefinition vis;
        holder_->accept(vis);

        if (whitespaceOpe) {
            ope = std::make_shared<Sequence>(whitespaceOpe, ope);
            whitespaceOpe->accept(vis);
        }

        if (wordOpe) {
            wordOpe->accept(vis);
        }

        Context cxt(path, s, n, vis.ids.size(), whitespaceOpe, wordOpe, enablePackratParsing, tracer);
        auto len = ope->parse(s, n, sv, cxt, dt);
        return Result{ success(len), len, cxt.error_pos, cxt.message_pos, cxt.message };
    }

    std::shared_ptr<Holder> holder_;
    mutable std::once_flag  is_token_init_;
    mutable bool            is_token_;
};

/*
 * Implementations
 */

inline size_t parse_literal(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt,
        const std::string& lit, bool& init_is_word, bool& is_word)
{
    size_t i = 0;
    for (; i < lit.size(); i++) {
        if (i >= n || s[i] != lit[i]) {
            c.set_error_pos(s);
            return static_cast<size_t>(-1);
        }
    }

    // Word check
    static Context dummy_c(nullptr, lit.data(), lit.size(), 0, nullptr, nullptr, false, nullptr);
    static SemanticValues dummy_sv;
    static any dummy_dt;

    if (!init_is_word) { // TODO: Protect with mutex
        if (c.wordOpe) {
            auto len = c.wordOpe->parse(lit.data(), lit.size(), dummy_sv, dummy_c, dummy_dt);
            is_word = success(len);
        }
        init_is_word = true;
    }

    if (is_word) {
        auto ope = std::make_shared<NotPredicate>(c.wordOpe);
        auto len = ope->parse(s + i, n - i, dummy_sv, dummy_c, dummy_dt);
        if (fail(len)) {
            return static_cast<size_t>(-1);
        }
        i += len;
    }

    // Skip whiltespace
    if (!c.in_token) {
        if (c.whitespaceOpe) {
            auto len = c.whitespaceOpe->parse(s + i, n - i, sv, c, dt);
            if (fail(len)) {
                return static_cast<size_t>(-1);
            }
            i += len;
        }
    }

    return i;
}

inline size_t LiteralString::parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const {
    c.trace("LiteralString", s, n, sv, dt);
    return parse_literal(s, n, sv, c, dt, lit_, init_is_word_, is_word_);
}

inline size_t TokenBoundary::parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const {
    c.in_token = true;
    auto se = make_scope_exit([&]() { c.in_token = false; });
    const auto& rule = *ope_;
    auto len = rule.parse(s, n, sv, c, dt);
    if (success(len)) {
        sv.tokens.push_back(std::make_pair(s, len));

        if (c.whitespaceOpe) {
            auto l = c.whitespaceOpe->parse(s + len, n - len, sv, c, dt);
            if (fail(l)) {
                return static_cast<size_t>(-1);
            }
            len += l;
        }
    }
    return len;
}

inline size_t Holder::parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const {
    if (!ope_) {
        throw std::logic_error("Uninitialized definition ope was used...");
    }

    c.trace(outer_->name.c_str(), s, n, sv, dt);
    c.nest_level++;
    auto se = make_scope_exit([&]() {
        c.nest_level--;
    });

    // Macro reference
    // TODO: need packrat support
    if (outer_->is_macro) {
        return ope_->parse(s, n, sv, c, dt);
    }

    size_t len;
    any    val;

    c.packrat(s, outer_->id, len, val, [&](any& a_val) {
        if (outer_->enter) {
            outer_->enter(s, n, dt);
        }

        auto se2 = make_scope_exit([&]() {
            c.pop();

            if (outer_->leave) {
                outer_->leave(s, n, len, a_val, dt);
            }
        });

        auto& chldsv = c.push();

        len = ope_->parse(s, n, chldsv, c, dt);

        // Invoke action
        if (success(len)) {
            chldsv.s_ = s;
            chldsv.n_ = len;
            chldsv.name_ = outer_->name;

            if (!IsPrioritizedChoice::is_prioritized_choice(*ope_)) {
                chldsv.choice_count_ = 0;
                chldsv.choice_ = 0;
            }

            try {
                a_val = reduce(chldsv, dt);
            } catch (const parse_error& e) {
                if (e.what()) {
                    if (c.message_pos < s) {
                        c.message_pos = s;
                        c.message = e.what();
                    }
                }
                len = static_cast<size_t>(-1);
            }
        }
    });

    if (success(len)) {
        if (!outer_->ignoreSemanticValue) {
            sv.emplace_back(val);
        }
    } else {
        if (outer_->error_message) {
            if (c.message_pos < s) {
                c.message_pos = s;
                c.message = outer_->error_message();
            }
        }
    }

    return len;
}

inline any Holder::reduce(SemanticValues& sv, any& dt) const {
    if (outer_->action) {
        return outer_->action(sv, dt);
    } else if (sv.empty()) {
        return any();
    } else {
        return std::move(sv.front());
    }
}

inline size_t Reference::parse(
    const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const {
    if (rule_) {
        // Reference rule
        if (rule_->is_macro) {
            // Macro
            FindReference vis(c.top_args(), rule_->params);

            // Collect arguments
            std::vector<std::shared_ptr<Ope>> args;
            for (auto arg: args_) {
                arg->accept(vis);
                args.push_back(vis.found_ope);
            }

            c.push_args(args);
            auto se = make_scope_exit([&]() { c.pop_args(); });
            auto ope = get_core_operator();
            return ope->parse(s, n, sv, c, dt);
        } else {
            // Definition
            auto ope = get_core_operator();
            return ope->parse(s, n, sv, c, dt);
        }
    } else {
        // Reference parameter in macro
        const auto& args = c.top_args();
        return args[iarg_]->parse(s, n, sv, c, dt);
    }
}

inline std::shared_ptr<Ope> Reference::get_core_operator() const {
    return rule_->holder_;
}

inline size_t BackReference::parse(const char* s, size_t n, SemanticValues& sv, Context& c, any& dt) const {
    c.trace("BackReference", s, n, sv, dt);
    auto it = c.capture_scope_stack.rbegin();
    while (it != c.capture_scope_stack.rend()) {
        const auto& captures = *it;
        if (captures.find(name_) != captures.end()) {
            const auto& lit = captures.at(name_);
            auto init_is_word = false;
            auto is_word = false;
            return parse_literal(s, n, sv, c, dt, lit, init_is_word, is_word);
        }
        ++it;
    }
    throw std::runtime_error("Invalid back reference...");
}

inline void Sequence::accept(Visitor& v) { v.visit(*this); }
inline void PrioritizedChoice::accept(Visitor& v) { v.visit(*this); }
inline void ZeroOrMore::accept(Visitor& v) { v.visit(*this); }
inline void OneOrMore::accept(Visitor& v) { v.visit(*this); }
inline void Option::accept(Visitor& v) { v.visit(*this); }
inline void AndPredicate::accept(Visitor& v) { v.visit(*this); }
inline void NotPredicate::accept(Visitor& v) { v.visit(*this); }
inline void LiteralString::accept(Visitor& v) { v.visit(*this); }
inline void CharacterClass::accept(Visitor& v) { v.visit(*this); }
inline void Character::accept(Visitor& v) { v.visit(*this); }
inline void AnyCharacter::accept(Visitor& v) { v.visit(*this); }
inline void CaptureScope::accept(Visitor& v) { v.visit(*this); }
inline void Capture::accept(Visitor& v) { v.visit(*this); }
inline void TokenBoundary::accept(Visitor& v) { v.visit(*this); }
inline void Ignore::accept(Visitor& v) { v.visit(*this); }
inline void User::accept(Visitor& v) { v.visit(*this); }
inline void WeakHolder::accept(Visitor& v) { v.visit(*this); }
inline void Holder::accept(Visitor& v) { v.visit(*this); }
inline void Reference::accept(Visitor& v) { v.visit(*this); }
inline void Whitespace::accept(Visitor& v) { v.visit(*this); }
inline void BackReference::accept(Visitor& v) { v.visit(*this); }

inline void AssignIDToDefinition::visit(Holder& ope) {
    auto p = static_cast<void*>(ope.outer_);
    if (ids.count(p)) {
        return;
    }
    auto id = ids.size();
    ids[p] = id;
    ope.outer_->id = id;
    ope.ope_->accept(*this);
}

inline void AssignIDToDefinition::visit(Reference& ope) {
    if (ope.rule_) {
        for (auto arg: ope.args_) {
            arg->accept(*this);
        }
        ope.rule_->accept(*this);
    }
}

inline void TokenChecker::visit(Reference& ope) {
    if (ope.is_macro_) {
        ope.rule_->accept(*this);
        for (auto arg: ope.args_) {
            arg->accept(*this);
        }
    } else {
        has_rule_ = true;
    }
}

inline void DetectLeftRecursion::visit(Reference& ope) {
    if (ope.name_ == name_) {
        error_s = ope.s_;
    } else if (!refs_.count(ope.name_)) {
        refs_.insert(ope.name_);
        if (ope.rule_) {
            ope.rule_->accept(*this);
            if (done_ == false) {
                return;
            }
        }
    }
    done_ = true;
}

inline void ReferenceChecker::visit(Reference& ope) {
    auto it = std::find(params_.begin(), params_.end(), ope.name_);
    if (it != params_.end()) {
        return;
    }

    if (!grammar_.count(ope.name_)) {
        error_s[ope.name_] = ope.s_;
        error_message[ope.name_] = "'" + ope.name_ + "' is not defined.";
    } else {
        const auto& rule = grammar_.at(ope.name_);
        if (rule.is_macro) {
            if (!ope.is_macro_ || ope.args_.size() != rule.params.size()) {
                error_s[ope.name_] = ope.s_;
                error_message[ope.name_] = "incorrect number of arguments.";
            }
        } else if (ope.is_macro_) {
            error_s[ope.name_] = ope.s_;
            error_message[ope.name_] = "'" + ope.name_ + "' is not macro.";
        }
    }
}

inline void LinkReferences::visit(Reference& ope) {
    if (grammar_.count(ope.name_)) {
        auto& rule = grammar_.at(ope.name_);
        ope.rule_ = &rule;
    } else {
        for (size_t i = 0; i < params_.size(); i++) {
            const auto& param = params_[i];
            if (param == ope.name_) {
                ope.iarg_ = i;
                break;
            }
        }
    }
    for (auto arg: ope.args_) {
        arg->accept(*this);
    }
}

inline void FindReference::visit(Reference& ope) {
    for (size_t i = 0; i < args_.size(); i++) {
        const auto& name = params_[i];
        if (name == ope.name_) {
            found_ope = args_[i];
            return;
        }
    }
    found_ope = ope.shared_from_this();
}

/*-----------------------------------------------------------------------------
 *  PEG parser generator
 *---------------------------------------------------------------------------*/

typedef std::unordered_map<std::string, std::shared_ptr<Ope>> Rules;
typedef std::function<void (size_t, size_t, const std::string&)> Log;

class ParserGenerator
{
public:
    static std::shared_ptr<Grammar> parse(
        const char*  s,
        size_t       n,
        const Rules& rules,
        std::string& start,
        Log          log)
    {
        return get_instance().perform_core(s, n, rules, start, log);
    }

     static std::shared_ptr<Grammar> parse(
        const char*  s,
        size_t       n,
        std::string& start,
        Log          log)
    {
        Rules dummy;
        return parse(s, n, dummy, start, log);
    }

    // For debuging purpose
    static Grammar& grammar() {
        return get_instance().g;
    }

private:
    static ParserGenerator& get_instance() {
        static ParserGenerator instance;
        return instance;
    }

    ParserGenerator() {
        make_grammar();
        setup_actions();
    }

    struct Data {
        std::shared_ptr<Grammar>                         grammar;
        std::string                                      start;
        std::vector<std::pair<std::string, const char*>> duplicates;

        Data(): grammar(std::make_shared<Grammar>()) {}
    };

    void make_grammar() {
        // Setup PEG syntax parser
        g["Grammar"]    <= seq(g["Spacing"], oom(g["Definition"]), g["EndOfFile"]);
        g["Definition"] <= cho(seq(g["Ignore"], g["IdentCont"], g["Parameters"], g["LEFTARROW"], g["Expression"]),
                               seq(g["Ignore"], g["Identifier"], g["LEFTARROW"], g["Expression"]));

        g["Expression"] <= seq(g["Sequence"], zom(seq(g["SLASH"], g["Sequence"])));
        g["Sequence"]   <= zom(g["Prefix"]);
        g["Prefix"]     <= seq(opt(cho(g["AND"], g["NOT"])), g["Suffix"]);
        g["Suffix"]     <= seq(g["Primary"], opt(cho(g["QUESTION"], g["STAR"], g["PLUS"])));
        g["Primary"]    <= cho(seq(g["Ignore"], g["IdentCont"], g["Arguments"], npd(g["LEFTARROW"])),
                               seq(g["Ignore"], g["Identifier"], npd(seq(opt(g["Parameters"]), g["LEFTARROW"]))),
                               seq(g["OPEN"], g["Expression"], g["CLOSE"]),
                               seq(g["BeginTok"], g["Expression"], g["EndTok"]),
                               seq(g["BeginCapScope"], g["Expression"], g["EndCapScope"]),
                               seq(g["BeginCap"], g["Expression"], g["EndCap"]),
                               g["BackRef"], g["Literal"], g["Class"], g["DOT"]);

        g["Identifier"] <= seq(g["IdentCont"], g["Spacing"]);
        g["IdentCont"]  <= seq(g["IdentStart"], zom(g["IdentRest"]));

        const static std::vector<std::pair<char32_t, char32_t>> range = {{ 0x0080, 0xFFFF }};
        g["IdentStart"] <= cho(cls("a-zA-Z_%"), cls(range));

        g["IdentRest"]  <= cho(g["IdentStart"], cls("0-9"));

        g["Literal"]    <= cho(seq(cls("'"), tok(zom(seq(npd(cls("'")), g["Char"]))), cls("'"), g["Spacing"]),
                               seq(cls("\""), tok(zom(seq(npd(cls("\"")), g["Char"]))), cls("\""), g["Spacing"]));

        g["Class"]      <= seq(chr('['), tok(zom(seq(npd(chr(']')), g["Range"]))), chr(']'), g["Spacing"]);

        g["Range"]      <= cho(seq(g["Char"], chr('-'), g["Char"]), g["Char"]);
        g["Char"]       <= cho(seq(chr('\\'), cls("nrt'\"[]\\")),
                               seq(chr('\\'), cls("0-3"), cls("0-7"), cls("0-7")),
                               seq(chr('\\'), cls("0-7"), opt(cls("0-7"))),
                               seq(lit("\\x"), cls("0-9a-fA-F"), opt(cls("0-9a-fA-F"))),
                               seq(lit("\\u"), cls("0-9a-fA-F"), cls("0-9a-fA-F"), cls("0-9a-fA-F"), cls("0-9a-fA-F")),
                               seq(npd(chr('\\')), dot()));

#if defined(PEGLIB_NO_UNICODE_CHARS)
        g["LEFTARROW"]  <= seq(lit("<-"), g["Spacing"]);
#else
        g["LEFTARROW"]  <= seq(cho(lit("<-"), lit(u8"←")), g["Spacing"]);
#endif
        ~g["SLASH"]     <= seq(chr('/'), g["Spacing"]);
        g["AND"]        <= seq(chr('&'), g["Spacing"]);
        g["NOT"]        <= seq(chr('!'), g["Spacing"]);
        g["QUESTION"]   <= seq(chr('?'), g["Spacing"]);
        g["STAR"]       <= seq(chr('*'), g["Spacing"]);
        g["PLUS"]       <= seq(chr('+'), g["Spacing"]);
        ~g["OPEN"]      <= seq(chr('('), g["Spacing"]);
        ~g["CLOSE"]     <= seq(chr(')'), g["Spacing"]);
        g["DOT"]        <= seq(chr('.'), g["Spacing"]);

        ~g["Spacing"]   <= zom(cho(g["Space"], g["Comment"]));
        g["Comment"]    <= seq(chr('#'), zom(seq(npd(g["EndOfLine"]), dot())), g["EndOfLine"]);
        g["Space"]      <= cho(chr(' '), chr('\t'), g["EndOfLine"]);
        g["EndOfLine"]  <= cho(lit("\r\n"), chr('\n'), chr('\r'));
        g["EndOfFile"]  <= npd(dot());

        ~g["BeginTok"]  <= seq(chr('<'), g["Spacing"]);
        ~g["EndTok"]    <= seq(chr('>'), g["Spacing"]);

        ~g["BeginCapScope"] <= seq(chr('$'), chr('('), g["Spacing"]);
        ~g["EndCapScope"]   <= seq(chr(')'), g["Spacing"]);

        g["BeginCap"]   <= seq(chr('$'), tok(g["IdentCont"]), chr('<'), g["Spacing"]);
        ~g["EndCap"]    <= seq(chr('>'), g["Spacing"]);

        g["BackRef"]    <= seq(chr('$'), tok(g["IdentCont"]), g["Spacing"]);

        g["IGNORE"]     <= chr('~');

        g["Ignore"]     <= opt(g["IGNORE"]);
        g["Parameters"] <= seq(g["OPEN"], g["Identifier"], zom(seq(g["COMMA"], g["Identifier"])), g["CLOSE"]);
        g["Arguments"]  <= seq(g["OPEN"], g["Expression"], zom(seq(g["COMMA"], g["Expression"])), g["CLOSE"]);
        ~g["COMMA"]     <= seq(chr(','), g["Spacing"]);


        // Set definition names
        for (auto& x: g) {
            x.second.name = x.first;
        }
    }

    void setup_actions() {
        g["Definition"] = [&](const SemanticValues& sv, any& dt) {
            auto is_macro = sv.choice() == 0;
            auto ignore = sv[0].get<bool>();
            auto name = sv[1].get<std::string>();

            std::vector<std::string> params;
            std::shared_ptr<Ope> ope;
            if (is_macro) {
                params = sv[2].get<std::vector<std::string>>();
                ope = sv[4].get<std::shared_ptr<Ope>>();
            } else {
                ope = sv[3].get<std::shared_ptr<Ope>>();
            }

            Data& data = *dt.get<Data*>();

            auto& grammar = *data.grammar;
            if (!grammar.count(name)) {
                auto& rule = grammar[name];
                rule <= ope;
                rule.name = name;
                rule.ignoreSemanticValue = ignore;
                rule.is_macro = is_macro;
                rule.params = params;

                if (data.start.empty()) {
                    data.start = name;
                }
            } else {
                data.duplicates.emplace_back(name, sv.c_str());
            }
        };

        g["Expression"] = [&](const SemanticValues& sv) {
            if (sv.size() == 1) {
                return sv[0].get<std::shared_ptr<Ope>>();
            } else {
                std::vector<std::shared_ptr<Ope>> opes;
                for (auto i = 0u; i < sv.size(); i++) {
                    opes.emplace_back(sv[i].get<std::shared_ptr<Ope>>());
                }
                const std::shared_ptr<Ope> ope = std::make_shared<PrioritizedChoice>(opes);
                return ope;
            }
        };

        g["Sequence"] = [&](const SemanticValues& sv) {
            if (sv.size() == 1) {
                return sv[0].get<std::shared_ptr<Ope>>();
            } else {
                std::vector<std::shared_ptr<Ope>> opes;
                for (const auto& x: sv) {
                    opes.emplace_back(x.get<std::shared_ptr<Ope>>());
                }
                const std::shared_ptr<Ope> ope = std::make_shared<Sequence>(opes);
                return ope;
            }
        };

        g["Prefix"] = [&](const SemanticValues& sv) {
            std::shared_ptr<Ope> ope;
            if (sv.size() == 1) {
                ope = sv[0].get<std::shared_ptr<Ope>>();
            } else {
                assert(sv.size() == 2);
                auto tok = sv[0].get<char>();
                ope = sv[1].get<std::shared_ptr<Ope>>();
                if (tok == '&') {
                    ope = apd(ope);
                } else { // '!'
                    ope = npd(ope);
                }
            }
            return ope;
        };

        g["Suffix"] = [&](const SemanticValues& sv) {
            auto ope = sv[0].get<std::shared_ptr<Ope>>();
            if (sv.size() == 1) {
                return ope;
            } else {
                assert(sv.size() == 2);
                auto tok = sv[1].get<char>();
                if (tok == '?') {
                    return opt(ope);
                } else if (tok == '*') {
                    return zom(ope);
                } else { // '+'
                    return oom(ope);
                }
            }
        };

        g["Primary"] = [&](const SemanticValues& sv, any& dt) -> std::shared_ptr<Ope> {
            Data& data = *dt.get<Data*>();

            switch (sv.choice()) {
                case 0:   // Macro Reference
                case 1: { // Reference
                    auto is_macro = sv.choice() == 0;
                    auto ignore = sv[0].get<bool>();
                    const auto& ident = sv[1].get<std::string>();

                    std::vector<std::shared_ptr<Ope>> args;
                    if (is_macro) {
                        args = sv[2].get<std::vector<std::shared_ptr<Ope>>>();
                    }

                    if (ignore) {
                        return ign(ref(*data.grammar, ident, sv.c_str(), is_macro, args));
                    } else {
                        return ref(*data.grammar, ident, sv.c_str(), is_macro, args);
                    }
                }
                case 2: { // (Expression)
                    return sv[0].get<std::shared_ptr<Ope>>();
                }
                case 3: { // TokenBoundary
                    return tok(sv[0].get<std::shared_ptr<Ope>>());
                }
                case 4: { // CaptureScope
                    return csc(sv[0].get<std::shared_ptr<Ope>>());
                }
                case 5: { // Capture
                    const auto& name = sv[0].get<std::string>();
                    auto ope = sv[1].get<std::shared_ptr<Ope>>();
                    return cap(ope, [name](const char* a_s, size_t a_n, Context& c) {
                        c.capture_scope_stack.back()[name] = std::string(a_s, a_n);
                    });
                }
                default: {
                    return sv[0].get<std::shared_ptr<Ope>>();
                }
            }
        };

        g["IdentCont"] = [](const SemanticValues& sv) {
            return std::string(sv.c_str(), sv.length());
        };

        g["IdentStart"] = [](const SemanticValues& /*sv*/) {
            return std::string();
        };

        g["IdentRest"] = [](const SemanticValues& /*sv*/) {
            return std::string();
        };

        g["Literal"] = [](const SemanticValues& sv) {
            const auto& tok = sv.tokens.front();
            return lit(resolve_escape_sequence(tok.first, tok.second));
        };
        g["Class"] = [](const SemanticValues& sv) {
            auto ranges = sv.transform<std::pair<char32_t, char32_t>>();
            return cls(ranges);
        };
        g["Range"] = [](const SemanticValues& sv) {
            switch (sv.choice()) {
                case 0: {
                    auto s1 = sv[0].get<std::string>();
                    auto s2 = sv[1].get<std::string>();
                    auto cp1 = decode_codepoint(s1.c_str(), s1.length());
                    auto cp2 = decode_codepoint(s2.c_str(), s2.length());
                    return std::make_pair(cp1, cp2);
                }
                case 1: {
                    auto s = sv[0].get<std::string>();
                    auto cp = decode_codepoint(s.c_str(), s.length());
                    return std::make_pair(cp, cp);
                }
            }
            return std::make_pair<char32_t, char32_t>(0, 0);
        };
        g["Char"] = [](const SemanticValues& sv) {
            return resolve_escape_sequence(sv.c_str(), sv.length());
        };

        g["AND"]      = [](const SemanticValues& sv) { return *sv.c_str(); };
        g["NOT"]      = [](const SemanticValues& sv) { return *sv.c_str(); };
        g["QUESTION"] = [](const SemanticValues& sv) { return *sv.c_str(); };
        g["STAR"]     = [](const SemanticValues& sv) { return *sv.c_str(); };
        g["PLUS"]     = [](const SemanticValues& sv) { return *sv.c_str(); };

        g["DOT"] = [](const SemanticValues& /*sv*/) { return dot(); };

        g["BeginCap"] = [](const SemanticValues& sv) { return sv.token(); };

        g["BackRef"] = [&](const SemanticValues& sv) {
            return bkr(sv.token());
        };

        g["Ignore"] = [](const SemanticValues& sv) { return sv.size() > 0; };

        g["Parameters"] = [](const SemanticValues& sv) {
            return sv.transform<std::string>();
        };

        g["Arguments"] = [](const SemanticValues& sv) {
            return sv.transform<std::shared_ptr<Ope>>();
        };
    }

    std::shared_ptr<Grammar> perform_core(
        const char*  s,
        size_t       n,
        const Rules& rules,
        std::string& start,
        Log          log)
    {
        Data data;
        any dt = &data;
        auto r = g["Grammar"].parse(s, n, dt);

        if (!r.ret) {
            if (log) {
                if (r.message_pos) {
                    auto line = line_info(s, r.message_pos);
                    log(line.first, line.second, r.message);
                } else {
                    auto line = line_info(s, r.error_pos);
                    log(line.first, line.second, "syntax error");
                }
            }
            return nullptr;
        }

        auto& grammar = *data.grammar;

        // User provided rules
        for (const auto& x: rules) {
            auto name = x.first;
            bool ignore = false;
            if (!name.empty() && name[0] == '~') {
                ignore = true;
                name.erase(0, 1);
            }
            if (!name.empty()) {
                auto& rule = grammar[name];
                rule <= x.second;
                rule.name = name;
                rule.ignoreSemanticValue = ignore;
            }
        }

        // Check duplicated definitions
        bool ret = data.duplicates.empty();

        for (const auto& x: data.duplicates) {
            if (log) {
                const auto& name = x.first;
                auto ptr = x.second;
                auto line = line_info(s, ptr);
                log(line.first, line.second, "'" + name + "' is already defined.");
            }
        }

        // Check missing definitions
        for (auto& x: grammar) {
            auto& rule = x.second;

            ReferenceChecker vis(*data.grammar, rule.params);
            rule.accept(vis);
            for (const auto& y: vis.error_s) {
                const auto& name = y.first;
                const auto ptr = y.second;
                if (log) {
                    auto line = line_info(s, ptr);
                    log(line.first, line.second, vis.error_message[name]);
                }
                ret = false;
            }
        }

        if (!ret) {
            return nullptr;
        }

        // Link references
        for (auto& x: grammar) {
            auto& rule = x.second;
            LinkReferences vis(*data.grammar, rule.params);
            rule.accept(vis);
        }

        // Check left recursion
        ret = true;

        for (auto& x: grammar) {
            const auto& name = x.first;
            auto& rule = x.second;

            DetectLeftRecursion vis(name);
            rule.accept(vis);
            if (vis.error_s) {
                if (log) {
                    auto line = line_info(s, vis.error_s);
                    log(line.first, line.second, "'" + name + "' is left recursive.");
                }
                ret = false;;
            }
        }

        if (!ret) {
            return nullptr;
        }

        // Set root definition
        start = data.start;

        // Automatic whitespace skipping
        if (grammar.count(WHITESPACE_DEFINITION_NAME)) {
            auto& rule = (*data.grammar)[start];
            rule.whitespaceOpe = wsp((*data.grammar)[WHITESPACE_DEFINITION_NAME].get_core_operator());
        }

        // Word expression
        if (grammar.count(WORD_DEFINITION_NAME)) {
            auto& rule = (*data.grammar)[start];
            rule.wordOpe = (*data.grammar)[WORD_DEFINITION_NAME].get_core_operator();
        }

        return data.grammar;
    }

    Grammar g;
};

/*-----------------------------------------------------------------------------
 *  AST
 *---------------------------------------------------------------------------*/

const int AstDefaultTag = -1;

#ifndef PEGLIB_NO_CONSTEXPR_SUPPORT
inline constexpr unsigned int str2tag(const char* str, int h = 0) {
    return !str[h] ? 5381 : (str2tag(str, h + 1) * 33) ^ static_cast<unsigned char>(str[h]);
}

namespace udl {
inline constexpr unsigned int operator "" _(const char* s, size_t) {
    return str2tag(s);
}
}
#endif

template <typename Annotation>
struct AstBase : public Annotation
{
    AstBase(const char* a_path, size_t a_line, size_t a_column,
            const char* a_name, size_t a_position, size_t a_length,
            size_t a_choice_count, size_t a_choice,
            const std::vector<std::shared_ptr<AstBase>>& a_nodes)
        : path(a_path ? a_path : "")
        , line(a_line)
        , column(a_column)
        , name(a_name)
        , position(a_position)
        , length(a_length)
        , choice_count(a_choice_count)
        , choice(a_choice)
        , original_name(a_name)
        , original_choice_count(a_choice_count)
        , original_choice(a_choice)
#ifndef PEGLIB_NO_CONSTEXPR_SUPPORT
        , tag(str2tag(a_name))
        , original_tag(tag)
#endif
        , is_token(false)
        , nodes(a_nodes)
    {}

    AstBase(const char* a_path, size_t a_line, size_t a_column,
            const char* a_name, size_t a_position, size_t a_length,
            size_t a_choice_count, size_t a_choice,
            const std::string& a_token)
        : path(a_path ? a_path : "")
        , line(a_line)
        , column(a_column)
        , name(a_name)
        , position(a_position)
        , length(a_length)
        , choice_count(a_choice_count)
        , choice(a_choice)
        , original_name(a_name)
        , original_choice_count(a_choice_count)
        , original_choice(a_choice)
#ifndef PEGLIB_NO_CONSTEXPR_SUPPORT
        , tag(str2tag(a_name))
        , original_tag(tag)
#endif
        , is_token(true)
        , token(a_token)
    {}

    AstBase(const AstBase& ast, const char* a_original_name,
            size_t a_position, size_t a_length,
            size_t a_original_choice_count, size_t a_original_choise)
        : path(ast.path)
        , line(ast.line)
        , column(ast.column)
        , name(ast.name)
        , position(a_position)
        , length(a_length)
        , choice_count(ast.choice_count)
        , choice(ast.choice)
        , original_name(a_original_name)
        , original_choice_count(a_original_choice_count)
        , original_choice(a_original_choise)
#ifndef PEGLIB_NO_CONSTEXPR_SUPPORT
        , tag(ast.tag)
        , original_tag(str2tag(a_original_name))
#endif
        , is_token(ast.is_token)
        , token(ast.token)
        , nodes(ast.nodes)
        , parent(ast.parent)
    {}

    const std::string                 path;
    const size_t                      line;
    const size_t                      column;

    const std::string                 name;
    size_t                            position;
    size_t                            length;
    const size_t                      choice_count;
    const size_t                      choice;
    const std::string                 original_name;
    const size_t                      original_choice_count;
    const size_t                      original_choice;
#ifndef PEGLIB_NO_CONSTEXPR_SUPPORT
    const unsigned int                tag;
    const unsigned int                original_tag;
#endif

    const bool                        is_token;
    const std::string                 token;

    std::vector<std::shared_ptr<AstBase<Annotation>>> nodes;
    std::weak_ptr<AstBase<Annotation>>                parent;
};

template <typename T>
void ast_to_s_core(
    const std::shared_ptr<T>& ptr,
    std::string& s,
    int level,
    std::function<std::string (const T& ast, int level)> fn) {

    const auto& ast = *ptr;
    for (auto i = 0; i < level; i++) {
        s += "  ";
    }
    auto name = ast.original_name;
    if (ast.original_choice_count > 0) {
        name += "/" + std::to_string(ast.original_choice);
    }
    if (ast.name != ast.original_name) {
        name += "[" + ast.name + "]";
    }
    if (ast.is_token) {
        s += "- " + name + " (" + ast.token + ")\n";
    } else {
        s += "+ " + name + "\n";
    }
    if (fn) {
      s += fn(ast, level + 1);
    }
    for (auto node : ast.nodes) {
        ast_to_s_core(node, s, level + 1, fn);
    }
}

template <typename T>
std::string ast_to_s(
    const std::shared_ptr<T>& ptr,
    std::function<std::string (const T& ast, int level)> fn = nullptr) {

    std::string s;
    ast_to_s_core(ptr, s, 0, fn);
    return s;
}

struct AstOptimizer
{
    AstOptimizer(bool optimize_nodes, const std::vector<std::string>& filters = {})
        : optimize_nodes_(optimize_nodes)
        , filters_(filters) {}

    template <typename T>
    std::shared_ptr<T> optimize(std::shared_ptr<T> original, std::shared_ptr<T> parent = nullptr) {

        auto found = std::find(filters_.begin(), filters_.end(), original->name) != filters_.end();
        bool opt = optimize_nodes_ ? !found : found;

        if (opt && original->nodes.size() == 1) {
            auto child = optimize(original->nodes[0], parent);
            return std::make_shared<T>(
                *child, original->name.c_str(), original->choice_count,
                original->position, original->length, original->choice);
        }

        auto ast = std::make_shared<T>(*original);
        ast->parent = parent;
        ast->nodes.clear();
        for (auto node : original->nodes) {
            auto child = optimize(node, ast);
            ast->nodes.push_back(child);
        }
        return ast;
    }

private:
    const bool                     optimize_nodes_;
    const std::vector<std::string> filters_;
};

struct EmptyType {};
typedef AstBase<EmptyType> Ast;

/*-----------------------------------------------------------------------------
 *  parser
 *---------------------------------------------------------------------------*/

class parser
{
public:
    parser() = default;

    parser(const char* s, size_t n, const Rules& rules) {
        load_grammar(s, n, rules);
    }

    parser(const char* s, const Rules& rules)
        : parser(s, strlen(s), rules) {}

    parser(const char* s, size_t n)
        : parser(s, n, Rules()) {}

    parser(const char* s)
        : parser(s, strlen(s), Rules()) {}

    operator bool() {
        return grammar_ != nullptr;
    }

    bool load_grammar(const char* s, size_t n, const Rules& rules) {
        grammar_ = ParserGenerator::parse(s, n, rules, start_, log);
        return grammar_ != nullptr;
    }

    bool load_grammar(const char* s, size_t n) {
        return load_grammar(s, n, Rules());
    }

    bool load_grammar(const char* s, const Rules& rules) {
        auto n = strlen(s);
        return load_grammar(s, n, rules);
    }

    bool load_grammar(const char* s) {
        auto n = strlen(s);
        return load_grammar(s, n);
    }

    bool parse_n(const char* s, size_t n, const char* path = nullptr) const {
        if (grammar_ != nullptr) {
            const auto& rule = (*grammar_)[start_];
            auto r = rule.parse(s, n, path);
            output_log(s, n, r);
            return r.ret && r.len == n;
        }
        return false;
    }

    bool parse(const char* s, const char* path = nullptr) const {
        auto n = strlen(s);
        return parse_n(s, n, path);
    }

    bool parse_n(const char* s, size_t n, any& dt, const char* path = nullptr) const {
        if (grammar_ != nullptr) {
            const auto& rule = (*grammar_)[start_];
            auto r = rule.parse(s, n, dt, path);
            output_log(s, n, r);
            return r.ret && r.len == n;
        }
        return false;
    }

    bool parse(const char* s, any& dt, const char* path = nullptr) const {
        auto n = strlen(s);
        return parse_n(s, n, dt, path);
    }

    template <typename T>
    bool parse_n(const char* s, size_t n, T& val, const char* path = nullptr) const {
        if (grammar_ != nullptr) {
            const auto& rule = (*grammar_)[start_];
            auto r = rule.parse_and_get_value(s, n, val, path);
            output_log(s, n, r);
            return r.ret && r.len == n;
        }
        return false;
    }

    template <typename T>
    bool parse(const char* s, T& val, const char* path = nullptr) const {
        auto n = strlen(s);
        return parse_n(s, n, val, path);
    }

    template <typename T>
    bool parse_n(const char* s, size_t n, any& dt, T& val, const char* path = nullptr) const {
        if (grammar_ != nullptr) {
            const auto& rule = (*grammar_)[start_];
            auto r = rule.parse_and_get_value(s, n, dt, val, path);
            output_log(s, n, r);
            return r.ret && r.len == n;
        }
        return false;
    }

    template <typename T>
    bool parse(const char* s, any& dt, T& val, const char* /*path*/ = nullptr) const {
        auto n = strlen(s);
        return parse_n(s, n, dt, val);
    }

    bool search(const char* s, size_t n, size_t& mpos, size_t& mlen) const {
        const auto& rule = (*grammar_)[start_];
        if (grammar_ != nullptr) {
            size_t pos = 0;
            while (pos < n) {
                size_t len = n - pos;
                auto r = rule.parse(s + pos, len);
                if (r.ret) {
                    mpos = pos;
                    mlen = len;
                    return true;
                }
                pos++;
            }
        }
        mpos = 0;
        mlen = 0;
        return false;
    }

    bool search(const char* s, size_t& mpos, size_t& mlen) const {
        auto n = strlen(s);
        return search(s, n, mpos, mlen);
    }

    Definition& operator[](const char* s) {
        return (*grammar_)[s];
    }

    const Definition& operator[](const char* s) const {
        return (*grammar_)[s];
    }

    std::vector<std::string> get_rule_names(){
        std::vector<std::string> rules;
        rules.reserve(grammar_->size());
        for (auto const& r : *grammar_) {
            rules.emplace_back(r.first);
        }
        return rules;
    }

    void enable_packrat_parsing() {
        if (grammar_ != nullptr) {
            auto& rule = (*grammar_)[start_];
            rule.enablePackratParsing = true;
        }
    }

    template <typename T = Ast>
    parser& enable_ast() {
        for (auto& x: *grammar_) {
            const auto& name = x.first;
            auto& rule = x.second;

            if (!rule.action) {
                rule.action = [&](const SemanticValues& sv) {
                    auto line = line_info(sv.ss, sv.c_str());

                    if (rule.is_token()) {
                        return std::make_shared<T>(
                            sv.path, line.first, line.second,
                            name.c_str(), std::distance(sv.ss, sv.c_str()), sv.length(), sv.choice_count(), sv.choice(),
                            sv.token());
                    }

                    auto ast = std::make_shared<T>(
                        sv.path, line.first, line.second,
                        name.c_str(), std::distance(sv.ss, sv.c_str()), sv.length(), sv.choice_count(), sv.choice(),
                        sv.transform<std::shared_ptr<T>>());

                    for (auto node: ast->nodes) {
                        node->parent = ast;
                    }
                    return ast;
                };
            }
        }
        return *this;
    }

    void enable_trace(Tracer tracer) {
        if (grammar_ != nullptr) {
            auto& rule = (*grammar_)[start_];
            rule.tracer = tracer;
        }
    }

    Log log;

private:
    void output_log(const char* s, size_t n, const Definition::Result& r) const {
        if (log) {
            if (!r.ret) {
                if (r.message_pos) {
                    auto line = line_info(s, r.message_pos);
                    log(line.first, line.second, r.message);
                } else {
                    auto line = line_info(s, r.error_pos);
                    log(line.first, line.second, "syntax error");
                }
            } else if (r.len != n) {
                auto line = line_info(s, s + r.len);
                log(line.first, line.second, "syntax error");
            }
        }
    }

    std::shared_ptr<Grammar> grammar_;
    std::string              start_;
};

} // namespace peg

#endif

// vim: et ts=4 sw=4 cin cino={1s ff=unix
