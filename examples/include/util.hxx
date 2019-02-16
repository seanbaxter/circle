#pragma once

template<template<typename...> class temp, typename type_t>
struct is_spec_t {
  enum { value = false };
};

template<template<typename...> class temp, typename... types_t>
struct is_spec_t<temp, temp<types_t...> > {
  enum { value = true };
};