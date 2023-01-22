#pragma feature interface
#include <iostream>

template<...> struct list;

template<template auto... Ts>
using sort_by_name = list<Ts~sort(_1~string < _2~string)...>;

template<typename T>
concept my_concept = true;

interface my_interface;

template<typename...>
interface my_interface_template;

int main() {
  using MyList = sort_by_name<
    "foo",                   // A string constant 
    int,                     // Type
    list,                    // Type template
    std::is_same_v,          // Variable template
    my_concept,              // Concept
    my_interface,            // Interface
    my_interface_template,   // Interface template
    std                      // Namespace
  >;

  std::cout<< MyList~universal_args~string + "\n" ...;
}
