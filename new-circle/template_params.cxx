#pragma feature interface
#include <iostream>
#include <concepts>
#include <vector>

template<
                auto      nontype,
                typename  type,
  template<...> typename  type_template,
  template<...> auto      var_template,
  template<...> concept   concept_,
                interface interface_,
  template<...> interface interface_template, 
                namespace namespace_,
  template      auto      universal
> void f() {
  std::cout<< "nontype            = {}\n".format(nontype~string);
  std::cout<< "type               = {}\n".format(type~string);
  std::cout<< "type_template      = {}\n".format(type_template~string);
  std::cout<< "var_template       = {}\n".format(var_template~string);
  std::cout<< "concept            = {}\n".format(concept_~string);
  std::cout<< "interface          = {}\n".format(interface_~string);
  std::cout<< "interface_template = {}\n".format(interface_template~string);
  std::cout<< "namespace          = {}\n".format(namespace_~string);
  std::cout<< "universal          = {}\n".format(universal~string);
}

interface IPrint { };

template<interface IBase>
interface IClone : IBase { };

int main() {
  f<
    5,                 // non-type
    char[3],           // type
    std::basic_string, // type template
    std::is_signed_v,  // variable template
    std::integral,     // concept
    IPrint,            // interface
    IClone,            // interface template
    std,               // namespace
    void               // universal
  >();
}