#feature on edition_carbon_2023
#include <string_view>
#include <iostream>

using str = std::string_view;

interface IAnimal {
  fn static make_new(name: str) -> Self;

  fn name() const -> str;
  fn noise() const -> str;

  // Traits can provide default method definitions.
  fn talk() const {
    std::cout<< name()<< " says "<< noise()<< " \n";
  }
}

struct Sheep { 
  var naked : bool;
  var name : str;

  fn is_naked() const noexcept -> bool { return naked; }
  fn shear() noexcept { 
    if(naked) {
      std::cout<< name<< " is already naked...\n";
    } else {
      std::cout<< name<< " gets a haircut!\n";
      naked = true; 
    }
  }
}

impl Sheep : IAnimal {
  fn static make_new(name: str) -> Sheep {
    return { .naked=false, .name=name };
  }

  fn name() const -> str {
    return self.name;
  }

  fn noise() const -> str {
    return self.naked ?
      "baaaaah?" :
      "baaaaah!";
  }

  // Default trait methods can be overriden.
  fn talk() const {
    std::cout<< name()<< " pauses briefly... "<< noise()<< "\n";
  }
}

fn main() -> int {
  // Create a Sheep instance through IAnimal's static method.
  var dolly := impl!<Sheep, IAnimal>::make_new("Dolly");

  // Put the Sheep : IAnimal impl in scope so member lookup works.
  using impl Sheep : IAnimal;

  // Call a mix of interface methods and class member functions.
  dolly.talk();
  dolly.shear();
  dolly.talk();
}