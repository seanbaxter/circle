#include <iostream>

struct Player { std::string name; int hitpoints; int coins; };

void get_hint(const Player& p) {
  @match(p) {
    [.hitpoints: 1] => std::cout << "You're almost destroyed. Give up!\n";
    [.hitpoints: 10, .coins: 10] => std::cout << "I need the hints from you!\n";
    [.coins: 10] => std::cout << "Get more hitpoints!\n";
    [.hitpoints: 10] => std::cout << "Get more ammo!\n";
    [.name: _n] => {
      if (_n != "The Bruce Dickenson") {
        std::cout << "Get more hitpoints and ammo!\n";
      } else {
        std::cout << "More cowbell!\n";
      }
    }
  };
}

int main() {
  get_hint(Player { "Batman", 10, 15 });
  get_hint(Player { "Spider-man", 5, 10 });
  get_hint(Player { "Aquaman", 10, 10 });
  get_hint(Player { "Iron Man", 5, 3 });
   
  return 0;

}