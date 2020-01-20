#include <vector>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

int main() {
  using std::cout;

  cout<< "set:           ";
  std::set<int> si { 1, 2, 3, 4, 5, 6 };
  cout<< si[:]<< ' ' ...;

  cout<< "\nmap:           ";
  std::map<std::string, int> msi {{"one", 1}, {"two", 2}, {"three", 3}};
  cout<< msi[:].first<< ':'<< msi[:].second<< ' ' ...;

  cout<< "\nunordered map: ";
  std::unordered_map<std::string, int> umsi {{"one", 1}, {"two", 2}, {"three", 3}};
  cout<< umsi[:].first<< ':'<< umsi[:].second<< ' ' ...;

  cout<< "\nunordered set: ";
  std::unordered_set<int> usi { 1, 2, 3, 4, 5, 6 };
  cout<< usi[:]<< ' ' ...;
  cout<< '\n';
}