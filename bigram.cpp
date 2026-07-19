#include "makemore.hpp"
#include <iostream>
#include <vector>

int main() {
  std::vector<std::string> names = load_names("data/names.txt");

  std::cout << "count: " << names.size() << '\n';
  std::cout << "first: " << names[0] << '\n';

  for (char c : names[0])
    std::cout << stoi_(c) << " ";
  std::cout << "\n";

  for (char c : names[0])
    std::cout << itos_(stoi_(c));
  std::cout << "\n";

  std::cout << stoi_('.') << " " << stoi_('a') << " " << stoi_('z') << "\n";

  return 0;
}
