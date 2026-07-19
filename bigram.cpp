#include "makemore.hpp"
#include <iostream>
#include <vector>

int main() {
  std::vector<std::string> names = load_names("data/names.txt");

  /*
    std::cout << "count: " << names.size() << '\n';
    std::cout << "first: " << names[0] << '\n';

    for (char c : names[0])
      std::cout << stoi_(c) << " ";
    std::cout << "\n";

    for (char c : names[0])
      std::cout << itos_(stoi_(c));
    std::cout << "\n";

    std::cout << stoi_('.') << " " << stoi_('a') << " " << stoi_('z') << "\n";
  */

  std::vector<std::vector<i32>> N(27, std::vector<i32>(27, 0));
  for (const auto &name : names) {
    std::string s = '.' + name + '.';

    for (size_t i = 1; i < s.size(); ++i) {
      char prev = s[i - 1];
      char curr = s[i];

      N[stoi_(prev)][stoi_(curr)]++;
    }
  }

  std::cout << "N[.][e] = " << N[0][5] << "  (expect 1531)\n";
  std::cout << "N[a][.] = " << N[stoi_('a')][0] << "  (expect 6640)\n";
  std::cout << "N[n][.] = " << N[(stoi_('n'))][0] << "  (expect 6763)\n";

  i64 total = 0;

  for (const auto &v : N)
    for (i32 c : v)
      total += c;

  std::cout << "total   = " << total << "  (expect 228146)\n";

  return 0;
}
