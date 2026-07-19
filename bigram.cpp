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

  std::vector<std::vector<f32>> P(27, std::vector<f32>(27, 0.0));

  for (size_t i = 0; i < N.size(); ++i) {
    i32 sum = 0;

    for (size_t j = 0; j < N[i].size(); ++j) {
      sum += N[i][j];
    }

    if (sum == 0)
      continue;

    for (size_t j = 0; j < N[i].size(); ++j) {
      P[i][j] = static_cast<float>(N[i][j]) / static_cast<float>(sum);
    }
  }

  double row0 = 0.0, row1 = 0.0;
  for (int j = 0; j < 27; ++j)
    row0 += P[0][j];
  for (int j = 0; j < 27; ++j)
    row1 += P[1][j];

  std::cout << "row 0 sum = " << row0 << "  (expect ~1.0)\n";
  std::cout << "row 1 sum = " << row1 << "  (expect ~1.0)\n";
  std::cout << "P[.][e]   = " << P[0][5] << "  (expect ~0.0478)\n";
  std::cout << "P[a][.]   = " << P[1][0] << "  (expect ~0.1961)\n";

  return 0;
}
