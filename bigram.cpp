#include "makemore.hpp"
#include <iostream>
#include <random>
#include <vector>

std::mt19937 gen(std::random_device{}());

i32 sample_row(const std::vector<f32> &row, std::mt19937 &gen) {
  std::uniform_real_distribution<f32> dist(0.0, 1.0);
  f32 r = dist(gen);

  f32 cumulative = 0.0;

  for (size_t j = 0; j < row.size(); ++j) {
    cumulative += row[j];

    if (cumulative > r)
      return j;
  }

  return 26;
}

void print_n_random_names(i32 num, std::vector<std::vector<f32>> &P) {

  for (i32 n = 0; n < num; ++n) {
    std::string name;
    i32 cur = 0;

    while (true) {
      i32 next = sample_row(P[cur], gen);

      if (next == 0)
        break;

      name += itos_(next);
      cur = next;
    }

    std::cout << name << '\n';
  }
}

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

  /*
    std::cout << "N[.][e] = " << N[0][5] << "  (expect 1531)\n";
    std::cout << "N[a][.] = " << N[stoi_('a')][0] << "  (expect 6640)\n";
    std::cout << "N[n][.] = " << N[(stoi_('n'))][0] << "  (expect 6763)\n";

    i64 total = 0;

    for (const auto &v : N)
      for (i32 c : v)
        total += c;

    std::cout << "total   = " << total << "  (expect 228146)\n";
  */

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

  /*
  for (const auto &v1 : P) {
    char ch = 'a' - 1;
    for (const auto &v2 : v1) {
      std::cout << ch << " " << v2 << ", ";
      ch++;
    }

    std::cout << '\n';
  }

  */
  /*
  double row0 = 0.0, row1 = 0.0;
  for (int j = 0; j < 27; ++j)
    row0 += P[0][j];
  for (int j = 0; j < 27; ++j)
    row1 += P[1][j];

  std::cout << "row 0 sum = " << row0 << "  (expect ~1.0)\n";
  std::cout << "row 1 sum = " << row1 << "  (expect ~1.0)\n";
  std::cout << "P[.][e]   = " << P[0][5] << "  (expect ~0.0478)\n";
  std::cout << "P[a][.]   = " << P[1][0] << "  (expect ~0.1961)\n";
*/

  print_n_random_names(50, P);

  f64 accumulate = 0.0;
  i64 count = 0;

  for (const auto &name : names) {
    std::string s = '.' + name + '.';

    for (size_t i = 1; i < s.size(); ++i) {
      char prev = s[i - 1];
      char curr = s[i];

      accumulate += -log(P[stoi_(prev)][stoi_(curr)]);
      ++count;
    }
  }

  std::cout << "avg NLL = " << accumulate / count << "  (expect ~2.45)\n";

  return 0;
}
