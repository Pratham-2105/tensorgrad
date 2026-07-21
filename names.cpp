#include "makemore.hpp"
#include <iostream>

int main() {
  auto names = load_names("data/names.txt");
  Dataset ds = build_dataset(names, 3);

  std::cout << "examples: " << ds.X.size() << " (expect 228146)\n";

  for (int k = 0; k < 5; ++k) {
    std::cout << "[" << ds.X[k][0] << "," << ds.X[k][1] << "," << ds.X[k][2]
              << "] -> " << ds.Y[k] << "\n";
  }

  return 0;
}
