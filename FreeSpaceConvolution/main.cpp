#include <iostream>
#include <vector>
#include "lib-conv/Convolution.hpp"


void buildArray(double *input);

int main() {
  std::vector<int> sizes{32, 32, 32};
  double *input = new double [32];
  double *output = new double [32];
  double *symbol = new double [32];
  //std::vector<double> symbol(sizes[0] * sizes[1] * sizes[2], 2.0); // Initialize symbol
  buildArray(input);
  //buildArray(output);
  buildArray(symbol);

  std::vector<double*> args{output, input, symbol};
  Convolution convolutionProblem(args, sizes);
  

  for (double value : output) {
    std::cout << value << std::endl;
  }

  return 0;
}