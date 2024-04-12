#include <iostream>
#include <vector>
#include <complex>
#include "lib-conv/LibraryXProblem.hpp"
#include "lib-conv/Convolution.hpp"


void buildArray(double *input){ 
  for(int i=0; i<32; i++){
    
  }
  return;
}

int main() {
  std::vector<int> sizes{32, 32, 32};
  double *input = new double [32];
  double *output = new double [32];

  std::vector<std::complex<double> > symbol(32);

  buildArray(input);
  buildArray(output);
  buildArray((double *)symbol.data());

  std::vector<double*> args{output, input, (double *)symbol.data()};
  
  Convolution* convolutionProblem = new Convolution(args, sizes);
  
  for (int i=0; i<32; i++) {
    std::cout << output[i] << std::endl;
  }

  return 0;
}