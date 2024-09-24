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
  
  
  input[0]=1.2;
  input[1]=4.2;
  output[0]=2.2;
  output[1]=3.2;
  
  //Convolution convolutionProblem1({input}, sizes);
  Convolution convolutionProblem2({input, output}, sizes);

  return 0;
}