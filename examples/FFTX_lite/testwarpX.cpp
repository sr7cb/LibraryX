#include <iostream>
#include <vector>
#include "domainx.hpp"
#include <complex>

struct WarpXconfig{
    int n = 80;
    int np = n+1;
    int inFields = 11;
    int outFields = 6;
    int nf = n+2;
    int xdim = nf/2;
    int ydim = n;
    int zdim = n;
    double cvar = 10;
    double ep0var = 10;
    double c2 = cvar*cvar;
    double invep0 = 1/ep0var; 
    NDTensor<std::complex<double>> symbol;
};

#include "interface.hpp"
#include "WarpXObj.hpp"
#include "libraryx.hpp"

NDTensor<std::complex<double>> warpXBuildSymbol(WarpXconfig w) {
  return NDTensor<std::complex<double>>(8, {w.nf,w.n,w.n});
}

int main() {
  
    WarpXconfig conf;

    NDTensor<double> input(11, {conf.n,conf.n,conf.n});
    NDTensor<double> output(6, {conf.n,conf.n,conf.n});
   
    input.fillRandom();//ingest(argv[1]);
    conf.symbol = warpXBuildSymbol(conf);//argv[2]

    WarpXProblem warpX(conf);

    warpX.step(output,input);

    std::cout << "WarpX accuracy: " << warpX.residual() << std::endl; 
    
    return 0;
}




