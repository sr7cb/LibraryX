#include <iostream>
#include <vector>
#include "domainx.hpp"
#include "interface.hpp"

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
}config;

#include "WarpXObj.hpp"

int main() {
    int n = 80;
    int m = 80;
    int k = 80;

    std::vector<int> sizes{n,m,k};
    NDTensor input(11, sizes);
    NDTensor output(6, sizes);
    NDTensor symbol(11, sizes);
    input.buildTensor();
    symbol.buildTensor();

    WarpXconfig conf;
    std::vector<void*> args{(void*)&input, (void*)&output, (void*)&symbol, (void*)&conf};
    WarpXProblem warpX(args, sizes, "warpX");

    warpX.step();

    // std::cout <<  warpX.residual() << std::endl; 
    
    return 0;
}