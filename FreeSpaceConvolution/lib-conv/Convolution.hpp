#include <vector>
#include "MDDFTProblem.hpp"
#include "IMDDFTProblem.hpp"
#include "PointwiseMultiplier.hpp"

#pragma once

class Convolution : public LibraryXProblem {
public:
    Convolution(std::vector<double*>& args,
                  std::vector<int>& sizes) : 
                  args(args), sizes(sizes) {
        semantics();               
    }

private:
    std::vector<double*> args;
    std::vector<int> sizes;

    void semantics (){                                  
        std::vector<std::complex<double> > temp1;
        MDDFTProblem mddftProblem(args, sizes);
        IMDDFTProblem imddftProblem(args, sizes);
        PointwiseMultiplier pointwiseMultiplier(args, sizes);
    }
};


