#include <vector>
#include "MDDFTProblem.hpp"
#include "IMDDFTProblem.hpp"
#include "PointwiseMultiplier.hpp"

#pragma once

class Convolution : public LibraryXProblem {
public:
    Convolution(std::initializer_list<void*> args,
                  std::vector<int> sizes) : 
                  args(args), sizes(sizes) {
        semantics();               
    }

private:
    std::initializer_list<void*> args;
    std::vector<int> sizes;

    void semantics (){                                  
        std::vector<std::complex<double*> > temp1;
        for (auto it = args.begin(); it != args.end(); ++it) {
            if (*it) {  // Ensure the pointer is not null
                std::cout << ((double*)(*it))[0] << " "<<((double*)(*it))[1]<< " In Convolution.hpp" <<std::endl;  // Double dereference to get the value
            }
            else{
                std::cout<<"no"<<std::endl; //replace this with how to handle null pointer if found?
            }
        }

        double *input = static_cast<double*>(*args.begin());
        double *output = static_cast<double*>(*std::next(args.begin(), 1));

        MDDFTProblem mddftProblem({input}, sizes);
        IMDDFTProblem imddftProblem({input, output}, sizes);
        PointwiseMultiplier pointwiseMultiplier({input, (double*)temp1.data(), output}, sizes);
    }
};


