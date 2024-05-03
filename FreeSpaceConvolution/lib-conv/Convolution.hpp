#include <vector>
#include "MDDFTProblem.hpp"
#include "IMDDFTProblem.hpp"
#include "PointwiseMultiplier.hpp"

#pragma once

class Convolution : public LibraryXProblem {
public:
    // Convolution(std::vector<double*>& args,
    //               std::vector<int>& sizes) : 
    //               args(args), sizes(sizes) {
    //     semantics();               
    // }

    Convolution(std::initializer_list<void*> args,
                  std::vector<int> sizes) : 
                  args(args), sizes(sizes) {
        semantics();               
    }

private:
    // std::vector<double*> args;
    std::initializer_list<void*> args;
    std::vector<int> sizes;

    void semantics (){                                  
        std::vector<std::complex<double> > temp1;
        for (auto it = args.begin(); it != args.end(); ++it) {
            if (*it) {  // Ensure the pointer is not null
                std::cout << ((double*)(*it))[0] << " "<<((double*)(*it))[1]<< std::endl;  // Double dereference to get the value
            }
            else{
                std::cout<<"no"<<std::endl;
            }
        }

        // MDDFTProblem mddftProblem({(*args[1]), temp}, sizes);
        // IMDDFTProblem imddftProblem(args, sizes);
        // PointwiseMultiplier pointwiseMultiplier(args, sizes);
        //std::cout<<args[0][1]<<std::endl;
    }
};


