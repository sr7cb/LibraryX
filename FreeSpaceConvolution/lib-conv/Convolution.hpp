#include <vector>
#include "MDDFTProblem.hpp"
#include "IMDDFTProblem.hpp"
#include "PointwiseMultiplier.hpp"

#pragma once

class Convolution : public LibraryXProblem {
public:
    double* output;
    
    Convolution(const std::vector<int>& sizes){}

    Convolution(std::vector<double*> args,
                    const std::vector<int>& sizes) {
            semantics(args[0], args[1], args[2], sizes); //args[1]=input1                        
        }
    
    //~Convolution();
    void semantics(double* output,
                                double* input1, 
                                double* input2,
                                const std::vector<int>& sizes) {
        std::vector<std::complex<double> > temp1;
        MDDFTProblem mddftProblem(sizes);
        IMDDFTProblem imddftProblem(sizes);
        PointwiseMultiplier pointwiseMultiplier;
        mddftProblem.semantics(input1, temp1);
        pointwiseMultiplier.semantics(temp1, input2); //expects complex arguments? override semantics function or not>?
        imddftProblem.semantics(temp1, output); //Output not defined
        //removing these semantics lines here
    }
};


