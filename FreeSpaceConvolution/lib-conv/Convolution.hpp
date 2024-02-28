#include <vector>
#include "MDDFTProblem.hpp"
#include "IMDDFTProblem.hpp"
#include "PointwiseMultiplier.hpp"

#pragma once

class Convolution : public LibraryXProblem {
public:
    double* output;
    MDDFTProblem mddftProblem;
    IMDDFTProblem imddftProblem;
    PointwiseMultiplier pointwiseMultiplier;
    Convolution(const std::vector<int>& sizes) :
        mddftProblem(sizes),
        imddftProblem(sizes),
        pointwiseMultiplier() {}

    Convolution(const std::vector<double*> args,
                    const std::vector<int>& sizes) : 
                    mddftProblem(sizes),
                    imddftProblem(sizes),
                    pointwiseMultiplier() {
            double *temp1;
            output = args[0];
            std::vector<double> input2;
            semantics(args[1], input2, sizes); //args[1]=input1                        
        }
    
    ~Convolution();
    void semantics(const std::vector<double>& input1,
                                const std::vector<double>& input2, 
                                const std::vector<int>& sizes) {
        double *temp1;
        mddftProblem.semantics(input1, temp1);
        pointwiseMultiplier.semantics(temp1, input2); //expects complex arguments? override semantics function or not>?
        imddftProblem.semantics(temp1, output); //Output not defined
    }
};


