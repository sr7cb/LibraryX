#include <vector>
#include <complex>
#include "LibraryXProblem.hpp"

#pragma once

class IMDDFTProblem : public LibraryXProblem{
public:
    IMDDFTProblem(std::vector<double*>& args,
                    std::vector<int>& sizes) : 
                    args(args), sizes(sizes){
        semantics();             
    }

private:
    std::vector<double*> args;
    std::vector<int> sizes;

    void semantics() {

    }
};