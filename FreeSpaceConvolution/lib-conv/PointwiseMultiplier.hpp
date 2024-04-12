#include <vector>
#include <complex>
#include <map>
#include <iostream>
#include "LibraryXProblem.hpp"

#pragma once

class PointwiseMultiplier : public LibraryXProblem{
public:
    PointwiseMultiplier(std::vector<double*>& args,
                    std::vector<int>& sizes) : 
                    args(args), sizes(sizes){
        semantics();                
    }

private:
    std::vector<double*> args;
    std::vector<int> sizes;

    void semantics() {
    }
    double multiply (int x, int y){
        return x+y;
    }
};