#include <vector>
#include <complex>
#include <map>
#include <iostream>
#include "LibraryXProblem.hpp"

#pragma once

class PointwiseMultiplier : public LibraryXProblem{
public:
    PointwiseMultiplier() {
        // Initialization for this constructor
    }

    //~PointwiseMultiplier();

    void semantics(std::vector<std::complex<double> >& arr1, double *arr2) {
        //std::map(arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), multiply()); // writes to arr1
    }
    double multiply (int x, int y){
        return x+y;
    }
};