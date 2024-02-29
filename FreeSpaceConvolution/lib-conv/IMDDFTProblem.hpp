#include <vector>
#include <complex>
#include "LibraryXProblem.hpp"

#pragma once

class IMDDFTProblem : public LibraryXProblem{
public:
    IMDDFTProblem(const std::vector<int>& sizes) {
        // Initialize any necessary data or objects for IMDDFT
    }

    //~IMDDFTProblem();

    void semantics(std::vector<std::complex<double> > input, double* output) {
        // Implement IMDDFT logic here
    }
};