#include <vector>
#include "LibraryXProblem.hpp"

#pragma once

class MDDFTProblem : public LibraryXProblem{
public:
    MDDFTProblem(const std::vector<int>& sizes) {
        // Initialize any necessary data or objects for IMDDFT
    }

    //~MDDFTProblem();

    void semantics(double* input, std::vector<std::complex<double> > output) {
        // Implement IMDDFT logic here
    } //different from libraryXProblem semantics? overrriding? or need this?
};