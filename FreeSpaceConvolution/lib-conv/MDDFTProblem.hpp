#include <vector>
#include "LibraryXProblem.hpp"

#pragma once

class MDDFTProblem : public LibraryXProblem{
public:
    MDDFTProblem(std::vector<double*>& args,
                 std::vector<int>& sizes) : 
                 args(args), sizes(sizes) {
        semantics();                
    }

private:
    std::vector<double*> args;
    std::vector<int> sizes;

    void semantics() {

    }
};