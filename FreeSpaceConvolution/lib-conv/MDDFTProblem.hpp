#include <vector>
#include "LibraryXProblem.hpp"

#pragma once

class MDDFTProblem : public LibraryXProblem{
public:
    MDDFTProblem(std::initializer_list<double*> args,
                  std::vector<int> sizes) : 
                  args(args), sizes(sizes) {
        semantics();               
    }

private:
    std::initializer_list<double*> args;
    std::vector<int> sizes;

    void semantics() {
        for (auto it = args.begin(); it != args.end(); ++it) {
            if (*it) {  // Ensure the pointer is not null
                std::cout << (*it)[0] << " "<<(*it)[1]<< std::endl;  // Double dereference to get the value
            }
        }

    }
};