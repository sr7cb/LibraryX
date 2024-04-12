#include <vector>

#pragma once

class LibraryXProblem {

public:
LibraryXProblem() {}
    LibraryXProblem (const std::vector<int>& sizes) {}

    virtual void semantics() = 0;
                 
};

