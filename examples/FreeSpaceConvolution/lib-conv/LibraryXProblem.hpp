#include <vector>

#pragma once

class LibraryXProblem {

public:
LibraryXProblem() {}

LibraryXProblem(std::initializer_list<void*>& args,
                  std::vector<int>& sizes){}

    virtual void semantics() = 0;
                 
};

