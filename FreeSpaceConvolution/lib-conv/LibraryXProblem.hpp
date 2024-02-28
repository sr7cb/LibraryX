#include <vector>

#pragma once

class LibraryXProblem {

public:
LibraryXProblem() {}
    LibraryXProblem (const std::vector<int>& sizes) {    }
//what is the use of this constructor? 
    LibraryXProblem (const std::vector<double>& input1,
                    const std::vector<double>& input2, 
                    const std::vector<int>& sizes) {
        semantics (input1, input2, sizes);
   }

    ~LibraryXProblem();

    void semantics(const std::vector<double>& input1,
                    const std::vector<double>& input2, 
                    const std::vector<int>& sizes){}
};

