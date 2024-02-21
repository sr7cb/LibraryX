#include <vector>
#include "MDDFTProblem.hpp"
#include "IMDDFTProblem.hpp"
#include "PointwiseMultiplier.hpp"

class LibraryXProblem {
public:

    LibraryXProblem(const std::vector<int>& sizes) :
        convolutionProblem(sizes) {}
    
    ~LibraryXProblem();

    std::vector<double> libraryXSpace(const std::vector<double>& input1,
                                  const std::vector<double>& input2, 
                                  const std::vector<int>& sizes) {
        convolutionProblem.semantics(input1, input2, sizes);
    }
    // private:
    // Convolution convolutionProblem;
};

