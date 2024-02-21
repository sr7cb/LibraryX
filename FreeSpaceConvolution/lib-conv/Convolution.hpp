#include <vector>
#include "LibraryXProblem.hpp"
#include "MDDFTProblem.hpp"
#include "IMDDFTProblem.hpp"
#include "PointwiseMultiplier.hpp"


class Convolution : public LibraryXProblem {
public:
    MDDFTProblem mddftProblem;
    IMDDFTProblem imddftProblem;
    PointwiseMultiplier pointwiseMultiplier;
    Convolution(const std::vector<int>& sizes) :
        mddftProblem(sizes),
        imddftProblem(sizes),
        pointwiseMultiplier() {}
    
    ~Convolution();

    std::vector<double> semantics(const std::vector<double>& input1,
                                  const std::vector<double>& input2, 
                                  const std::vector<int>& sizes) {
        double *temp1;
        mddftProblem.semantics(input1, temp1);
        pointwiseMultiplier.semantics(temp1, input2, sizes);
        imddftProblem.semantics(temp1, output);
    }


};


