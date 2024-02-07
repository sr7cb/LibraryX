#include <vector>
#include "MDDFTProblem.cpp"
#include "IMDDFTProblem.cpp"
#include "PointwiseMultiplier.cpp"
class Convolution {
public:
    Convolution(const std::vector<int>& sizes) :
        mddftProblem(sizes),
        imddftProblem(sizes),
        pointwiseMultiplier() {}
    
    ~Convolution();

    std::vector<double> convolveSpace(const std::vector<double>& input1,
                                  const std::vector<double>& input2, 
                                  const std::vector<int>& sizes) {
        double *temp1;
        mddftProblem.performMDDFT(input1, temp1);
        pointwiseMultiplier.performPointwiseMultiplication(temp1, input2, sizes);
        imddftProblem.performIMDDFT(temp1, output);
        // additional functions if needed?
    }

private:
    MDDFTProblem mddftProblem;
    IMDDFTProblem imddftProblem;
    PointwiseMultiplier pointwiseMultiplier;
};

  //  #define Convolution::convolveSpace trace


