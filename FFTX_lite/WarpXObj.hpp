#include <iostream>
#include <functional>
#include <vector>
#include <complex>
#include <cmath>

#include "protox.hpp"
#include "fftx.hpp"
#include "graphblasx.hpp"

CSRMat createSparseMat(std::complex<double> *sym, int sym_lengths, int pos, int n, double c2, double invep0, int i, int j, int k){
    std::vector<int> rows{0,6,12,18,23,28,33};
    std::vector<int> cols{0,4,5,6,9,10,
                            1,3,5,7,9,10,
                            2,3,4,8,9,10,
                            1,2,3,7,8,
                            0,2,4,6,8,
                            0,1,5,6,7};
 
    double fmkx = 1;
    double fmky = 1;
    double fmkz = 1;
    double fcv = 1;
    double fsckv = 1;
    double fx1v = 1;
    double fx2v = 1;
    double fx3v = 1;

    std::vector<std::complex<double>> vals{
        std::complex<double>(0, fcv / (n*n*n)), 
        std::complex<double>(0, (-fmkz * c2 * fsckv) / (n*n*n)),
        std::complex<double>(0, (fmkz * c2 * fsckv )/ (n*n*n)),
        std::complex<double>(0, (-invep0 * fsckv) / (n*n*n)),
        std::complex<double>(0, fmkx * fx3v / (n*n*n)),
        std::complex<double>(0, -fmkx * fx2v / (n*n*n)),
        
        std::complex<double>(0, fcv / (n*n*n)), 
        std::complex<double>(0, (-fmkz * c2 * fsckv) / (n*n*n)),
        std::complex<double>(0, (fmkx * c2 * fsckv )/ (n*n*n)),
        std::complex<double>(0, (-invep0 * fsckv) / (n*n*n)),
        std::complex<double>(0, fmky * fx3v / (n*n*n)),
        std::complex<double>(0, -fmky * fx2v / (n*n*n)),
        
        std::complex<double>(0, fcv / (n*n*n)), 
        std::complex<double>(0, (-fmky * c2 * fsckv) / (n*n*n)),
        std::complex<double>(0, (fmkx * c2 * fsckv )/ (n*n*n)),
        std::complex<double>(0, (-invep0 * fsckv) / (n*n*n)),
        std::complex<double>(0, fmkz * fx3v / (n*n*n)),
        std::complex<double>(0, -fmkz * fx2v / (n*n*n)),
        
        std::complex<double>(0, (fmkz * fsckv) / (n*n*n)),
        std::complex<double>(0, (-fmky * fsckv )/ (n*n*n)),
        std::complex<double>(0, (fcv) / (n*n*n)),
        std::complex<double>(0, -fmkz * fx1v / (n*n*n)),
        std::complex<double>(0, fmky * fx1v / (n*n*n)),
        
        std::complex<double>(0, (fmkz * fsckv) / (n*n*n)),
        std::complex<double>(0, (-fmkx * fsckv )/ (n*n*n)),
        std::complex<double>(0, (fcv) / (n*n*n)),
        std::complex<double>(0, fmkz * fx1v / (n*n*n)),
        std::complex<double>(0, -fmkx * fx1v / (n*n*n)),
        
        std::complex<double>(0, (fmky * fsckv) / (n*n*n)),
        std::complex<double>(0, (-fmkx * fsckv )/ (n*n*n)),
        std::complex<double>(0, (fcv) / (n*n*n)),
        std::complex<double>(0, -fmky * fx1v / (n*n*n)),
        std::complex<double>(0, fmkx * fx1v / (n*n*n))};
    CSRMat c(rows, cols, vals);
    return c;
}

class WarpXProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    
    double residual() {
      return 0;
    }

    WarpXProblem(WarpXconfig conf) {
        args.push_back(&conf);
        args.push_back(&conf.symbol);
    }

    void semantics(NDTensor<double> X, NDTensor<double> Y, NDTensor<std::complex<double>> sym, WarpXconfig conf) {

       
        std::vector<int> modified{conf.n, conf.n, conf.nf};
        int n = conf.n;
        int np = conf.np;
        int inFields = conf.inFields;
        int outFields = conf.outFields;
        double c2 = conf.c2;
        double invep0 = conf.invep0;

        NDTensor<double> boxBig0(inFields, sizes);
        NDTensor<std::complex<double>> boxBig1(inFields, modified);
        NDTensor<std::complex<double>> boxBig2(outFields, modified);
        NDTensor<double> boxBig3(outFields, sizes);
      
        //can be viewed as a stencil half-shift 
        TResample({n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, boxBig0.getField(0), X.getField(0));// TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]), nth(boxBig0, 0), nth(X, 0),
        TResample({n, n, n}, {np, np, n}, {0.0, -0.5, 0.0}, boxBig0.getField(1), X.getField(1));// TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 1), nth(X, 1),
        TResample({n, n, n}, {np, np, np}, {-0.5, 0.0, 0.0}, boxBig0.getField(2), X.getField(2));// TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 2), nth(X, 2),
    

        TResample({n, n, n}, {n, n, np}, {-0.5, -0.5, 0.0}, boxBig0.getField(3), X.getField(3));// TResample([n, n, n], [n, n, np], [-0.5, -0.5, 0.0]), nth(boxBig0, 3), nth(X, 3),
        TResample({n, n, n}, {n, np, n}, {-0.5, 0.0, -0.5}, boxBig0.getField(4), X.getField(4));// TResample([n, n, n], [n, np, n], [-0.5, 0.0, -0.5]), nth(boxBig0, 4), nth(X, 4),
        TResample({n, n, n}, {np, n, n}, {0.0, -0.5, -0.5}, boxBig0.getField(5), X.getField(5));// TResample([n, n, n], [np, n, n], [0.0, -0.5, -0.5]), nth(boxBig0, 5), nth(X, 5),
        

        TResample({n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, boxBig0.getField(6), X.getField(6));// TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]), nth(boxBig0, 6), nth(X, 6),
        TResample({n, n, n}, {np, n, np}, {0.0, -0.5, 0.0}, boxBig0.getField(7), X.getField(7));// TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 7), nth(X, 7),
        TResample({n, n, n}, {n, np, np}, {-0.5, 0.0, 0.0}, boxBig0.getField(8), X.getField(8));// TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 8), nth(X, 8),

  
        TResample({n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, boxBig0.getField(9), X.getField(9));// TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 9), nth(X, 9),
        TResample({n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, boxBig0.getField(10), X.getField(10));// TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 10), nth(X, 10),
        

        MDPRDFT(boxBig0.data(), boxBig1.data()); // TTensorI(MDPRDFT([n, n, n], -1), inFields, APar, APar), boxBig1, boxBig0,

        //to be replaced with Proto forall
        for(int i = 0; i < boxBig1.getSize(0); i++) {
          for(int j = 0; j < boxBig1.getSize(1); j++) {
            for(int k = 0; k < boxBig1.getSize(2); k++) {
              CSRMat c = createSparseMat(sym.data(), sym.getTotalSize(), i, n, c2, invep0); // TRC(TMap(rmat, [iz, iy, ix], AVec, AVec)), boxBig2, boxBig1,
              suitesparse_spmv(boxBig2.getPencil(i, j, k), boxBig1.getPencil(i, j, k), c);
            }
          }
        }
        IMDPRDFT(boxBig2.data(), boxBig3.data()); // TTensorI(IMDPRDFT([n, n, n], 1), outFields, APar, APar), boxBig3, boxBig2,
       

        TResample({np, np, n}, {n, n, n}, {0.0, 0.0, 0.5}, Y.getField(0), boxBig3.getField(0));// TResample([np, np, n], [n, n, n], [0.0, 0.0, 0.5]), nth(Y, 0), nth(boxBig3, 0),
        TResample({np, n, np}, {n, n, n}, {0.0, 0.5, 0.0}, Y.getField(1), boxBig3.getField(1));// TResample([np, n, np], [n, n, n], [0.0, 0.5, 0.0]), nth(Y, 1), nth(boxBig3, 1),
        TResample({n, np, np}, {n, n, n}, {0.5, 0.0, 0.0}, Y.getField(2), boxBig3.getField(2));// TResample([n, np, np], [n, n, n], [0.5, 0.0, 0.0]), nth(Y, 2), nth(boxBig3, 2),
        
        TResample({n, n, np}, {n, n, n}, {0.5, 0.5, 0.0}, Y.getField(3), boxBig3.getField(3));// TResample([n, n, np], [n, n, n], [0.5, 0.5, 0.0]), nth(Y, 3), nth(boxBig3, 3),
        TResample({n, np, n}, {n, n, n}, {0.5, 0.0, 0.5}, Y.getField(4), boxBig3.getField(4));// TResample([n, np, n], [n, n, n], [0.5, 0.0, 0.5]), nth(Y, 4), nth(boxBig3, 4),
        TResample({np, n, n}, {n, n, n}, {0.0, 0.5, 0.5}, Y.getField(5), boxBig3.getField(5));// TResample([np, n, n], [n, n, n], [0.0, 0.5, 0.5]), nth(Y, 5), nth(boxBig3, 5),
    }
};
