#include <iostream>
#include <functional>
#include <vector>
#include <complex>
#include <cmath>

#include "protox.hpp"
#include "fftx.hpp"
#include "blasx.hpp"

// #lambda that takes an 11 point input and a 6 point output and triple that is the location
// and it calls matrix vector product

int lin_idx(int x, int y, int z){
    return 1;
};

void TResample(const std::vector<int>& out = {1, 2, 3}, const std::vector<int>& in = {1, 2, 3},  const std::vector<double>& shifts = {1, 2, 3}, double *output = NULL, double *input = NULL) {
    return;
};

void test(){
    return;
};

class CSRMat{
    private:
        std::vector<int> rows;
        std::vector<int> cols;
        std::vector<std::complex<double>> vals;

    public:
        CSRMat(std::vector<int> r, std::vector<int> c, std::vector<std::complex<double>> v) : rows(r), cols(c), vals(v) {}
};

void spmv(double *output, double *input_vec, CSRMat c){
    return;
};

struct createSpraseMat {
    private:
        // double fmkx;
        // double fmky;
        // double fmkz;
        // double fcv;
        // double fsckv;
        // double fx1v;
        // double fx2v;
        // double fx3v;
        // double invep0;
        // double c2;
    public:

        
        createSpraseMat() {}

        void operator () (double * output, double* input_arr, double rank, double * sym, int * sym_lengths, int x, int y, int z, int pos, int length, int n, double c2, double invep0) const {
            std::vector<double> loc;
            for(int i = 0; i< rank; i++) {
                loc.push_back(input_arr[i*length + pos]);
            }
            std::vector<int> rows{0,6,12,18,23,28,33};
            std::vector<int> cols{0,4,5,6,9,10,
                                  1,3,5,7,9,10,
                                  2,3,4,8,9,10,
                                  1,2,3,7,8,
                                  0,2,4,6,8,
                                  0,1,5,6,7};
            int ii = lin_idx(x,y,z);
            double fmkx = sym[x];
            double fmky = sym[sym_lengths[0]+y];
            double fmkz = sym[sym_lengths[0]+sym_lengths[1]+z];
            double fcv = sym[sym_lengths[0]+sym_lengths[1]+sym_lengths[2]+ii];
            double fsckv = sym[sym_lengths[0]+sym_lengths[1]+sym_lengths[2]+sym_lengths[3]+ii];
            double fx1v = sym[sym_lengths[0]+sym_lengths[1]+sym_lengths[2]+sym_lengths[3]+sym_lengths[4]+ii];
            double fx2v = sym[sym_lengths[0]+sym_lengths[1]+sym_lengths[2]+sym_lengths[3]+sym_lengths[4]+sym_lengths[5]+ii];
            double fx3v = sym[sym_lengths[0]+sym_lengths[1]+sym_lengths[2]+sym_lengths[3]+sym_lengths[4]+sym_lengths[5]+sym_lengths[6]+ii];

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
                
            //    std::complex<double>(0, fcv / (n*n*n)), 
                std::complex<double>(0, (fmkz * fsckv) / (n*n*n)),
                std::complex<double>(0, (-fmky * fsckv )/ (n*n*n)),
                std::complex<double>(0, (fcv) / (n*n*n)),
                std::complex<double>(0, -fmkz * fx1v / (n*n*n)),
                std::complex<double>(0, fmky * fx1v / (n*n*n)),
                
                //    std::complex<double>(0, fcv / (n*n*n)), 
                std::complex<double>(0, (fmkz * fsckv) / (n*n*n)),
                std::complex<double>(0, (-fmkx * fsckv )/ (n*n*n)),
                std::complex<double>(0, (fcv) / (n*n*n)),
                std::complex<double>(0, fmkz * fx1v / (n*n*n)),
                std::complex<double>(0, -fmkx * fx1v / (n*n*n)),
                
                //    std::complex<double>(0, fcv / (n*n*n)), 
                std::complex<double>(0, (fmky * fsckv) / (n*n*n)),
                std::complex<double>(0, (-fmkx * fsckv )/ (n*n*n)),
                std::complex<double>(0, (fcv) / (n*n*n)),
                std::complex<double>(0, -fmky * fx1v / (n*n*n)),
                std::complex<double>(0, fmkx * fx1v / (n*n*n))};
            std::vector<double> spmv_res;
            CSRMat c(rows, cols, vals);
            spmv(spmv_res.data(), loc.data(), c);
            for(int i = 0; i < 6; i++) {
                output[i*length + pos] = spmv_res[i];
            }
        } 

};

class WarpXProblem: public FFTXProblem {
public:
    using FFTXProblem::FFTXProblem;
    void randomProblemInstance() {
    }
    void semantics() {

        int total_size = sizes.at(0) * sizes.at(1) * sizes.at(2);
        std::vector<int> modified{sizes.at(0), sizes.at(1), (*(WarpXconfig*)args.at(3)).nf};
        int n = (*(WarpXconfig*)args.at(3)).n;
        int np = (*(WarpXconfig*)args.at(3)).np;
        int inFields = (*(WarpXconfig*)args.at(3)).inFields;
        int outFields = (*(WarpXconfig*)args.at(3)).outFields;

        NDTensor X = (*(NDTensor*)args.at(1));
        NDTensor Y = (*(NDTensor*)args.at(0));
        NDTensor boxBig0(inFields, sizes);
        NDTensor boxBig1(inFields, modified);
        NDTensor boxBig2(outFields, modified);
        NDTensor boxBig3(outFields, sizes);
      
        TResample({n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, boxBig0.getCube(0), X.getCube(0));// TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]), nth(boxBig0, 0), nth(X, 0),
        TResample({n, n, n}, {np, np, n}, {0.0, -0.5, 0.0}, boxBig0.getCube(1), X.getCube(1));// TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 1), nth(X, 1),
        TResample({n, n, n}, {np, np, np}, {-0.5, 0.0, 0.0}, boxBig0.getCube(2), X.getCube(2));// TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 2), nth(X, 2),
    

        TResample({n, n, n}, {n, n, np}, {-0.5, -0.5, 0.0}, boxBig0.getCube(3), X.getCube(3));// TResample([n, n, n], [n, n, np], [-0.5, -0.5, 0.0]), nth(boxBig0, 3), nth(X, 3),
        TResample({n, n, n}, {n, np, n}, {-0.5, 0.0, -0.5}, boxBig0.getCube(4), X.getCube(4));// TResample([n, n, n], [n, np, n], [-0.5, 0.0, -0.5]), nth(boxBig0, 4), nth(X, 4),
        TResample({n, n, n}, {np, n, n}, {0.0, -0.5, -0.5}, boxBig0.getCube(5), X.getCube(5));// TResample([n, n, n], [np, n, n], [0.0, -0.5, -0.5]), nth(boxBig0, 5), nth(X, 5),
        

        TResample({n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, boxBig0.getCube(6), X.getCube(6));// TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]), nth(boxBig0, 6), nth(X, 6),
        TResample({n, n, n}, {np, n, np}, {0.0, -0.5, 0.0}, boxBig0.getCube(7), X.getCube(7));// TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 7), nth(X, 7),
        TResample({n, n, n}, {n, np, np}, {-0.5, 0.0, 0.0}, boxBig0.getCube(8), X.getCube(8));// TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 8), nth(X, 8),

  
        TResample({n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, boxBig0.getCube(9), X.getCube(9));// TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 9), nth(X, 9),
        TResample({n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, boxBig0.getCube(10), X.getCube(10));// TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 10), nth(X, 10),
        
        // TTensorI(MDPRDFT([n, n, n], -1), inFields, APar, APar), boxBig1, boxBig0,
        // TRC(TMap(rmat, [iz, iy, ix], AVec, AVec)), boxBig2, boxBig1,
        // TTensorI(IMDPRDFT([n, n, n], 1), outFields, APar, APar), boxBig3, boxBig2,

        // std::vector<double> shift8{0.0, 0.0, 0.5};
        // std::vector<double> shift9{0.0, 0.5, 0.0};
        // std::vector<double> shift10{0.5, 0.0, 0.0};

        TResample({np, np, n}, {n, n, n}, {0.0, 0.0, 0.5}, (double*)args.at(0), boxBig3.data());// TResample([np, np, n], [n, n, n], [0.0, 0.0, 0.5]), nth(Y, 0), nth(boxBig3, 0),
        TResample({np, n, np}, {n, n, n}, {0.0, 0.5, 0.0}, (double*)args.at(0), boxBig3.data());// TResample([np, n, np], [n, n, n], [0.0, 0.5, 0.0]), nth(Y, 1), nth(boxBig3, 1),
        TResample({n, np, np}, {n, n, n}, {0.5, 0.0, 0.0}, (double*)args.at(0), boxBig3.data());// TResample([n, np, np], [n, n, n], [0.5, 0.0, 0.0]), nth(Y, 2), nth(boxBig3, 2),
        
        TResample({n, n, np}, {n, n, n}, {0.5, 0.5, 0.0}, (double*)args.at(0), boxBig3.data());// TResample([n, n, np], [n, n, n], [0.5, 0.5, 0.0]), nth(Y, 3), nth(boxBig3, 3),
        TResample({n, np, n}, {n, n, n}, {0.5, 0.0, 0.5}, (double*)args.at(0), boxBig3.data());// TResample([n, np, n], [n, n, n], [0.5, 0.0, 0.5]), nth(Y, 4), nth(boxBig3, 4),
        TResample({np, n, n}, {n, n, n}, {0.0, 0.5, 0.5}, (double*)args.at(0), boxBig3.data());// TResample([np, n, n], [n, n, n], [0.0, 0.5, 0.5]), nth(Y, 5), nth(boxBig3, 5),
    }
};




// // Base class for functors
// class BaseFunctor {
// public:
//     virtual ~BaseFunctor() = default;
//     virtual double operator()(double input_arr) const = 0;
//     virtual void setParameters(double a, double b, double c, double n, int s) = 0;
// };

// // Functor 1: divideFunc
// struct DivideFunc : public BaseFunctor {
// private:
//     double num;
//     double fcv;

// public:
//     double operator()(double input_arr) const override {
//         return input_arr * fcv / (std::pow(num, 3));
//     }

//     void setParameters(double v, double n) override {
//         fcv = v;
//         num = n;
//     }
// };

// // Functor 2: cxpack3argFunc
// struct Cxpack3argFunc : public BaseFunctor {
// private:
//     double fmk_x;
//     double c2;
//     double fsckv;
//     double num;
//     double sign;

// public:
//     double operator()(double input_arr) const override {
//         return std::complex<double>(0, (sign * fmk_x * c2 * fsckv) / (std::pow(num, 3))).real();
//     }

//     void setParameters(double a, double b, double c, double n, int s) override {
//         fmk_x = a;
//         c2 = b;
//         fsckv = c;
//         num = n;
//         sign = (s > 0) ? 1 : -1;
//     }
// };

// // Functor 3: cxpack2argFunc
// struct Cxpack2argFunc : public BaseFunctor {
// private:
//     double fmk_x;
//     double fsckv;
//     double num;
//     double sign;

// public:
//     double operator()(double input_arr) const override {
//         return std::complex<double>(0, (sign * fmk_x * fsckv) / (std::pow(num, 3))).real();
//     }

//     void setParameters(double a, double b, double n, int s) override {
//         fmk_x = a;
//         fsckv = b;
//         num = n;
//         sign = (s > 0) ? 1 : -1;
//     }
// };

// static std::vector<int> rows = {0,6,12,18,23,28,33};
// static std::vector<int> cols{0,4,5,6,9,10,
//                             1,3,5,7,9,10,
//                             2,3,4,8,9,10,
//                             1,2,3,7,8,
//                             0,2,4,6,8
//                             0,1,5,6,7};
// static std::vector<std::unique_ptr<BaseFunctor>> expr{
        // std::make_unique<DivideFunc>(),std::make_unique<Cxpack3argFunc>(),std::make_unique<Cxpack3argFunc>(),
        // std::make_unique<Cxpack2argFunc>(),std::make_unique<Cxpack2argFunc>(),std::make_unique<Cxpack2argFunc>(),

        // std::make_unique<DivideFunc>(),std::make_unique<Cxpack3argFunc>(),std::make_unique<Cxpack3argFunc>(),
        // std::make_unique<Cxpack2argFunc>(),std::make_unique<Cxpack2argFunc>(),std::make_unique<Cxpack2argFunc>(),

        // std::make_unique<DivideFunc>(),std::make_unique<Cxpack3argFunc>(),std::make_unique<Cxpack3argFunc>(),
        // std::make_unique<Cxpack2argFunc>(),std::make_unique<Cxpack2argFunc>(),std::make_unique<Cxpack2argFunc>(),

        // std::make_unique<Cxpack3argFunc>(),std::make_unique<Cxpack3argFunc>(),std::make_unique<DivideFunc>(),
        // std::make_unique<Cxpack2argFunc>(),std::make_unique<Cxpack2argFunc>(),

        // std::make_unique<Cxpack3argFunc>(),std::make_unique<Cxpack3argFunc>(),std::make_unique<DivideFunc>(),
        // std::make_unique<Cxpack2argFunc>(),std::make_unique<Cxpack2argFunc>(),

        // std::make_unique<Cxpack3argFunc>(),std::make_unique<Cxpack3argFunc>(),std::make_unique<DivideFunc>(),
        // std::make_unique<Cxpack2argFunc>(),std::make_unique<Cxpack2argFunc>()
        // };



// struct divideFunc {
//     private:
//         double num;
//         double fcv;
//     public:
//         divideFunc(double v, double n) : fcv(v), num(n) {}

//         double operator () (double input_arr) const {
//             return input_arr * fcv/(std::pow(num,3));
//         } 

// }

// struct cxpack3argFunc{
//     private:
//         double fmk_x;
//         double c2;
//         double fsckv;
//         double num;
//         double sign;
//     public:
//         cxpack3argFunc(double a, double b, double c, double n, int s) : fmk_x(a), c2(b), fsckv(c), num(n), (s > 0 ? sign = 1 : sign = -1){}

//         std::complex<double> operator() (double input_arr) const {
//             return std::complex<double> res(0, ((sign * fmk_x*c2*fsckv)/(std::pow(num,3))));
//         }

// }

// struct cxpack2argFunc{
//     private:
//         double fmk_x;
//         double fsckv;
//         double num;
//         double sign;
//     public:
//         cxpack2argFunc(double a, double b, double n, int s) : fmk_x(a), fsckv(b), num(n), (s > 0 ? sign = 1 : sign = -1) {}

//         std::complex<double> operator() (double input_arr) const {
//             return std::complex<double> res(0, ((sign * fmk_x*fsckv)/(std::pow(num,3))));
//         }
// }