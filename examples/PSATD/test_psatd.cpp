#include <iostream>
#include <vector> 
#include <complex>
#include "fftw3.h"
#include "domainx.hpp"
#include <cmath>
#include "spiral_generated_psatd.hpp"

template <typename T>
void print1(T* input, std::string s){
  std::cout << "New obj print " << s << std::endl;
  for(int i = 0; i < 10; i++)
    std::cout << input[i] << " ";
  std::cout << std::endl;
  std::cout << "New obj print end " << s <<  std::endl;
}

template <typename T>
void print(T** input, int rank){
    std::cout << "New meta obj print" << std::endl;
    for(int j = 0; j < rank; j++) {
      std::cout << j << ": ";
      for(int i = 0; i < 10; i++) {
        std::cout << input[j][i] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "New meta obj print end" << std::endl;
}

const double PI = std::acos(-1.0); // Value of π

// Function to compute the nth root of unity for a given k
//need to know n
std::complex<double> calculate_w(int n, int k) {
    return std::polar(1.0, 2 * PI * k / n);
}

//shift each points based on shifts vector

//either have the guru api that does the vector stride differently
//otherwise copy from 81 81 80 to 80 80 80 then fft pointwise inverse fft
//time domain and frequency domain ffts
void Resample(const std::vector<int>& out = {1, 2, 3}, 
                const std::vector<int>& in = {1, 2, 3},  
                const std::vector<double>& shifts = {1, 2, 3}, 
                double *output = NULL, 
                double *input = NULL) {
    // print1(input, "input");
     std::vector<double> temp(out.at(2)*out.at(1)*out.at(0));
     for(int i = 0; i < out.at(2); i++) { //loop over planes z
      for(int j = 0; j < out.at(0); j++) { // loop over rows x
        for(int k = 0; k < out.at(1); k++) { // loop over columns y
              temp.at(i*out.at(1)*out.at(0) + j*out.at(1) + k) = input[i*out.at(1)*out.at(0) + j*out.at(1) + k];
        }
      }   
    }
    // print1(temp.data(), "temp");
    std::vector<std::complex<double>> fft_out(out.at(2)*out.at(1)*out.at(0));
    fftw_plan p1 = fftw_plan_dft_r2c_3d(out.at(0), out.at(1), out.at(2), temp.data(),
                  (fftw_complex*)fft_out.data(), FFTW_ESTIMATE);
    fftw_execute(p1);
    // print1(fft_out.data(), "fft_out");
    for(int i = 0; i < out.at(2); i++) { //loop over planes z
      for(int j = 0; j < out.at(0); j++) { // loop over rows x
        for(int k = 0; k < out.at(1); k++) { // loop over columns y
            std::complex<double> shift = std::complex<double>(1,1);
            if(shifts[0] != 0){
              // enumerate the e^whatever * the index in the direction you are going k = loop index x n = dimension always
              shift *= calculate_w(out[0], j); 
            }
            if(shifts[1] != 0){
              shift *= calculate_w(out[1], k);
            }
            if(shifts[2] != 0){
              shift *= calculate_w(out[2], i);
            }
            int index = i*(out.at(0)*out.at(1)) + j*(out.at(1)) + k;
            fft_out[index] = shift * fft_out[index];
        }
      }
    }
    // print1(fft_out.data(), "fft_out");
   fftw_plan ip1 = fftw_plan_dft_c2r_3d(out.at(0), out.at(1), out.at(2), (fftw_complex*)fft_out.data(),
                  output, FFTW_ESTIMATE);  
   fftw_execute(ip1);
  //  print1(output, "output resample");
}

void buildSymbol(double**& symbol, int length, const std::vector<int>& dims = {1, 2, 3}) {
   symbol = new double*[length];
   for(int i = 0; i < length; i++){
      if(i == 0){
        symbol[i] = new double[dims.at(0)];
        for(int j = 0; j < dims.at(0); j++)
          symbol[i][j] = 1.0;
      }
      else if(i == 1){
        symbol[i] = new double[dims.at(1)];
        for(int j = 0; j < dims.at(0); j++)
          symbol[i][j] = 1.0;
      }
      else if(i == 2){
        symbol[i] = new double[dims.at(2)];
        for(int j = 0; j < dims.at(0); j++)
          symbol[i][j] = 1.0;
      }
      else {
        symbol[i] = new double[dims.at(0) * dims.at(1) * dims.at(2)];
        for(int j = 0; j < dims.at(0) * dims.at(1) * dims.at(2); j++){
          symbol[i][j] = 1.0;
        }
      }
   }
}

CSRMat createSparseMat(double** sym, const std::vector<int>& sym_dims = {1,2,3}, int n = 1, double c2 = 1, double invep0 = 1, int i = 0, int j = 0, int k = 0){
    std::vector<int> rows{0,6,12,18,23,28,33};
    std::vector<int> cols{0,4,5,6,9,10,
                            1,3,5,7,9,10,
                            2,3,4,8,9,10,
                            1,2,3,7,8,
                            0,2,4,6,8,
                            0,1,5,6,7};
 
    double fmkx = sym[0][j];
    double fmky = sym[1][k];
    double fmkz = sym[2][i];
    double fcv = sym[3][i*sym_dims.at(1)*sym_dims.at(0) + j*sym_dims.at(1) + k];
    double fsckv = sym[4][i*sym_dims.at(1)*sym_dims.at(0) + j*sym_dims.at(1) + k];
    double fx1v = sym[5][i*sym_dims.at(1)*sym_dims.at(0) + j*sym_dims.at(1) + k];
    double fx2v = sym[6][i*sym_dims.at(1)*sym_dims.at(0) + j*sym_dims.at(1) + k];
    double fx3v = sym[7][i*sym_dims.at(1)*sym_dims.at(0) + j*sym_dims.at(1) + k];

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

void suitesparse_spmv(std::complex<double>* output, std::complex<double>* input, CSRMat c){
    for(int i = 0; i < c.rows.size(); i++) {
      output[i] = std::complex<double>(0,0);
      for(int j = c.rows[i]; j < c.rows[i+1]; j++) {
        output[i] += c.vals[j] * input[c.cols[j]];
      }
    }
}

struct WarpXconfig{
    int n = 80;
    int np = n+1;
    int inFields = 11;
    int outFields = 6;
    int nf = n+2;
    int xdim = nf/2;
    int ydim = n;
    int zdim = n;
    double cvar = 10;
    double ep0var = 10;
    double c2 = cvar*cvar;
    double invep0 = 1/ep0var; 
    double** symbol;
};

int main() {

    WarpXconfig conf;
    int n = conf.n;
    int np = conf.np;
    int nf = conf.nf;  
    NDTensor<double> input(conf.inFields, {{np, np, n},
                                           {np, n, np},
                                           {n, np, np},
                                           {n, n, np},
                                           {n, np, n},
                                           {np, n, n},
                                           {np, np, n},
                                           {np, n, np},
                                           {n, np, np},
                                           {np, np, np},
                                           {np, np, np}});
                                           
    NDTensor<double> output(conf.outFields, {{np, np, n},
                                             {np, n, np},
                                             {n, np, np},
                                             {n, n, np},
                                             {n, np, n},
                                             {np, n, n}});
    
    input.buildTensor();
    buildSymbol(conf.symbol, 8, {nf/2, n, n});

    NDTensor<double> boxBig0(conf.inFields, {n,n,n});
    NDTensor<std::complex<double>> boxBig1(conf.inFields, {n, n, nf});
    NDTensor<std::complex<double>> boxBig2(conf.outFields, {n, n, nf});
    NDTensor<double> boxBig3(conf.outFields, {n,n,n});
                                                                       
    //can be viewed as a stencil half-shift 
    Resample({n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, boxBig0.getField(0), input.getField(0));// TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]), nth(boxBig0, 0), nth(X, 0),
    print1(boxBig0.getField(0), "boxBig0[0]");
    // exit(0);
    Resample({n, n, n}, {np, np, n}, {0.0, -0.5, 0.0}, boxBig0.getField(1), input.getField(1));// TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 1), nth(X, 1),
    Resample({n, n, n}, {np, np, np}, {-0.5, 0.0, 0.0}, boxBig0.getField(2), input.getField(2));// TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 2), nth(X, 2),
    Resample({n, n, n}, {n, n, np}, {-0.5, -0.5, 0.0}, boxBig0.getField(3), input.getField(3));// TResample([n, n, n], [n, n, np], [-0.5, -0.5, 0.0]), nth(boxBig0, 3), nth(X, 3),
    Resample({n, n, n}, {n, np, n}, {-0.5, 0.0, -0.5}, boxBig0.getField(4), input.getField(4));// TResample([n, n, n], [n, np, n], [-0.5, 0.0, -0.5]), nth(boxBig0, 4), nth(X, 4),
    Resample({n, n, n}, {np, n, n}, {0.0, -0.5, -0.5}, boxBig0.getField(5), input.getField(5));// TResample([n, n, n], [np, n, n], [0.0, -0.5, -0.5]), nth(boxBig0, 5), nth(X, 5),
  
    Resample({n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, boxBig0.getField(6), input.getField(6));// TResample([n, n, n], [np, np, n], [0.0, 0.0, -0.5]), nth(boxBig0, 6), nth(X, 6),
    Resample({n, n, n}, {np, n, np}, {0.0, -0.5, 0.0}, boxBig0.getField(7), input.getField(7));// TResample([n, n, n], [np, n, np], [0.0, -0.5, 0.0]), nth(boxBig0, 7), nth(X, 7),
    Resample({n, n, n}, {n, np, np}, {-0.5, 0.0, 0.0}, boxBig0.getField(8), input.getField(8));// TResample([n, n, n], [n, np, np], [-0.5, 0.0, 0.0]), nth(boxBig0, 8), nth(X, 8),
    Resample({n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, boxBig0.getField(9), input.getField(9));// TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 9), nth(X, 9),
    Resample({n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, boxBig0.getField(10), input.getField(10));// TResample([n, n, n], [np, np, np], [0.0, 0.0, 0.0]), nth(boxBig0, 10), nth(X, 10),
    
    fftw_plan p1 = fftw_plan_dft_r2c_3d(n, n, n, boxBig0.getField(0),
                  (fftw_complex*)boxBig1.getField(0), FFTW_ESTIMATE);
    fftw_plan p2 = fftw_plan_dft_r2c_3d(n, n, n, boxBig0.getField(1),
                  (fftw_complex*)boxBig1.getField(1), FFTW_ESTIMATE);
    fftw_plan p3 = fftw_plan_dft_r2c_3d(n, n, n, boxBig0.getField(2),
                  (fftw_complex*)boxBig1.getField(2), FFTW_ESTIMATE);
    fftw_plan p4 = fftw_plan_dft_r2c_3d(n, n, n, boxBig0.getField(3),
                  (fftw_complex*)boxBig1.getField(3), FFTW_ESTIMATE);
    fftw_plan p5 = fftw_plan_dft_r2c_3d(n, n, n, boxBig0.getField(4),
                  (fftw_complex*)boxBig1.getField(4), FFTW_ESTIMATE);
    fftw_plan p6 = fftw_plan_dft_r2c_3d(n, n, n, boxBig0.getField(5),
                  (fftw_complex*)boxBig1.getField(5), FFTW_ESTIMATE);
    fftw_plan p7 = fftw_plan_dft_r2c_3d(n, n, n, boxBig0.getField(6),
                  (fftw_complex*)boxBig1.getField(6), FFTW_ESTIMATE);
    fftw_plan p8 = fftw_plan_dft_r2c_3d(n, n, n, boxBig0.getField(7),
                  (fftw_complex*)boxBig1.getField(7), FFTW_ESTIMATE);
    fftw_plan p9 = fftw_plan_dft_r2c_3d(n, n, n, boxBig0.getField(8),
                  (fftw_complex*)boxBig1.getField(8), FFTW_ESTIMATE);
    fftw_plan p10 = fftw_plan_dft_r2c_3d(n, n, n, boxBig0.getField(9),
                  (fftw_complex*)boxBig1.getField(9), FFTW_ESTIMATE);
    fftw_plan p11 = fftw_plan_dft_r2c_3d(n, n, n, boxBig0.getField(10),
                  (fftw_complex*)boxBig1.getField(10), FFTW_ESTIMATE);

    fftw_execute(p1);
    fftw_execute(p2);
    fftw_execute(p3);
    fftw_execute(p4);
    fftw_execute(p5);
    fftw_execute(p6);
    fftw_execute(p7);
    fftw_execute(p8);
    fftw_execute(p9);
    fftw_execute(p10);
    fftw_execute(p11);
    
    std::complex<double> *temp = new std::complex<double>[6];
    for(int i = 0; i < boxBig1.getSize(2); i++) {
      for(int j = 0; j < boxBig1.getSize(0); j++) {
        for(int k = 0; k < boxBig1.getSize(1); k++) {
          CSRMat c = createSparseMat(conf.symbol, {nf/2, n, n}, conf.n, conf.c2, conf.invep0, i, j, k); // TRC(TMap(rmat, [iz, iy, ix], AVec, AVec)), boxBig2, boxBig1,
          suitesparse_spmv(temp, boxBig1.getPencil(i, j, k), c);
          boxBig2.setPencil(temp,i,j,k);
        }
      }
    }


    fftw_plan ip1 = fftw_plan_dft_c2r_3d(n, n, n, (fftw_complex*)boxBig2.getField(0),
                  boxBig3.getField(0), FFTW_ESTIMATE);
    fftw_plan ip2 = fftw_plan_dft_c2r_3d(n, n, n, (fftw_complex*)boxBig2.getField(1),
                  boxBig3.getField(1), FFTW_ESTIMATE);
    fftw_plan ip3 = fftw_plan_dft_c2r_3d(n, n, n, (fftw_complex*)boxBig2.getField(2),
                  boxBig3.getField(2), FFTW_ESTIMATE);
    fftw_plan ip4 = fftw_plan_dft_c2r_3d(n, n, n, (fftw_complex*)boxBig2.getField(3),
                  boxBig3.getField(3), FFTW_ESTIMATE);
    fftw_plan ip5 = fftw_plan_dft_c2r_3d(n, n, n, (fftw_complex*)boxBig2.getField(4),
                  boxBig3.getField(4), FFTW_ESTIMATE);
    fftw_plan ip6 = fftw_plan_dft_c2r_3d(n, n, n, (fftw_complex*)boxBig2.getField(5),
                  boxBig3.getField(5), FFTW_ESTIMATE);

    fftw_execute(ip1);
    fftw_execute(ip2);
    fftw_execute(ip3);
    fftw_execute(ip4);
    fftw_execute(ip5);
    fftw_execute(ip6);

    Resample({np, np, n}, {n, n, n}, {0.0, 0.0, 0.5}, output.getField(0), boxBig3.getField(0));// TResample([np, np, n], [n, n, n], [0.0, 0.0, 0.5]), nth(Y, 0), nth(boxBig3, 0),
    Resample({np, n, np}, {n, n, n}, {0.0, 0.5, 0.0}, output.getField(1), boxBig3.getField(1));// TResample([np, n, np], [n, n, n], [0.0, 0.5, 0.0]), nth(Y, 1), nth(boxBig3, 1),
    Resample({n, np, np}, {n, n, n}, {0.5, 0.0, 0.0}, output.getField(2), boxBig3.getField(2));// TResample([n, np, np], [n, n, n], [0.5, 0.0, 0.0]), nth(Y, 2), nth(boxBig3, 2),
  
    Resample({n, n, np}, {n, n, n}, {0.5, 0.5, 0.0}, output.getField(3), boxBig3.getField(3));// TResample([n, n, np], [n, n, n], [0.5, 0.5, 0.0]), nth(Y, 3), nth(boxBig3, 3),
    Resample({n, np, n}, {n, n, n}, {0.5, 0.0, 0.5}, output.getField(4), boxBig3.getField(4));// TResample([n, np, n], [n, n, n], [0.5, 0.0, 0.5]), nth(Y, 4), nth(boxBig3, 4),
    Resample({np, n, n}, {n, n, n}, {0.0, 0.5, 0.5}, output.getField(5), boxBig3.getField(5));// TResample([np, n, n], [n, n, n], [0.0, 0.5, 0.5]), nth(Y, 5), nth(boxBig3, 5),

    print(output.data(), 6);

    double** spiral_output = new double*[6];
    spiral_output[0] = new double[np * np *n];
    spiral_output[1] = new double[np * n * np];
    spiral_output[2] = new double[n * np * np];
    spiral_output[3] = new double[n * n * np];
    spiral_output[4] = new double[n * np * n];
    spiral_output[5] = new double[np * n * n];

    init_psatd_spiral();
    psatd_spiral(spiral_output, input.data(), conf.symbol, conf.cvar, conf.ep0var); 
    destroy_psatd_spiral();
    print(spiral_output, 6);
    return 0;
}