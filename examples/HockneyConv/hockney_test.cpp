#include <iostream>
#include <complex>
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>
#include "fftw3.h"
// #include "libraryX.hpp"



void zeroPad(std::vector<double>& input, int x_old, int y_old, int z_old, int x, int y, int z) {
    // Create a new output vector initialized with zeros
    std::vector<double> output(x * y * z, 0.0);

    // Copy old data into the appropriate positions in the new vector
    for (int i = 0; i < x_old; ++i) {
        for (int j = 0; j < y_old; ++j) {
            for (int k = 0; k < z_old; ++k) {
                // Calculate the linear index in the old and new layouts
                int oldIndex = i * y_old * z_old + j * z_old + k;
                int newIndex = i * y * z + j * z + k;

                // Copy the value from the old index to the new index
                output[newIndex] = input[oldIndex];
            }
        }
    }

    // Replace the input vector with the padded output
    input = output;
}

void buildInput(std::vector<double>& input) {
    for(int i = 0; i < input.size(); i++) {
        input[i] = 1.0;
    }
}

void buildInput(std::vector<std::complex<double>>& input) {
    for(int i = 0; i < input.size(); i++){
        input[i] = std::complex<double>(1,0);
    }
}

void extractOutput(std::vector<double>& output,const std::vector<double>& input, int x, int y, int z, int x_small, int y_small, int z_small) {
    // Resize the output vector to hold the smaller cube
    output.resize(x_small * y_small * z_small);

    // Copy data from the larger cube into the smaller output cube
    for (int i = 0; i < x_small; ++i) {
        for (int j = 0; j < y_small; ++j) {
            for (int k = 0; k < z_small; ++k) {
                // Calculate the index in the larger input vector
                int inputIndex = i * y * z + j * z + k;
                // Calculate the index in the smaller output vector
                int outputIndex = i * y_small * z_small + j * z_small + k;

                // Copy the value from input to output
                output[outputIndex] = input[inputIndex];
            }
        }
    }
}

void unitTest() {
        std::vector<double> test(3*3*3, 1);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            for (int k = 0; k < 3; k++){
                std::cout << test[i*3*3 + j *3 +k ] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    zeroPad(test, 3, 3, 3, 5, 5, 5);

      for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            for (int k = 0; k < 5; k++){
                std::cout << test[i*5*5 + j*5 +k ] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

int main() {
    int nx = 32;
    int ny = 32;
    int nz = 128;
    int Nx = nx * 2;
    int Ny = ny * 2;
    int Nz = nz * 2;
    int mx = 32;
    int my = 32;
    int mz = 128;
    std::vector<double> input(nx*ny*nz);
    std::vector<double> output(mx*my*mz);
    std::vector<std::complex<double>> input2(Nx*Ny*(Nz/2+1));
    
    // will not be materialized
    std::vector<std::complex<double>> temp(Nx*Ny*(Nz/2+1));
    std::vector<std::complex<double>> out(Nx*Ny*(Nz/2+1));
    std::vector<double> out2(Nx*Ny*Nz);

    buildInput(input);
    buildInput(input2);

    // no-op, just collecting parameters
    zeroPad(input, nx, ny, nz, Nx, Ny, Nz);
    // unitTest();
    // no-op, just collecting parameters
    fftw_plan p = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, input.data(),
                  (fftw_complex*)out.data(), FFTW_ESTIMATE);
    // no-op, just collecting parameters
    fftw_execute(p); 
    for(int i = 0; i < 10; i++)
        std::cout << out[i] << std::endl;
    
    // no-op, just collecting parameters
    auto complex_multiply = std::multiplies<
                            std::complex<double>>{}; 
    std::transform(out.begin(), //start location 
                out.end(), //end location
                input2.begin(), //2nd input
                temp.begin(), //output 
                complex_multiply); //operator
    for(int i = 0; i < 10; i++)
        std::cout << temp[i] << std::endl;
    // exit(0);
    // no-op, just collecting parameters
    fftw_plan p2 = fftw_plan_dft_c2r_3d(Nx, Ny, Nz,
                   (fftw_complex*)temp.data(), out2.data(), FFTW_ESTIMATE);
    // no-op, just collecting parameters
    fftw_execute(p2); 
    for(int i = 0; i < 10; i++)
        std::cout << out2[i] << std::endl;
    // output now contains the correct result, 
    // but temporaries were never materialized
    extractOutput(output, out2, Nx, Ny, Nz, mx, my, mz); 

    std::cout << output.size() << std::endl;

    // for(int i = 0 ; i < output.size(); i++)
    //     std::cout << output[i] << std::endl;
    int sum = 0;
    for(int i = 0; i < output.size(); i++)
        if(abs(output[i]-output[0]) > 1e-7)
            std::cout << output[i] << " " << output[0] << std::endl;
        
    std::cout << sum << std::endl;
    return 0;
}