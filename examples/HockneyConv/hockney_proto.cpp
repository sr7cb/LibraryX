#include <iostream>
#include <complex>
// #include <cstring>
#include <vector>
// #include <algorithm>
// #include <functional>
#include "fftw3.h"
#include "Proto.H"

void buildInput(Proto::BoxData<double,1>& input){
    input.setVal(1);
}

void buildInput(Proto::BoxData<std::complex<double>,1>& input) {
    for(int i = 0; i < input.size(); i++){
        input.data()[i] = std::complex<double>(1,0);
    }
}

Proto::BoxData<double,1> zeroPad(const Proto::BoxData<double,1>& input, const std::vector<int> sizes = {1,2,3}) {
    Proto::BoxData<double,1> extended_input(Proto::Box(Proto::Point::Zeros(), Proto::Point({sizes.at(0)-1,sizes.at(1)-1,sizes.at(2)-1})));
    extended_input.setVal(0);
    input.copyTo(extended_input);
    return extended_input;
}

Proto::BoxData<double,1> extract(const Proto::BoxData<double,1>& extended_output, const std::vector<int> sizes = {1,2,3}) {
    Proto::BoxData<double,1> output(Proto::Box(Proto::Point::Zeros(), Proto::Point({sizes.at(0)-1,sizes.at(1)-1,sizes.at(2)-1})));    
    extended_output.copyTo(output);
    return output;
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

    Proto::BoxData<double,1> input(Proto::Box(Proto::Point::Zeros(), Proto::Point({nx-1,ny-1,nz-1})));
    // std::vector<std::complex<double>> input2(Nx*Ny*(Nz/2+1));
    Proto::BoxData<std::complex<double>,1> input2(Proto::Box(Proto::Point::Zeros(), Proto::Point({Nx-1,Ny-1,(Nz/2+1)-1})));
    Proto::BoxData<double, 1> output;
    

    Proto::BoxData<double,1> extended_input;
    Proto::BoxData<double,1> extended_output(Proto::Box(Proto::Point::Zeros(), Proto::Point({Nx-1,Ny-1, Nz-1})));
    extended_output.setVal(0.0);
    std::vector<std::complex<double>> temp(Nx*Ny*(Nz/2+1));
    std::vector<std::complex<double>> out(Nx*Ny*(Nz/2+1));

    buildInput(input);
    buildInput(input2);
   
    extended_input = zeroPad(input, {Nx,Ny,Nz});
    
    fftw_plan p = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, (double*)extended_input.data(),
                  (fftw_complex*)out.data(), FFTW_ESTIMATE);
    fftw_execute(p); 
    
    auto complex_multiply = std::multiplies<
                            std::complex<double>>{}; 
    std::transform(out.begin(), //start location 
                out.end(), //end location
                input2.data(), //2nd input
                temp.begin(), //output 
                complex_multiply); //operator

    fftw_plan p2 = fftw_plan_dft_c2r_3d(Nx, Ny, Nz,
                   (fftw_complex*)temp.data(), (double*)extended_output.data(), FFTW_ESTIMATE);
    // no-op, just collecting parameters
    fftw_execute(p2); 
    // no-op, just collecting parameters
    // fftw_execute(p);
    // extended_output.copyTo(output);
    output = extract(extended_output, {mx,my,mz});
    // for(int i = 0; i < 10; i++)
    //     std::cout << output.data()[i] << std::endl;
}