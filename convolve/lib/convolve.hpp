#include <iostream>
#include <complex>
#include <vector>
#include <algorithm>
#include <fftw3.h>
#include "shim.hpp"

void print_vec(auto out) {
	std::cout << "array = ";
	for(auto i: out)
		std::cout << i << ", ";
	std::cout << "\n";
}

class Convolver {
	int length;
	std::vector<double> _a, _b, _output;
	std::vector<std::complex<double>> _out_a, _out_b, _out_pointwise;

	fftw_plan forward_a, forward_b, backward;
public:
	Convolver(int padded_length) : length(padded_length) {
		_a.assign(length, 0);
		_b.assign(length, 0);

	    _out_a.assign(length, 0);
	    _out_b.assign(length, 0);
	    _out_pointwise.assign(length, 0);
	    _output.assign(length, 0);

	    forward_a = fftw_plan_dft_r2c_1d(length, _a.data(),
	     reinterpret_cast<fftw_complex*>(_out_a.data()), FFTW_ESTIMATE);
	    forward_b = fftw_plan_dft_r2c_1d(length, _b.data(),
	     reinterpret_cast<fftw_complex*>(_out_b.data()), FFTW_ESTIMATE);

	    backward = fftw_plan_dft_c2r_1d(length,
	     reinterpret_cast<fftw_complex*>(_out_pointwise.data()), _output.data(), FFTW_ESTIMATE);
	}

	~Convolver(){
		fftw_destroy_plan(forward_a);
		fftw_destroy_plan(forward_b);
		fftw_destroy_plan(backward);
	}

	void convolve(
		const std::vector<double> &input_a,
		const std::vector<double> &input_b,
		std::vector<double> &output) {
			// copy inputs
            dag.addInputArg(input_a);
            dag.addInputArg(input_b);
            dag.addOutputArg(output);
			std::copy(input_a.begin(), input_a.end(), _a.begin());
			std::copy(input_b.begin(), input_b.end(), _b.begin());

			// print_vec(input_a);
			// print_vec(input_b);
			// print_vec(_a);
			// print_vec(_b);
			
			// Forward FFT
			fftw_execute(forward_a);
			fftw_execute(forward_b);

			// print_vec(_out_a);
			// print_vec(_out_b);

			// Point-wise
			std::transform(_out_a.begin(), _out_a.end(),
			_out_b.begin(),
			_out_pointwise.begin(),
			std::multiplies<std::complex<double>>());

			// print_vec(_out_pointwise);

			// Inverse FFT
			fftw_execute(backward);

			// print_vec(_output);

			// Copy back output
			std::copy(_output.begin(), _output.begin() + output.size(), output.begin()); // do I only copy a half	
	};
};