#include <iostream>
#include <complex>
#include <vector>
#include <algorithm>
#include <fftw3.h>
#include "shim.hpp"

using namespace std;

void print_vec(auto out) {
	cout << "array = ";
	for(auto i: out)
		cout << i << ", ";
	cout << "\n";
}

class Convolver {
	int length;
	vector<double> _a, _b, _output;
	vector<complex<double>> _out_a, _out_b, _out_pointwise;

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
		const vector<double> &input_a,
		const vector<double> &input_b,
		vector<double> &output) {
			// copy inputs
			copy(input_a.begin(), input_a.end(), _a.begin());
			copy(input_b.begin(), input_b.end(), _b.begin());

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
			transform(_out_a.begin(), _out_a.end(),
			_out_b.begin(),
			_out_pointwise.begin(),
			multiplies<complex<double>>());

			// print_vec(_out_pointwise);

			// Inverse FFT
			fftw_execute(backward);

			// print_vec(_output);

			// Copy back output
			copy(_output.begin(), _output.begin() + output.size(), output.begin()); // do I only copy a half	
	};
};