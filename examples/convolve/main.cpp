#include <iostream>
#include "lib/convolve.hpp"
#include <vector>


std::vector<double> vectorize_int(int n) {
	std::vector<double> vec;
	for(;n > 0; n /= 10) {
		vec.push_back(n % 10);
	}

	std::reverse(vec.begin(), vec.end());

	return vec;
}

int devectorize_int(std::vector<double> &vec, int divide_by = 1) {
	int num = 0;
	for(int i = 0; i < (int)vec.size() - 1; i++) {
		int n = (int)(vec[i] / divide_by + 1e-6); 
		num += n / 10;
		num = num * 10 + (n % 10);
	}

	return num;
}

int main() {
	// auto input_a = vectorize_int(23213), input_b = vectorize_int(41434);
	auto input_a = vectorize_int(23), input_b = vectorize_int(41);
	int N = 2 * input_a.size();
	std::vector<double> output(N);

	Convolver convolver(N);
	convolver.convolve(input_a, input_b, output);
	// print_vec(output);

	int n = devectorize_int(output, N);
	std::cout << "n = " << n << "\n";
}