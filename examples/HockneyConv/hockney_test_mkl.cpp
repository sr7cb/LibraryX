#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include "chrono"
#include "oneapi/mkl.hpp"

using namespace sycl;

void checkOutputBuffers(double *spiral_Y, double *sycl_Y, long arrsz) {
    bool correct = true;
    double maxdelta = 0.0;

    for (int indx = 0; indx < arrsz; indx++) {
        double s = spiral_Y[indx];
        double c = sycl_Y[indx];

        double deltar = std::abs(s - c);
        bool elem_correct = (deltar < 1e-7);
        maxdelta = maxdelta < deltar ? deltar : maxdelta;
        correct &= elem_correct;
    }

    std::cout << "Correct: " << (correct ? "True" : "False") << "\tMax delta = " << maxdelta << std::endl;
}

void buildInput(std::vector<double> &input) {
    std::fill(input.begin(), input.end(), 1.0);
}

void buildInput(std::vector<std::complex<double>> &input) {
    std::fill(input.begin(), input.end(), std::complex<double>(1.0, 0.0));
}

sycl::event zeroPad(queue &q, double *d_output, const std::vector<double> &input, 
             int x_old, int y_old, int z_old, int x, int y, int z) {
    q.fill(d_output, 0.0, x * y * z).wait();

    buffer<double, 1> input_buf(input.data(), range<1>(x_old * y_old * z_old));

    sycl::event e = q.submit([&](handler &h) {
        auto input_acc = input_buf.get_access<access::mode::read>(h);
        h.parallel_for(range<3>(x_old, y_old, z_old), [=](id<3> idx) {
            int i = idx[0], j = idx[1], k = idx[2];
            int oldIndex = i * y_old * z_old + j * z_old + k;
            int newIndex = i * y * z + j * z + k;
            d_output[newIndex] = input_acc[oldIndex];
        });
    });
	e.wait();
	return e;
}

sycl::event extract(queue &q, std::vector<double> &output, double *d_input, 
             int x, int y, int z, int x_small, int y_small, int z_small) {
    output.resize(x_small * y_small * z_small);

    buffer<double, 1> output_buf(output.data(), range<1>(x_small * y_small * z_small));

    sycl::event e = q.submit([&](handler &h) {
        auto output_acc = output_buf.get_access<access::mode::write>(h);
        h.parallel_for(range<3>(x_small, y_small, z_small), [=](id<3> idx) {
            int i = idx[0], j = idx[1], k = idx[2];
            int inputIndex = i * y * z + j * z + k;
            int outputIndex = i * y_small * z_small + j * z_small + k;
            output_acc[outputIndex] = d_input[inputIndex];
        });
    });
	e.wait();

	return e;
}

int main() {
    constexpr int nx = 32, ny = 32, nz = 128;
    constexpr int Nx = nx * 2, Ny = ny * 2, Nz = nz * 2;
    constexpr int mx = 32, my = 32, mz = 128;

    std::vector<double> input(nx * ny * nz);
    std::vector<double> output(mx * my * mz);
    std::vector<double> spiral_output(mx * my * mz);
    std::vector<std::complex<double>> input2(Nx * Ny * (Nz / 2 + 1));

    buildInput(input);
    buildInput(input2);

    queue q{sycl::gpu_selector{},
                      sycl::property::queue::enable_profiling{}};

    double *d_extended_input = malloc_device<double>(Nx * Ny * Nz, q);
    std::complex<double> *d_out = malloc_device<std::complex<double>>(Nx * Ny * (Nz / 2 + 1), q);
    std::complex<double> *d_temp = malloc_device<std::complex<double>>(Nx * Ny * (Nz / 2 + 1), q);
    double *d_out2 = malloc_device<double>(Nx * Ny * Nz, q);

    // Copy `input2` to `d_temp` for element-wise complex multiplication
    q.memcpy(d_temp, input2.data(), Nx * Ny * (Nz / 2 + 1) * sizeof(std::complex<double>)).wait();
   
	// Zero-pad input
    sycl::event start = zeroPad(q, d_extended_input, input, nx, ny, nz, Nx, Ny, Nz);

    // Create cuFFT-like plans and execute
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL> fftPlan({Nx, Ny, Nz});
    fftPlan.commit(q);
    oneapi::mkl::dft::compute_forward(fftPlan, d_extended_input, reinterpret_cast<double *>(d_out)).wait();

/*	std::vector<std::complex<double>> hdout(1);
	q.memcpy(hdout.data(), d_out, sizeof(std::complex<double>)).wait();
	std::cout << hdout[0] << std::endl;
*/

    // Launch kernel for element-wise complex multiplication
    q.parallel_for(range<1>(Nx * Ny * (Nz / 2 + 1)), [=](id<1> idx) {
        int tid = idx[0];
        std::complex<double> a = d_out[tid];
        std::complex<double> b = d_temp[tid];
        d_temp[tid] = a * b;
    }).wait();

    // Inverse FFT
    oneapi::mkl::dft::compute_backward(fftPlan, reinterpret_cast<double *>(d_temp), d_out2).wait();

    // Copy the result back to the host
    sycl::event end = extract(q, output, d_out2, Nx, Ny, Nz, mx, my, mz);

	end.wait();

    // Get profiling info
    auto starttime = start.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto endtime = end.get_profiling_info<sycl::info::event_profiling::command_end>();

    // Calculate total elapsed time in milliseconds
    double elapsed_time = (endtime - starttime) / 1e6; // Convert nanoseconds to milliseconds

    std::cout << "Total execution time for sequential events: " << elapsed_time << " ms" << std::endl;



    for(int i = 0; i < 10; i++) {
		std::cout << output[i] << std::endl;
	}
	// Check against spiral_output (assuming hockney results are in `spiral_output`)
    //checkOutputBuffers(output.data(), spiral_output.data(), mx * my * mz);

    // Clean up
    free(d_extended_input, q);
    free(d_out, q);
    free(d_temp, q);
    free(d_out2, q);

    return 0;
}
