#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <numeric> // for std::accumulate
#include "oneapi/mkl.hpp"

using namespace sycl;

// void printBuffer(q, buffer<double,1> b, int vals) {
    
// }

size_t product3(const std::vector<int> &dims) {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
}

std::complex<double> calculate_w(int n, int k) {
    constexpr double PI = 3.14159265358979323846;
    double angle = 2.0 * PI * k / n;
    return std::complex<double>(std::cos(angle), std::sin(angle));
}

// SYCL kernel function to apply shifts
void apply_shift(sycl::queue *q, sycl::buffer<double, 1> &fft_out, sycl::int3 dims, sycl::int3 shifts, int total_elements) {
    q->submit([&](sycl::handler &h) {
        // Accessor for the buffer
        auto acc = fft_out.get_access<sycl::access::mode::read_write>(h);

        h.parallel_for(sycl::range<1>(total_elements), [=](sycl::id<1> idx) {
            int index = idx[0];

            // Compute the 3D indices from the 1D index
            int i = index / (dims.x() * dims.y());
            int j = (index / dims.y()) % dims.x();
            int k = index % dims.y();

            // Each complex number has two components (real, imag)
            int base_index = 2 * index; // Real part at `base_index`, imaginary part at `base_index + 1`

            // Load the current complex value
            double real = acc[base_index];
            double imag = acc[base_index + 1];
            std::complex<double> current_value(real, imag);

            // Initialize the shift as a unit complex number
            std::complex<double> shift(1.0, 0.0);

            // Apply shifts in each dimension
            if (shifts.x() != 0) {
                shift *= calculate_w(dims.x(), j);
            }
            if (shifts.y() != 0) {
                shift *= calculate_w(dims.y(), k);
            }
            if (shifts.z() != 0) {
                shift *= calculate_w(dims.z(), i);
            }

            // Apply the shift to the complex number
            std::complex<double> result = shift * current_value;

            // Write back the shifted complex number
            acc[base_index] = result.real();
            acc[base_index + 1] = result.imag();
        });
    });
}

sycl::event extract(queue *q, buffer<double,1>& output, buffer<double,1>& d_input, 
             int x, int y, int z, int x_small, int y_small, int z_small, std::vector<int> in, int offset) {
    // output.resize(x_small * y_small * z_small);

    // buffer<double, 1> output_buf(output.data(), range<1>(x_small * y_small * z_small));

    sycl::event e = q->submit([&](handler &h) {
        auto output_acc = output.get_access<access::mode::write>(h);
        // auto input_acc = d_input.get_access<access::mode::write>(h);
        sycl::accessor input_acc(d_input, h, range<1>(product3(in)), id<1>(offset));
        
        h.parallel_for(range<3>(x_small, y_small, z_small), [=](id<3> idx) {
            int i = idx[0], j = idx[1], k = idx[2];
            int inputIndex = i * y * z + j * z + k;
            int outputIndex = i * y_small * z_small + j * z_small + k;
            output_acc[outputIndex] = input_acc[inputIndex];
        });
    });
	e.wait();

	return e;
}

void Resample(queue *q = nullptr, const std::vector<int>& out = {1, 2, 3}, 
                const std::vector<int>& in = {1, 2, 3},  
                const std::vector<double>& shifts = {1, 2, 3}, 
                buffer<double,1>* output = nullptr, 
                buffer<double,1>* input = nullptr,
                int offset = 0) {
    std::vector<double> eo(product3(out),0);
    buffer<double,1> eoutput(eo.data(), range<1>(product3(out)));
    extract(q, eoutput, *input, in.at(0), in.at(1), in.at(2), out.at(0), out.at(1), out.at(2), in, offset);

    std::vector<double> fft((out.at(0)*out.at(1)*(out.at(2)/2+1))*2);
    buffer<double,1> fft_buffer(fft.data(), range<1>((out.at(0)*out.at(1)*(out.at(2)/2+1))*2));
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL> fftPlan({out.at(0), out.at(1), out.at(2)});
    fftPlan.commit(*q);
    oneapi::mkl::dft::compute_forward(fftPlan, eoutput, fft_buffer);
    // q.wait();

    int3 dims = {out.at(0), out.at(1), out.at(2)};
    int3 dshifts = {shifts.at(0), shifts.at(1), shifts.at(2)};
    apply_shift(q, fft_buffer, dims, dshifts, dims.x() * dims.y() *dims.z());

    std::vector<double> d_out2(out.at(0)*out.at(1)*out.at(2));
    buffer<double,1> d_out2_buffer(d_out2.data(), range<1>(out.at(0)*out.at(1)*out.at(2)));

    oneapi::mkl::dft::compute_backward(fftPlan, fft_buffer, d_out2_buffer);

    q->submit([&](sycl::handler& h) {
        auto src_acc = d_out2_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = output->get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(product3(out)), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    
}

buffer<double,1> createvals(buffer<double,1> symbol, int i, int j, int k, int dimx, int dimy, int dimz, int n, double c2, double invep0) {

    host_accessor<double, 1, access::mode::read> sym(symbol);

    double fmkx = sym[j]; //0-nf/2
    double fmky = sym[dimx+k]; //nf/2 - n
    double fmkz = sym[dimx+dimy+i]; // n - n
    double fcv = sym[(dimx+dimy+dimz) + i*dimx*dimy + j*dimy + k];
    double fsckv = sym[dimx+dimy+dimz +(dimx*dimy*dimz) + i*dimx*dimy + j*dimy + k];
    double fx1v = sym[dimx+dimy+dimz+(2*(dimx*dimy*dimz)) + i*dimx*dimy + j*dimy + k];
    double fx2v = sym[dimx+dimy+dimz+(3*(dimx*dimy*dimz)) + i*dimx*dimy + j*dimy + k];
    double fx3v = sym[dimx+dimy+dimz+(4*(dimx*dimy*dimz)) + i*dimx*dimy + j*dimy + k];

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

        buffer<double,1> dvals((double*)vals.data(), range<1>(2*vals.size()));
        return dvals;
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

    queue q{sycl::gpu_selector{},
                      sycl::property::queue::enable_profiling{}};

    WarpXconfig conf;
    int n = conf.n;
    int np = conf.np;
    int nf = conf.nf;
    int symFields = 8;
    std::vector<std::vector<int> > input_sizes = {{np, np, n},
                                           {np, n, np},
                                           {n, np, np},
                                           {n, n, np},
                                           {n, np, n},
                                           {np, n, n},
                                           {np, np, n},
                                           {np, n, np},
                                           {n, np, np},
                                           {np, np, np},
                                           {np, np, np}};  
    
    std::vector<std::vector<int> > output_sizes = {{np, np, n},
                                             {np, n, np},
                                             {n, np, np},
                                             {n, n, np},
                                             {n, np, n},
                                             {np, n, n}};

    size_t totalinputsize = 1;
    size_t totaloutputsize = 1;
    for(int i = 0; i < input_sizes.size(); i++) {
        totalinputsize += product3(input_sizes.at(i));
    }
     for(int i = 0; i < output_sizes.size(); i++) {
        totaloutputsize += product3(output_sizes.at(i));
    }
    
    std::vector<double> input(totalinputsize,1);
    std::vector<double> output(totaloutputsize,0);
    size_t totalsymbolsize = (nf/2)*n*n*(5*(nf/2)*n*n);
    std::vector<double> symbol(totalsymbolsize, 1);

    buffer<double,1> input_buffer(input.data(),  range<1>(totalinputsize));
    buffer<double,1> output_buffer(input.data(), range<1>(totaloutputsize));
    buffer<double,1> symbol_buffer(input.data(), range<1>(totalsymbolsize));

    std::vector<double> boxBig0(conf.inFields*n*n*n, 0);
    std::vector<double> boxBig1(conf.inFields*n*n*nf*2, 0);
    std::vector<double> boxBig2(conf.outFields*n*n*nf*2, 0);
    std::vector<double> boxBig3(conf.outFields*n*n*n, 0);
    buffer<double,1> boxBig0_buffer(boxBig0.data(), range<1>(conf.inFields*n*n*n));
    buffer<double,1> boxBig1_buffer(boxBig1.data(), range<1>(conf.inFields*n*n*nf*2));
    buffer<double,1> boxBig2_buffer(boxBig2.data(), range<1>(conf.outFields*n*n*nf*2));
    buffer<double,1> boxBig3_buffer(boxBig3.data(), range<1>(conf.outFields*n*n*n));
    

    Resample(&q, {n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, &boxBig0_buffer, &input_buffer, 0);//1

    Resample(&q, {n, n, n}, {np, np, n}, {0.0, -0.5, 0.0}, &boxBig0_buffer, &input_buffer, product3(input_sizes.at(0)));//2

    Resample(&q, {n, n, n}, {np, np, np}, {-0.5, 0.0, 0.0}, &boxBig0_buffer, &input_buffer, product3(input_sizes.at(1)) + product3(input_sizes.at(0)));//3

    Resample(&q, {n, n, n}, {n, n, np}, {-0.5, -0.5, 0.0}, &boxBig0_buffer, &input_buffer, product3(input_sizes.at(2)) + product3(input_sizes.at(1)) + product3(input_sizes.at(0)));//4

    Resample(&q, {n, n, n}, {n, np, n}, {-0.5, 0.0, -0.5}, &boxBig0_buffer, &input_buffer, product3(input_sizes.at(3)) + product3(input_sizes.at(2)) + product3(input_sizes.at(1)) + product3(input_sizes.at(0)));//5

    Resample(&q, {n, n, n}, {np, n, n}, {0.0, -0.5, -0.5}, &boxBig0_buffer, &input_buffer, product3(input_sizes.at(4)) + product3(input_sizes.at(3)) + product3(input_sizes.at(2)) + product3(input_sizes.at(1)) + product3(input_sizes.at(0)));//6

    Resample(&q, {n, n, n}, {np, np, n}, {0.0, 0.0, -0.5}, &boxBig0_buffer, &input_buffer, product3(input_sizes.at(5)) + product3(input_sizes.at(4)) + product3(input_sizes.at(3)) + product3(input_sizes.at(2)) + product3(input_sizes.at(1)) + product3(input_sizes.at(0)));//7

    Resample(&q, {n, n, n}, {np, n, np}, {0.0, -0.5, 0.0}, &boxBig0_buffer, &input_buffer, product3(input_sizes.at(6)) + product3(input_sizes.at(5)) + product3(input_sizes.at(4)) + product3(input_sizes.at(3)) + product3(input_sizes.at(2)) + product3(input_sizes.at(1)) + product3(input_sizes.at(0)));//8

    Resample(&q, {n, n, n}, {n, np, np}, {-0.5, 0.0, 0.0}, &boxBig0_buffer, &input_buffer, product3(input_sizes.at(7)) + product3(input_sizes.at(6)) + product3(input_sizes.at(5)) + product3(input_sizes.at(4)) + product3(input_sizes.at(3)) + product3(input_sizes.at(2)) + product3(input_sizes.at(1)) + product3(input_sizes.at(0)));//9

    Resample(&q, {n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, &boxBig0_buffer, &input_buffer, product3(input_sizes.at(8)) + product3(input_sizes.at(7)) + product3(input_sizes.at(6)) + product3(input_sizes.at(5)) + product3(input_sizes.at(4)) + product3(input_sizes.at(3)) + product3(input_sizes.at(2)) + product3(input_sizes.at(1)) + product3(input_sizes.at(0)));//10

    Resample(&q, {n, n, n}, {np, np, np}, {0.0, 0.0, 0.0}, &boxBig0_buffer, &input_buffer, product3(input_sizes.at(9)) + product3(input_sizes.at(8)) + product3(input_sizes.at(7)) + product3(input_sizes.at(6)) + product3(input_sizes.at(5)) + product3(input_sizes.at(4)) + product3(input_sizes.at(3)) + product3(input_sizes.at(2)) + product3(input_sizes.at(1)) + product3(input_sizes.at(0)));//11


    std::vector<double> fft_in(n*n*n);
    buffer<double,1> fft_in_buffer(fft_in.data(), range<1>(n*n*n));
    std::vector<double> fft_out((n*n*(nf/2+1))*2);
    buffer<double,1> fft_out_buffer(fft_out.data(), range<1>((n*n*(nf/2+1))*2));
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL> fftPlan({n, n, n});
    fftPlan.commit(q);

    int offset = 0;
    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig0_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = fft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_forward(fftPlan, fft_in_buffer, fft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = fft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig1_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig0_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = fft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_forward(fftPlan, fft_in_buffer, fft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = fft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig1_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig0_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = fft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_forward(fftPlan, fft_in_buffer, fft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = fft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig1_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig0_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = fft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_forward(fftPlan, fft_in_buffer, fft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = fft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig1_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig0_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = fft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_forward(fftPlan, fft_in_buffer, fft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = fft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig1_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig0_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = fft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_forward(fftPlan, fft_in_buffer, fft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = fft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig1_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig0_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = fft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_forward(fftPlan, fft_in_buffer, fft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = fft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig1_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig0_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = fft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_forward(fftPlan, fft_in_buffer, fft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = fft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig1_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig0_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = fft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_forward(fftPlan, fft_in_buffer, fft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = fft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig1_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig0_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = fft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_forward(fftPlan, fft_in_buffer, fft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = fft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig1_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig0_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = fft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_forward(fftPlan, fft_in_buffer, fft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = fft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig1_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    
    

    std::vector<int> rows{0,6,12,18,23,28,33};
    std::vector<int> cols{0,4,5,6,9,10,
                            1,3,5,7,9,10,
                            2,3,4,8,9,10,
                            1,2,3,7,8,
                            0,2,4,6,8,
                            0,1,5,6,7};
    buffer<int,1> drows(rows.data(), range<1>(rows.size()));
    buffer<int,1> dcols(cols.data(), range<1>(cols.size()));

    int length = rows.size() -1;
    for(int i = 0; i < nf/2; i++) {
        for(int j = 0; j < n; j++) {
            for(int k = 0; k < n; k++) {
                buffer<double,1> dvals = createvals(symbol_buffer, i, j, k, (nf/2), n, n, n, conf.c2, conf.invep0);
                int index = i*n*n + j*n + k;
                q.submit([&](sycl::handler& h){
                    auto row_acc = drows.get_access<sycl::access::mode::read>(h);
                    auto col_acc = drows.get_access<sycl::access::mode::read>(h);
                    auto val_acc = dvals.get_access<sycl::access::mode::read>(h);
                    auto output = boxBig2_buffer.get_access<sycl::access::mode::write>(h);
                    auto input = boxBig1_buffer.get_access<sycl::access::mode::read>(h);
                    h.parallel_for(sycl::range<1>(length), [=](sycl::id<1> idx) {
                        for(int i = idx; i < length; i++) {
                            double output_real = 0;
                            double output_imag = 0;
                            for(int j = row_acc[i]; j < row_acc[i+1]; j++) {
                                output_real += input[col_acc[j]*(n*n*(nf/2)) + 2*index] * val_acc[2*j] - input[col_acc[j]*(n*n*(nf/2)) + 2*index+1] * val_acc[2*j+1]; 
                                output_imag += input[col_acc[j]*(n*n*(nf/2)) +2*index] * val_acc[2*j+1] + input[col_acc[j]*(n*n*(nf/2)) +2*index+1] * val_acc[2*j];
                            }
                            output[i*(n*n*(nf/2)) +2*index] = output_real;
                            output[i*(n*n*(nf/2))+2*index+1] = output_imag;
                        }
                    });
                }).wait();
            }
        }
    }

    std::vector<double> ifft_in(n*n*(n/2+1)*2);
    buffer<double,1> ifft_in_buffer(ifft_in.data(), range<1>(n*n*(nf/2)*2));
    std::vector<double> ifft_out(n*n*n*2);
    buffer<double,1> ifft_out_buffer(ifft_out.data(), range<1>(n*n*n*2));
    offset = 0;
    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig3_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = ifft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_backward(fftPlan, ifft_in_buffer, ifft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = ifft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig3_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig3_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = ifft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_backward(fftPlan, ifft_in_buffer, ifft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = ifft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig3_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig3_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = ifft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_backward(fftPlan, ifft_in_buffer, ifft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = ifft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig3_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig3_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = ifft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_backward(fftPlan, ifft_in_buffer, ifft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = ifft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig3_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig3_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = ifft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_backward(fftPlan, ifft_in_buffer, ifft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = ifft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig3_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    offset += (n*n*n);

    q.submit([&](sycl::handler& h) {
        auto src_acc = boxBig3_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = ifft_in_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx] = src_acc[idx + offset];  // Copy to the second half of destination_buffer
        });
    }).wait();
    oneapi::mkl::dft::compute_backward(fftPlan, ifft_in_buffer, ifft_out_buffer);
    q.submit([&](sycl::handler& h) {
        auto src_acc = ifft_out_buffer.get_access<sycl::access::mode::read>(h);
        auto dest_acc = boxBig3_buffer.get_access<sycl::access::mode::write>(h);
    
        h.parallel_for(sycl::range<1>(n*n*n), [=](sycl::id<1> idx) {
            dest_acc[idx + offset] = src_acc[idx];  // Copy to the second half of destination_buffer
        });
    }).wait();
    

    
    Resample(&q, {np, np, n}, {n, n, n}, {0.0, 0.0, 0.5}, &output_buffer, &boxBig3_buffer, 0);
    Resample(&q, {np, n, np}, {n, n, n}, {0.0, 0.5, 0.0}, &output_buffer, &boxBig3_buffer, product3(output_sizes.at(0)));
    Resample(&q, {n, np, np}, {n, n, n}, {0.5, 0.0, 0.0}, &output_buffer, &boxBig3_buffer, product3(output_sizes.at(1)) + product3(output_sizes.at(0)));
  
    Resample(&q, {n, n, np}, {n, n, n}, {0.5, 0.5, 0.0}, &output_buffer, &boxBig3_buffer,
product3(output_sizes.at(2)) + product3(output_sizes.at(1)) + product3(output_sizes.at(0)));

    Resample(&q, {n, np, n}, {n, n, n}, {0.5, 0.0, 0.5}, &output_buffer, &boxBig3_buffer, product3(output_sizes.at(3)) + product3(output_sizes.at(2)) + product3(output_sizes.at(1)) + product3(output_sizes.at(0)));

    Resample(&q, {np, n, n}, {n, n, n}, {0.0, 0.5, 0.5}, &output_buffer, &boxBig3_buffer, product3(output_sizes.at(4)) + product3(output_sizes.at(3)) + product3(output_sizes.at(2)) + product3(output_sizes.at(1)) + product3(output_sizes.at(0)));
    
    

    return 0;
}