#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>

void checkErr(cl_int err, const std::string &name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::string readKernelFile(const std::string &filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open kernel file: " + filePath);
    }

    std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return source;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel_file_path>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string kernelFilePath = argv[1];
    cl_int err;

    // Read kernel source code from file
    std::string kernelSource = readKernelFile(kernelFilePath);

    // Get all available platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found.");
    }

    // Use the first platform and get its devices
    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL devices found.");
    }

    // Use the first device
    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkErr(err, "CommandQueue");

    // Build the kernel program
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});
    cl::Program program(context, sources);

    err = program.build({device});
    if (err != CL_SUCCESS) {
        std::cerr << "Build Error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        checkErr(err, "Program::build");
    }

    // Create the kernel (assuming kernel name is "kernel_func")
    cl::Kernel kernel(program, "add", &err);
    checkErr(err, "Kernel");


    const size_t bufferSize = 1024;
    std::vector<float> inputData(bufferSize, 1.0f); // Initialize with 1.0
    std::vector<float> inputData2(bufferSize, 1.0f); // Initialize with 1.0
    std::vector<float> outputData(bufferSize, 0.0f); // For storing results 

    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(float) * bufferSize, inputData.data(), &err);
    checkErr(err, "Input Buffer Creation"); 
    cl::Buffer inputBuffer2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(float) * bufferSize, inputData2.data(), &err);      
    checkErr(err, "Input Buffer 2 Creation"); 
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, 
                        sizeof(float) * bufferSize, nullptr, &err);
    checkErr(err, "Output Buffer Creation");                                

    kernel.setArg(0, bufferSize);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, inputBuffer);
    kernel.setArg(3, inputBuffer2);
    // // Set up a dummy buffer for kernel arguments (if required by your kernel)
    
    // cl::Buffer buffer(context, CL_MEM_READ_WRITE, bufferSize, nullptr, &err);
    // checkErr(err, "Buffer");
    // kernel.setArg(0, buffer);

    // Launch the kernel
    cl::Event event;
    err = queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(bufferSize),
        cl::NDRange(64),
        nullptr,
        &event
    );
    checkErr(err, "enqueueNDRangeKernel");

    // Wait for the kernel to finish and measure time
    event.wait();

    cl_ulong timeStart, timeEnd;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);

    double nanoSeconds = timeEnd - timeStart;
    std::cout << "Kernel execution time: " << nanoSeconds / 1e6 << " ms" << std::endl;

    return EXIT_SUCCESS;
}
