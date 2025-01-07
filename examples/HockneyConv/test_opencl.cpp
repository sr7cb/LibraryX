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
    // for (size_t i = 0; i < platforms.size(); ++i) {
    //         std::cout << "Platform " << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
    //         std::cout << "    Vendor: " << platforms[i].getInfo<CL_PLATFORM_VENDOR>() << std::endl;
    //         std::cout << "    Version: " << platforms[i].getInfo<CL_PLATFORM_VERSION>() << std::endl;
    // }
    // Use the first platform and get its devices
    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL devices found.");
    }
    // for (size_t j = 0; j < devices.size(); ++j) {
    //             std::cout << "    Device " << j << ": " << devices[j].getInfo<CL_DEVICE_NAME>() << std::endl;
    //             std::cout << "        Type: " << (devices[j].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU ? "GPU" : 
    //                                              devices[j].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU ? "CPU" :
    //                                              "Other") << std::endl;
    //             std::cout << "        Compute Units: " << devices[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    //             std::cout << "        Global Memory: " << devices[j].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << " MB" << std::endl;
    //             std::cout << "        Local Memory: " << devices[j].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KB" << std::endl;
    //             std::cout << "        Max Workgroup Size: " << devices[j].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
    //         }
    // Use the first device
    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkErr(err, "CommandQueue");

    size_t maxWorkGroupSize;
size_t localMemSize;
size_t maxGlobalSize[3];
device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSize);
device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, maxGlobalSize);
std::cout << "Max Work Group Size: " << maxWorkGroupSize << std::endl;
std::cout << "Local Memory Size: " << localMemSize << " bytes" << std::endl;
std::cout << "Max Global Size: {" << maxGlobalSize[0] << ", " << maxGlobalSize[1] << ", " << maxGlobalSize[2] << "}" << std::endl;


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
    cl::Kernel kernel0(program, "ker_hockneyconv_spiral0", &err);
    checkErr(err, "Kernel0");
    cl::Kernel kernel1(program, "ker_hockneyconv_spiral1", &err);
    checkErr(err, "Kernel1");
    cl::Kernel kernel2(program, "ker_hockneyconv_spiral2", &err);
    checkErr(err, "Kernel2");
    cl::Kernel kernel3(program, "ker_hockneyconv_spiral3", &err);
    checkErr(err, "Kernel3");
    cl::Kernel kernel4(program, "ker_hockneyconv_spiral4", &err);
    checkErr(err, "Kernel4");
    cl::Kernel kernel5(program, "ker_hockneyconv_spiral5", &err);
    checkErr(err, "Kernel5");


    const size_t bufferSize = 32*32*128;
    const size_t bufferSize2 = 64*64*(256/2+1);
    std::vector<double> X(bufferSize, 1.0); // Initialize with 1.0
    std::vector<double> X2(bufferSize2, 1.0); // Initialize with 1.0
    std::vector<double> Y(bufferSize, 0.0); // Initialize with 1.0
    std::vector<double> P1(1056768, 0.0); // For storing results 
    std::vector<double> P2(528384, 0.0);
    int localsize1 = 1156;
    int localsize2 = 3072;
    int localsize3 = 578;

    cl::Buffer d_X(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(double) * bufferSize, X.data(), &err);
    checkErr(err, "X Creation");
    cl::Buffer d_X2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                       sizeof(double) * bufferSize2, X2.data(), &err);
    checkErr(err, "X Creation");  
    cl::Buffer d_Y(context, CL_MEM_WRITE_ONLY, 
                       sizeof(double) * bufferSize, nullptr, &err);      
    checkErr(err, "Y Creation"); 
    cl::Buffer d_p1(context, CL_MEM_READ_WRITE, 
                        sizeof(double) * 1056768, nullptr, &err);
    checkErr(err, "P1r Creation");
    cl::Buffer d_p2(context, CL_MEM_READ_WRITE, 
                        sizeof(double) * 528384, nullptr, &err);
    checkErr(err, "P2r Creation");                                
                                

    kernel0.setArg(0, d_X);
    kernel0.setArg(1, d_p1);
    kernel0.setArg(2, cl::Local(localsize1 *sizeof(double)));

    size_t globalWorkSize[3] = {4*256, 1, 1};
    size_t localWorkSize[3] = {4, 1, 1};
    cl::NDRange global(globalWorkSize[0], globalWorkSize[1], globalWorkSize[2]);
    cl::NDRange local(localWorkSize[0], localWorkSize[1], localWorkSize[2]);
    // Launch the kernel
    cl::Event event;
    err = queue.enqueueNDRangeKernel(
        kernel0,
        cl::NullRange,
        global,
        local,
        nullptr,
        &event
    );
    checkErr(err, "enqueueNDRangeKernel0");

    kernel1.setArg(0, d_p1);
    kernel1.setArg(1, d_p2);
    kernel1.setArg(2, cl::Local(localsize2 *sizeof(double)));

    size_t globalWorkSize2[3] = {192*172, 1, 1};
    size_t localWorkSize2[3] = {192, 1, 1};
    cl::NDRange global1(globalWorkSize2[0], globalWorkSize2[1], globalWorkSize2[2]);
    cl::NDRange local1(localWorkSize2[0], localWorkSize2[1], localWorkSize2[2]);
    // Launch the kernel
    cl::Event event1;
    err = queue.enqueueNDRangeKernel(
        kernel1,
        cl::NullRange,
        global1,
        local1,
        nullptr,
        &event1
    );
    checkErr(err, "enqueueNDRangeKernel1");

    kernel2.setArg(0, d_X2);
    kernel2.setArg(1, d_p1);
    kernel2.setArg(2, d_p2);
    kernel2.setArg(3, cl::Local(localsize2 *sizeof(double)));

    size_t globalWorkSize3[3] = {344*192, 1, 1};
    size_t localWorkSize3[3] = {192, 1, 1};
    cl::NDRange global2(globalWorkSize3[0], globalWorkSize3[1], globalWorkSize3[2]);
    cl::NDRange local2(localWorkSize3[0], localWorkSize3[1], localWorkSize3[2]);
    // Launch the kernel
    cl::Event event2;
    err = queue.enqueueNDRangeKernel(
        kernel2,
        cl::NullRange,
        global2,
        local2,
        nullptr,
        &event2
    );
    checkErr(err, "enqueueNDRangeKernel2");

    kernel3.setArg(0, d_p1);
    kernel3.setArg(1, d_p2);
    kernel3.setArg(2, cl::Local(localsize2 *sizeof(double)));

    size_t globalWorkSize4[3] = {344*192, 1, 1};
    size_t localWorkSize4[3] = {192, 1, 1};
    cl::NDRange global3(globalWorkSize4[0], globalWorkSize4[1], globalWorkSize4[2]);
    cl::NDRange local3(localWorkSize4[0], localWorkSize4[1], localWorkSize4[2]);
    // Launch the kernel
    cl::Event event3;
    err = queue.enqueueNDRangeKernel(
        kernel3,
        cl::NullRange,
        global3,
        local3,
        nullptr,
        &event3
    );
    checkErr(err, "enqueueNDRangeKernel3");

    kernel4.setArg(0, d_p1);
    kernel4.setArg(1, d_p2);
    kernel4.setArg(2, cl::Local(localsize2 *sizeof(double)));

    size_t globalWorkSize5[3] = {192*172, 1, 1};
    size_t localWorkSize5[3] = {192, 1, 1};
    cl::NDRange global4(globalWorkSize5[0], globalWorkSize5[1], globalWorkSize5[2]);
    cl::NDRange local4(localWorkSize5[0], localWorkSize5[1], localWorkSize5[2]);
    // Launch the kernel
    cl::Event event4;
    err = queue.enqueueNDRangeKernel(
        kernel4,
        cl::NullRange,
        global4,
        local4,
        nullptr,
        &event4
    );
    checkErr(err, "enqueueNDRangeKernel4");

    kernel5.setArg(0, d_Y);
    kernel5.setArg(1, d_p1);
    kernel5.setArg(2, cl::Local(localsize1 *sizeof(double)));

    size_t globalWorkSize6[3] = {2*512, 1, 1};
    size_t localWorkSize6[3] = {2, 1, 1};
    cl::NDRange global5(globalWorkSize6[0], globalWorkSize6[1], globalWorkSize6[2]);
    cl::NDRange local5(localWorkSize6[0], localWorkSize6[1], localWorkSize6[2]);
    // Launch the kernel
    cl::Event event5;
    err = queue.enqueueNDRangeKernel(
        kernel5,
        cl::NullRange,
        global5,
        local5,
        nullptr,
        &event5
    );
    checkErr(err, "enqueueNDRangeKernel5");

    // Wait for the kernel to finish and measure time
    event.wait();
    event1.wait();
    event2.wait();
    event3.wait();
    event4.wait();
    event5.wait();
    
    cl_ulong timeStart, timeEnd;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    event5.getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);

    double nanoSeconds = timeEnd - timeStart;
    std::cout << "Kernel execution time: " << nanoSeconds / 1e6 << " ms" << std::endl;

    queue.finish();

    // err = clEnqueueReadBuffer(queue, d_Y, CL_TRUE, 0, sizeof(double) * bufferSize, Y.data(), 0, NULL, NULL);
    err = queue.enqueueReadBuffer(d_Y, CL_TRUE, 0,
                              sizeof(double) * bufferSize, Y.data());
    checkErr(err, "Reading Output Buffer");

    // Print the output to the screen
    std::cout << "Kernel output:" << std::endl;
    for (size_t i = 0; i < 10; ++i) {
        std::cout << Y[i] << " ";
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
