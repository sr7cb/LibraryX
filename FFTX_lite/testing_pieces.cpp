#include <iostream>
#include <complex>
#include <vector>

struct WarpXconfig{
    int a = 1;
    int b = 2;
    int c = 3; 
}config;

void myfunc(const std::vector<int>& t = {1, 2, 3}) {
    // Process the vector elements
    for (int element : t) {
        // Do something with element
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Call the function with the default vector
    myfunc();

    // Call the function with a different vector
    myfunc({4, 5, 6});

    return 0;
}


// int main() {
//     double a = 2;
//     double c = .5;
//     std::vector<std::complex<double>> v{std::complex<double>(0, a/c)};
//     std::cout << v.at(0) << std::endl;

//     WarpXconfig conf;
//     std::vector<void*> args{(void*)&conf};
//     std::cout << (*(WarpXconfig*)args.at(0)).a << std::endl;
//     return 0;
// }