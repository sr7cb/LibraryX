#include <iostream>
#include <vector>
#include <string>
#include "cpp_header.h"
// #include "interface.hpp"
// #include "cpubackend.hpp"

class test{
    public:
    std::string local;
    void dummyFunc();
    test(std::string s):local(s){} 
};

void test::dummyFunc() {
    std::cout << "hello from dummyFunc with str " << local << std::endl;
}

void(*) generateCode(const char * str) {
    test(str);
    //std::cout << str << std::endl;
    return test.dummyFunc();
}