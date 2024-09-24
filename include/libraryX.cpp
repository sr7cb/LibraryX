#include <iostream>
#include <vector>
#include "cudabackend.hpp"
#include "interface.hpp"
#include "libraryX.hpp"

class InternalProblem: public FFTXProblem {
    public:
        using FFTXProblem::FFTXProblem;
        void semantics() {  
            std::cout << script << std::endl;
        }
        void setString(std::string s){
            script = s;
        }
    private:
        std::string script; 
};

float (Executor::*generateDynamicCode(const char * script))(void*&)  {
    InternalProblem ip;
    std::string iscript{script};
    ip.setString(iscript);
    return ip.generateFunc();
}
