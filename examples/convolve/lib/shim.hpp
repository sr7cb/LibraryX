#include <fftw3.h>
#include <algorithm>
#include <functional>
#include <tuple>
#include <optional>
#include <stdexcept>
#include <any>
#include <iostream>
#include <vector>
#include "interface.hpp"

// std::map<std::string, std::function> function_lookup_table;
// std::vector<std::function> call_trace;

std::vector<double>::iterator convolve_output_begin;

/*

FunctionRecorder is a class which captures a function call.
First, it binds the arguments with the function reference. 
    std::forward is added, so that both r-value and l-value references are passed as they come in.
    invoke_result_t is able to infer the return type of the bound function.
Second, the provided params are stored a tuple for use later.

*/
class FunctionRecorder {
public:
    // Vardic templates being used to infer the type of each argument.
    template<typename F, typename... Args>
    void addFunction(F&& f, Args&&... args) { // The && allows us to accept both r-value and l-value references
        auto boundFunc = std::bind(std::forward<F>(f), std::forward<Args>(args)...); // bind the function with the provided arguments.
        _f = [boundFunc]() {
            return std::make_any<std::invoke_result_t<F, Args...>>(boundFunc()); // Make a simple lambda from the bound function, lambda is easier to call and store.
        };

        params = std::make_tuple(std::forward<Args>(args)...); // Save the arguments
    }

    /*

    This method is required because the bound function would return just void and must be handled accordingly.
    */
    template<typename F, typename... Args>
    void addFunctionVoid(F&& f, Args&&... args) { // For void functions, the std::any should be empty.
        auto boundFunc = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
        _f = [boundFunc]() {
            boundFunc();
            return std::any();
        };

        params = std::make_tuple(std::forward<Args>(args)...);
    }

    /*

        invokes the bound function and returns the value.
    */
    std::optional<std::any> invoke() {
        if (_result) {
            return _result.value();
        }

        _result = _f();
        return _result;
    }

    void reset() {
        _result.reset();
    }

private:
    std::function<std::any()> _f;
    std::tuple<std::any> params;
    std::optional<std::any> _result;
};


enum string_code {
    fftw_plan_dft_r2c_1d_switch,
    fftw_plan_dft_c2r_1d_switch,
    fftw_execute_switch,
    copy_switch,
    copy2_switch,
    transform_switch,
    mone
};

string_code hashit(std::string const& inString) {
    if(inString == "fftw_plan_dft_r2c_1d") return fftw_plan_dft_r2c_1d_switch;
    if(inString == "fftw_plan_dft_c2r_1d") return fftw_plan_dft_c2r_1d_switch;
    if(inString == "fftw_execute") return fftw_execute_switch;
    if(inString == "copy") return copy_switch;
    if(inString == "copy2") return copy2_switch;
    if(inString == "transform") return transform_switch;
    return mone;
}


// This contains the list of all recorded procedure calls.
class Trace {
public:
    std::vector<std::any> inputs;
    std::vector<std::any> outputs;

    Trace(){
    }

    template<typename F, typename... Args>
    void recordProcedure(F&& f, Args&&... args) {
        auto rec = FunctionRecorder();
        rec.addFunction(f, args...);
        recorded_procedures.emplace_back(rec);
    }

    // This method is supposed to be used when the arguments have inputs or outputs, and we want to capture them
    // Ideally there should be a better way to identify the inputs and outputs. But this is what it is right now.
    template<typename F, typename... Args>
    void recordProcedureAndTrackIO(F&& f, Args&&... args) {
        auto rec = FunctionRecorder();
        rec.addFunction(f, args...);
        recorded_procedures.emplace_back(rec);

        copy_calls_count++;
        if (copy_calls_count < 3) {
            auto [I1, I2, O3] = std::make_tuple(args...);
            inputs.emplace_back(std::vector(I1, I2));
        } else if (copy_calls_count == 3) {
            auto [I1, I2, O3] = std::make_tuple(args...);
            outputs.emplace_back(std::vector(O3, O3 + (I2 - I1))); // TODO: The output is not being captured correctly.
            // outputs.emplace_back(std::vector(I1, I2));

            // delayedExecute();

            // std::vector<double> output = std::any_cast<std::vector<double>>(outputs[0]);

            // std::cout << "array = ";
            // for(auto i: output)
            //     std::cout << i << ", ";
            // std::cout << "\n";

        }
    }

    template<typename F, typename... Args>
    void recordProcedureVoid(F&& f, Args&&... args) {
        auto rec = FunctionRecorder();
        rec.addFunctionVoid(f, args...);
        recorded_procedures.emplace_back(rec);
    }

    template<typename T>
    void addInputArg(T& a) {
        inputs.push_back(a);
    }

    template<typename T>
    void addOutputArg(T& a) {
       outputs.push_back(a);
    }

    void setSpiralInfo(std::string text) {
        spiral_info.push_back(text);
    }

    void printToSpiral() {
        std::cout << "transform := let(symvar := var(\"sym\", TPtr(TReal)),TFCall(";
        for(int i = spiral_info.size()-1; i > 0; i--) {
            std::cout << spiral_info.at(i) << " * ";
        }
        std::cout << spiral_info.at(0) << ",";
        std::cout << "rec(fname := name, params := [symvar])));" << std::endl;
    }

    /*
    Executes the bound methods
    TODO: Track the return variables. And automatically infer the parameters, then use a for loop to execute. 
    */
    void delayedExecute() {
        auto forward_1 = recorded_procedures[0].invoke().value();
        auto forward_2 = recorded_procedures[1].invoke().value();
        auto backward = recorded_procedures[2].invoke().value();

        recorded_procedures[3].invoke();
        recorded_procedures[4].invoke();
        ::fftw_execute(std::any_cast<fftw_plan>(forward_1));
        ::fftw_execute(std::any_cast<fftw_plan>(forward_2));
        recorded_procedures[7].invoke();
        ::fftw_execute(std::any_cast<fftw_plan>(backward));
        recorded_procedures[9].invoke();
    }
private:
    std::vector<FunctionRecorder> recorded_procedures;
    std::vector<std::string> spiral_info;
    int copy_calls_count = 0;
};

Trace dag;

namespace spiral_fftw {

    // Taken from https://www.fftw.org/fftw3_doc/FFTW-Reference.html
    fftw_plan fftw_plan_dft_r2c_1d(int n0, double *in, fftw_complex *out,
     unsigned flags) {
        std::cout << "We are inside spiral_fftw::fftw_plan_dft_r2c_1d" << std::endl;

        dag.recordProcedure(&::fftw_plan_dft_r2c_1d, n0, in, out, flags);
        std::string spiral = "DFT(" + std::to_string(n0) + ", -1)";
        dag.setSpiralInfo(spiral);
        // return ::fftw_plan_dft_r2c_1d(n0, in, out, flags);
        return NULL;
    }

    fftw_plan fftw_plan_dft_c2r_1d(
        int n0, fftw_complex *in, double *out, unsigned flags) {
        std::cout << "We are inside spiral_fftw::fftw_plan_dft_c2r_1d" << std::endl;
        
        dag.recordProcedure(&::fftw_plan_dft_c2r_1d, n0, in, out, flags);
        std::string spiral = "DFT(" + std::to_string(n0) + ", 1)";
        dag.setSpiralInfo(spiral);
        // return ::fftw_plan_dft_c2r_1d(n0, in, out, flags);
        return NULL;
    }

    void fftw_execute(const fftw_plan plan){
        std::cout << "We are inside spiral_fftw::fftw_execute" << std::endl;
        dag.recordProcedureVoid(&::fftw_execute, plan);

        // ::fftw_execute(plan);
    }

#if defined(HP_USING_STD) && !defined(HP)
    // Taken from https://en.cppreference.com/w/cpp/algorithm/transform
    template< class InputIt1, class InputIt2,
          class OutputIt, class BinaryOperation >
    OutputIt transform( InputIt1 first1, InputIt1 last1, InputIt2 first2,
                    OutputIt d_first, BinaryOperation binary_op ) {
        std::cout << "We are inside spiral_fftw::transform" << std::endl;
        dag.recordProcedure(&std::transform<decltype(first1), decltype(first2), decltype(d_first), decltype(binary_op)>, first1, last1, first2, d_first, binary_op);
        std::string spiral = "Diag(FDataOfs(symvar," + std::to_string(last1-first1) + ", 0))";
        dag.setSpiralInfo(spiral);
        // return std::transform(first1, last1, first2, d_first, binary_op);
        return d_first;
    }

    // Taken from https://en.cppreference.com/w/cpp/algorithm/copy
    template< class InputIt, class OutputIt >
    OutputIt copy( InputIt first, InputIt last, OutputIt d_first ) {
        std::cout << "We are inside spiral_fftw::copy" << std::endl;
        convolve_output_begin = d_first;
        // dag.recordProcedureAndTrackIO(&std::copy<decltype(first), decltype(d_first)>, first, last, d_first);
        dag.recordProcedure(&std::copy<decltype(first), decltype(d_first)>, first, last, d_first);
        // return std::copy(first, last, d_first);
        return d_first;
    }
#endif
}

#if defined(HP) && !defined(HP_USING_STD)
namespace std {
    namespace spiral_fftw {
        // Taken from https://en.cppreference.com/w/cpp/algorithm/transform
        template< class InputIt1, class InputIt2,
              class OutputIt, class BinaryOperation >
        OutputIt transform( InputIt1 first1, InputIt1 last1, InputIt2 first2,
                        OutputIt d_first, BinaryOperation binary_op ) {
            std::cout << "We are inside spiral_fftw::transform" << std::endl;
            dag.recordProcedure(&std::transform<decltype(first1), decltype(first2), decltype(d_first), decltype(binary_op)>, first1, last1, first2, d_first, binary_op);
            std::string spiral = "Diag(FDataOfs(symvar," + std::to_string(last1-first1) + ", 0))";
            dag.setSpiralInfo(spiral);
            // return std::transform(first1, last1, first2, d_first, binary_op);
            return d_first;
        }

        // Taken from https://en.cppreference.com/w/cpp/algorithm/copy
        template< class InputIt, class OutputIt >
        OutputIt copy( InputIt first, InputIt last, OutputIt d_first ) {
            std::cout << "We are inside spiral_fftw::copy" << std::endl;
            convolve_output_begin = d_first;
            // dag.recordProcedureAndTrackIO(&std::copy<decltype(first), decltype(d_first)>, first, last, d_first);
            dag.recordProcedure(&std::copy<decltype(first), decltype(d_first)>, first, last, d_first);
            // return std::copy(first, last, d_first);
            return d_first;
        }
    }
}
#endif

#if defined(HP) || defined(HP_USING_STD)
#define fftw_plan_dft_r2c_1d spiral_fftw::fftw_plan_dft_r2c_1d
#define fftw_plan_dft_c2r_1d spiral_fftw::fftw_plan_dft_c2r_1d
#define fftw_execute spiral_fftw::fftw_execute
#define transform(a,b,c,d,e) spiral_fftw::transform(a,b,c,d,e)
#define copy(a,b,c) spiral_fftw::copy(a,b,c)
#endif

/*
    This class is used as a shim for Vector.
*/

#if defined(HP) || defined(HP_USING_STD)
namespace std {
    template <typename T>
    class LazyVector : public std::vector<T> {
        bool lazy = true;
    public:
        using std::vector<T>::vector;
        T& operator[](size_t index) {
            if (lazy and this->begin() == convolve_output_begin) {
                lazy = false; // compute and set lazy as false
                cout << "I am in LazyVector& operator[]" << endl;
                class ConvProblem: public FFTXProblem {
                public:
                    using FFTXProblem::FFTXProblem;
                    void randomProblemInstance() {
                    }
                    void semantics() {
                        std::cout << "name := \"" << name << "_spiral" << "\";" << std::endl;
                        dag.printToSpiral();
                    }
                };
                // // int i = 0;
                // // std::LazyVector<double> test_before = std::any_cast<std::LazyVector<double>>(dag.inputs.at(0));
                // // std::cout << "Output size is " << dag.outputs.size() << std::endl;
                // double * cp_output = new double[4]();
                // double * cp_input1 = new double[4]();
                // double * cp_input2 = new double[4]();
                // // for(int i = 0; i < std::any_cast<std::LazyVector<double>&>(dag.outputs.at(0)).size(); i++) {
                // //     cp_output[i] = (std::any_cast<std::LazyVector<double>&>(dag.outputs.at(0)).at(i));
                // //     std::cout << cp_output[i] << std::endl;
                // // }
                // // std::cout << std::endl;
                // // for(int i = 0; i < std::any_cast<std::LazyVector<double>&>(dag.inputs.at(0)).size(); i++) {
                // //     cp_input1[i] = (std::any_cast<std::LazyVector<double>&>(dag.inputs.at(0)).at(i));
                // //     std::cout << cp_input1[i] << std::endl;
                // // }
                // //  std::cout << std::endl;
                // // for(int i = 0; i < std::any_cast<std::LazyVector<double>&>(dag.inputs.at(1)).size(); i++) {
                // //     cp_input2[i] = (std::any_cast<std::LazyVector<double>&>(dag.inputs.at(1)).at(i));
                // //     std::cout << cp_input2[i] << std::endl;
                // // }
                // // std::cout << std::endl;
                // std::vector<void*> args{(void*)cp_output, (void*)cp_input1, (void*)cp_input2};
                // // for(int i = 0; i < dag.outputs.size(); i++) {
                // //     // std::LazyVector<double> test = std::any_cast<std::LazyVector<double>>(dag.outputs.at(i));
                // //     // args.push_back((void*)test.data());
                    
                // //     args.push_back((void*)std::any_cast<std::LazyVector<double>&>(dag.outputs.at(i)).data());

                // // }
                // // std::cout <<"Input size is " << dag.inputs.size() << std::endl;
                // // for(int i = 0; i < dag.inputs.size(); i++) {
                // //     // std::LazyVector<double> * test = &std::any_cast<std::LazyVector<double>>(dag.inputs.at(i));
                // //     args.push_back((void*)std::any_cast<std::LazyVector<double>&>(dag.inputs.at(i)).data());

                // // }
                // // std::cout << ((double*)args.at(1))[0] << std::endl;
                // // std::cout << ((double*)args.at(1))[1] << std::endl;

                // std::cout << (int)std::any_cast<std::LazyVector<double>>(dag.inputs.at(0)).size() << std::endl;
                // std::cout << (int)std::any_cast<std::LazyVector<double>>(dag.inputs.at(1)).size() << std::endl;
                // std::cout << (int)std::any_cast<std::LazyVector<double>>(dag.outputs.at(0)).size() << std::endl;
                // int a = (int)std::any_cast<std::LazyVector<double>>(dag.inputs.at(0)).size();
                // int b = (int)std::any_cast<std::LazyVector<double>>(dag.inputs.at(1)).size();
                // int c = (int)std::any_cast<std::LazyVector<double>>(dag.outputs.at(0)).size();
                // std::vector<int> sizes{a, 
                //                         b,
                //                         c};
                // ConvProblem cp(args, sizes, "conv");
                // cp.transformSPIRAL();
                // delete[] cp_output;
                // // delete[] cp_input1;
                // delete[] cp_input2;
                // dag.delayedExecute();
            }

            return vector<T>::operator[](index);
        }

        const T& operator[](size_t index) const { 
            if (lazy and this->begin() == convolve_output_begin) {
                lazy = false; // compute and set lazy as false
                cout << "I am in LazyVector& operator[]" << endl;
                dag.delayedExecute();
            }
            return vector<T>::operator[](index);
        }
    };
}

#define vector LazyVector
#endif