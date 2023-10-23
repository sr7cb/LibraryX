#include <functional>
#include "Proto.H"
#pragma once
#define SHOW(a) std::cout << #a << (a) << std::endl
namespace Operator {
    #define NUMCOMPS DIM+2
    // template <typename T, typename I>
    // spiral_fftw::forall<>() {
    //     std::cout << "We are inside spiral_fftw::fftw_plan_dft_r2c_1d" << std::endl;
    // }
    namespace spiral{
        template <typename T>
        Proto::BoxData<T, NUMCOMPS> deconvolve(const BoxData<T, NUMCOMPS>&  a_U) {
            std::cout << "We are inside deconvolve" << std::endl;
            Proto::BoxData<T, NUMCOMPS> dummy(Proto::Box::Cube(1));
            return dummy;
        }
        template <typename T>
        Proto::BoxData<double, NUMCOMPS> _convolve(BoxData<T, NUMCOMPS>&  input2, BoxData<T, NUMCOMPS>&  input1) {
            std::cout << "We are inside convolve" << std::endl;
            Proto::BoxData<T, NUMCOMPS> dummy(Proto::Box::Cube(1));
            return dummy;
        }

        template <typename T>
        Proto::BoxData<double, NUMCOMPS> deconvolveFace(BoxData<T, NUMCOMPS>&  input2, int direction) {
            std::cout << "We are inside deconvolveFace" << std::endl;
            Proto::BoxData<T, NUMCOMPS> dummy(Proto::Box::Cube(1));
            return dummy;
        }


    }

}

// namespace Stencil {
//     namespace spiral{
//         template<typename T>
//         Proto::BoxData<double, NUMCOMPS> _CellToFaceH(int dir)
//     }
// }

template<typename T, unsigned int C=1,
typename Func, typename... Srcs>
Proto::BoxData<T, C> spiral_forall(const Func& a_F, Srcs&&... a_srcs){
std::cout << "We are inside forall" << typeid(a_F).name() << std::endl;
Proto::BoxData<T, C> dummy(Proto::Box::Cube(1));
return dummy;
}

// #define forall spiral_forall
// #define deconvolve spiral::deconvolve
// #define _convolve spiral::_convolve
// #define deconvolveFace spiral::deconvolveFace

namespace Proto {
    template<typename T, unsigned int C=1, MemType MEM=MEMTYPE_DEFAULT,
        unsigned char D=1, unsigned char E=1>
    class myVar
    {
        public:


#ifdef PROTO_ACCEL
        __device__
        inline T& getValDevice(unsigned int a_c, unsigned char a_d = 0, unsigned char a_e = 0)
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            int idy = blockIdx.y;
    #if DIM == 3
            int idz = blockIdx.z;
            return (m_ptrs[a_c + C*a_d + C*D*a_e])[idx + boxDimX * (idy + idz * boxDimY)];
    #else
            return (m_ptrs[a_c + C*a_d + C*D*a_e])[idx + idy * boxDimX];
    #endif
        }

    #ifdef PROTO_HIP
        __host__
        inline T& getValDevice(unsigned int a_c, unsigned char a_d = 0, unsigned char a_e = 0)
        {
            return *(m_ptrs[a_c + C*a_d + C*D*a_e]);
        }
    #endif
#else
        inline T& getValDevice(unsigned int a_c, unsigned char a_d = 0, unsigned char a_e = 0)
        {
            return *(m_ptrs[a_c + C*a_d + C*D*a_e]);
        }
#endif
        /// Pointwise Accessor
        /**
          Access component (c,d,e) of the <code>BoxData<T,C,MEM,D,E></code> 
          associated with *this.

          \param a_c   First component index
          \param a_d   Second component index (default: 0)
          \param a_e   Third component index  (default: 0)
        */
        ACCEL_DECORATION
        __attribute__((always_inline))
        T& operator()(unsigned int a_c, unsigned char a_d = 0, unsigned char a_e = 0)  
        {
            PROTO_ASSERT(a_c < C, "Var::operator() | Error: index out of bounds");
            PROTO_ASSERT(a_d < D, "Var::operator() | Error: index out of bounds");
            PROTO_ASSERT(a_e < E, "Var::operator() | Error: index out of bounds");
#ifdef PROTO_ACCEL
            if(MEM==MemType::DEVICE)
                return getValDevice(a_c,a_d,a_e);
            else
#endif
                {
                std::cout << "1" << std::endl;
                std::cout << "nth(";
                SHOW(m_ptrs);
                std::cout << ", " << a_c + C*a_d + C*D*a_e << ")\n" << std::endl;
                return *(m_ptrs[a_c + C*a_d + C*D*a_e]);
                }
        }

        /// Pointwise Accessor (Const)
        /**
          Access component (c,d,e) of the <code>const BoxData<T,C,D,E></code> 
          associated with <code>*this</code>.

          \param a_c   First component index
          \param a_d   Second component index (default: 0)
          \param a_e   Third component index  (default: 0)
          */
#ifdef PROTO_ACCEL
        __device__
        inline const T& getValDevice(
                unsigned int a_c,
                unsigned char a_d = 0,
                unsigned char a_e = 0) const
        {
            int idx = threadIdx.x + blockIdx.x*blockDim.x;
            int idy = blockIdx.y;
#if DIM == 3
            int idz = blockIdx.z;
            return (m_ptrs[a_c + C*a_d + C*D*a_e])[idx + boxDimX * (idy + idz * boxDimY)];
#else
            return (m_ptrs[a_c + C*a_d + C*D*a_e])[idx + idy * boxDimX];
#endif
        }

#ifdef PROTO_HIP
        __host__
        inline const T& getValDevice(
                unsigned int a_c,
                unsigned char a_d = 0,
                unsigned char a_e = 0) const
        {
            return *(m_ptrs[a_c + C*a_d + C*D*a_e]);
        }
#endif
#else
        inline const T& getValDevice(
                unsigned int a_c,
                unsigned char a_d = 0,
                unsigned char a_e = 0) const
        {
            std::cout << "2" << std::endl;
                std::cout << "nth(";
                SHOW(m_ptrs);
                std::cout << ", " << a_c + C*a_d + C*D*a_e << ")\n" << std::endl;
            return *(m_ptrs[a_c + C*a_d + C*D*a_e]);
        }
#endif

        ACCEL_DECORATION
        __attribute__((always_inline))
        const T& operator()(
                unsigned int a_c,
                unsigned char a_d = 0,
                unsigned char a_e = 0) const 
        {
#ifdef PROTO_ACCEL
            if(MEM==MemType::DEVICE)
                return getValDevice(a_c,a_d,a_e);
            else
#endif
            {
                std::cout << "3" << std::endl;
                std::cout << "nth(";
                SHOW(m_ptrs);
                std::cout << ", " << a_c + C*a_d + C*D*a_e << ")\n" << std::endl;
                return *(m_ptrs[a_c + C*a_d + C*D*a_e]);
            }
        }

        ACCEL_DECORATION
        inline myVar&  operator+=(unsigned int& a_increment) 
        {
            for (int ii = 0; ii < C*D*E; ii++)
            {
                std::cout << "4" << std::endl;
                std::cout << "nth(";
                SHOW(m_ptrs);
                // std::cout << ", " << a_c + C*a_d + C*D*a_e << ")\n" << std::endl;
                m_ptrs[ii] += a_increment;
            }
            return *this;
        }

        // test
        ACCEL_DECORATION
        inline myVar&  operator+=(Point& a_p)
        {
#if DIM == 3
            unsigned int shift = a_p[0] + (a_p[1] + a_p[2]*boxDimY)*boxDimX;
#else
            unsigned int shift = a_p[0] + a_p[1]*boxDimX;
#endif
            std::cout << "5" << std::endl;
                std::cout << "nth(";
                SHOW(m_ptrs);
                // std::cout << ", " << a_c + C*a_d + C*D*a_e << ")\n" << std::endl;
            return *this+=shift;
        }

        ACCEL_DECORATION
        inline myVar&  operator+=(const Point& a_p)
        {
#if DIM == 3
            unsigned int shift = a_p[0] + (a_p[1] + a_p[2]*boxDimY)*boxDimX;
#else
            unsigned int shift = a_p[0] + a_p[1]*boxDimX;
#endif
            std::cout << "6" << std::endl;
                std::cout << "nth(";
                SHOW(m_ptrs);
                // std::cout << ", " << a_c + C*a_d + C*D*a_e << ")\n" << std::endl;
            return *this+=shift;
        }

        ACCEL_DECORATION
        inline myVar& operator++() 
        {
            for (int ii = 0; ii < C*D*E; ii++)
            {
                std::cout << "7" << std::endl;
                std::cout << "nth(";
                SHOW(m_ptrs);
                // std::cout << ", " << a_c + C*a_d + C*D*a_e << ")\n" << std::endl;
                ++m_ptrs[ii];
            }
            return *this;
        }

        ACCEL_DECORATION
        inline myVar& operator--() 
        {
            for (int ii = 0; ii < C*D*E; ii++)
            {
                std::cout << "8" << std::endl;
                std::cout << "nth(";
                SHOW(m_ptrs);
                // std::cout << ", " << a_c + C*a_d + C*D*a_e << ")\n" << std::endl;
                --m_ptrs[ii];
            }
            return *this;
        }

        unsigned int boxDimX;
        unsigned int boxDimY;
        unsigned int subBoxDimX;
        unsigned int subBoxDimY;
        T* m_ptrs[C*D*E];
    }; // End Class Var
}

class MyDouble2 {
private:
    double value;
    std::string var = "var";
    static int counter; 
public:

     // Conversion operator to double
    operator double() const {
        return value;
    }

    MyDouble2() {
        ++counter;
        var += std::to_string(counter);
        std::cout << "var(\"" << var << "\");" << std::endl;
    }

    MyDouble2(double val) : value(val) {
        ++counter;
        var += std::to_string(counter);
        std::cout << "assign(" << var 
        << ", " << "V(" << val << ")";
        std::cout << ");" << std::endl;
    }



    // Overload the -= operator (in-place subtraction)
    MyDouble2& operator-=(const MyDouble2& other) {
        std::cout << "assign_sub(" << var << counter 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        value -= other.value;
        return *this;
    }

        // Overload the *= operator (in-place multiplication)
    MyDouble2& operator*=(const MyDouble2& other) {
        std::cout << "assign_mul(" << var << counter 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        value *= other.value;
        return *this;
    }

    // Overload the /= operator (in-place division)
    MyDouble2& operator/=(const MyDouble2& other) {
        if (other.value == 0.0) {
            // Handle division by zero as needed
            // For example, throw an exception
            throw std::runtime_error("Division by zero");
        }
        std::cout << "assign_div(" << var << counter 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        value /= other.value;
        return *this;
    }

    // Overload the equality operator for comparison with regular doubles
    bool operator==(double other) const {
        return value == other;
    }

    // Overload the equality operator for comparison with regular ints
    bool operator==(int other) const {
        return value == other;
    }

    // Overload the equality operator for comparison with regular float
    bool operator==(float other) const {
        return value == other;
    }

    MyDouble2 operator-() const {
        return MyDouble2(-value);
    }

        // Overload multiplication with a scalar int
    MyDouble2 operator*(int scalar) const {
        return MyDouble2(value * static_cast<double>(scalar));
    }

    // Overload multiplication with a scalar double
    MyDouble2 operator*(double scalar) const {
        return MyDouble2(value * scalar);
    }

    // // Overload the - operator
     template<typename T>
    MyDouble2 operator+(const T& other) const {
        std::cout << "add(" << var  
        << ", " << other.var;
        std::cout << ");" << std::endl;
        return MyDouble2(value + static_cast<double>(other));
    }

    // // Overload the - operator
     template<typename T>
    MyDouble2 operator-(const T& other) const {
        std::cout << "sub(" << var  
        << ", " << other.var;
        std::cout << ");" << std::endl;
        return MyDouble2(value - static_cast<double>(other));
    }

    // // Overload the - operator
     template<typename T>
    MyDouble2 operator=(const T& other) const {
        std::cout << "assign(" << var  
        << ", " << other.var;
        std::cout << ");" << std::endl;
        return MyDouble2(static_cast<double>(other));
    }

    // // Overload the * operator
    MyDouble2 operator*(const MyDouble2& other) const {
        std::cout << "mul(" << var 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        return MyDouble2(value * other.value);
    }

    // Overload operator* for Double * T
      template<typename T>
    bool operator>(const T& other) const {
        return value > static_cast<double>(other) ? true : false;
    }

    // Overload operator* for Double * T
      template<typename T>
    bool operator<(const T& other) const {
        return value < static_cast<double>(other) ? true : false;
    }


    // Overload operator* for Double * T
      template<typename T>
    MyDouble2 operator/(const T& other) const {
        return MyDouble2(value / static_cast<double>(other));
    }

      // Overload operator* for Double * T
      template<typename T>
    MyDouble2 operator*(const T& other) const {
        return MyDouble2(value * static_cast<double>(other));
    }

    MyDouble2 operator~() const {
        if (value < 0) {
            // Handle negative numbers by returning a special value or throwing an exception
            std::cerr << "Cannot calculate the square root of a negative number." << std::endl;
            return MyDouble2(NAN); // Not-a-Number (NaN) or you can choose a different approach
        }
        return MyDouble2(sqrt(value));
    }

        // Overload the + operator
    MyDouble2 operator+(const MyDouble2& other) const {
        std::cout << "add(" << var << counter 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        return MyDouble2(value + other.value);
    }

    // Overload the += operator (in-place addition)
    MyDouble2& operator+=(const MyDouble2& other) {
        std::cout << "assgin_add(" << var << counter 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        value += other.value;
        return *this;
    }

    // Overload the equality operator for comparison with MyDouble objects
    bool operator==(const MyDouble2& other) const {
        return value == other.value;
    }

        // Overload the inequality operator for comparison with MyDouble objects
    bool operator!=(const MyDouble2& other) const {
        return value != other.value;
    }

    // // Overload the / operator
    MyDouble2 operator/(const MyDouble2& other) const {
        if (other.value == 0.0) {
            // Handle division by zero as needed
            // For example, throw an exception
            throw std::runtime_error("Division by zero");
        }
        std::cout << "div(" << var  
        << ", " << other.var;
        std::cout << ");" << std::endl;
        return MyDouble2(value / other.value);
    }

    // Overload the assignment operator
    MyDouble2& operator=(const MyDouble2& other) {
        std::cout << "assign(" << var 
        << ", " << other.var << ");" << std::endl;
        if (this != &other) {
            value = other.value;
        }
        return *this;
    }

     // // Overload the > operator
    bool operator>(const MyDouble2& other) const { 
        return value > other.value ? true : false;
    }


    ~MyDouble2()
    {
        counter--;
    }

    // Define getter to retrieve the underlying double value
    double getValue() const {
        return value;
    }
};
int MyDouble2::counter = 0;



class MyDouble {
private:
    double value;
    std::string var = "var";
    static int counter; 
public:

    MyDouble() {
        ++counter;
        var += std::to_string(counter);
        std::cout << "var(\"" << var << "\");" << std::endl;
    }

    MyDouble(double val) : value(val) {
        ++counter;
        var += std::to_string(counter);
        std::cout << "assign(" << var 
        << ", " << "V(" << val << ")";
        std::cout << ");" << std::endl;
    }

    // Overload the + operator
    MyDouble operator+(const MyDouble& other) const {
        std::cout << "add(" << var << counter 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        return MyDouble(value + other.value);
    }

    // Overload the += operator (in-place addition)
    MyDouble& operator+=(const MyDouble& other) {
        std::cout << "assgin_add(" << var << counter 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        value += other.value;
        return *this;
    }

    // Overload the -= operator (in-place subtraction)
    MyDouble& operator-=(const MyDouble& other) {
        std::cout << "assign_sub(" << var << counter 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        value -= other.value;
        return *this;
    }

        // Overload the *= operator (in-place multiplication)
    MyDouble& operator*=(const MyDouble& other) {
        std::cout << "assign_mul(" << var << counter 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        value *= other.value;
        return *this;
    }

    // Overload the /= operator (in-place division)
    MyDouble& operator/=(const MyDouble& other) {
        if (other.value == 0.0) {
            // Handle division by zero as needed
            // For example, throw an exception
            throw std::runtime_error("Division by zero");
        }
        std::cout << "assign_div(" << var << counter 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        value /= other.value;
        return *this;
    }

    // Overload the equality operator for comparison with MyDouble objects
    bool operator==(const MyDouble& other) const {
        return value == other.value;
    }

    // Overload the equality operator for comparison with regular doubles
    bool operator==(double other) const {
        return value == other;
    }

    // Overload the equality operator for comparison with regular ints
    bool operator==(int other) const {
        return value == other;
    }

    // Overload the equality operator for comparison with regular float
    bool operator==(float other) const {
        return value == other;
    }

    MyDouble operator-() const {
        return MyDouble(-value);
    }

        // Overload multiplication with a scalar int
    MyDouble operator*(int scalar) const {
        return MyDouble(value * static_cast<double>(scalar));
    }

    // Overload multiplication with a scalar double
    MyDouble operator*(double scalar) const {
        return MyDouble(value * scalar);
    }

    // Overload the inequality operator for comparison with MyDouble objects
    bool operator!=(const MyDouble& other) const {
        return value != other.value;
    }


    // // Overload the - operator
    MyDouble operator-(const MyDouble& other) const {
        std::cout << "sub(" << var  
        << ", " << other.var;
        std::cout << ");" << std::endl;
        return MyDouble(value - other.value);
    }

    // // Overload the * operator
    MyDouble operator*(const MyDouble& other) const {
        std::cout << "mul(" << var 
        << ", " << other.var;
        std::cout << ");" << std::endl;
        return MyDouble(value * other.value);
    }

    // // Overload the / operator
    MyDouble operator/(const MyDouble& other) const {
        if (other.value == 0.0) {
            // Handle division by zero as needed
            // For example, throw an exception
            throw std::runtime_error("Division by zero");
        }
        std::cout << "div(" << var  
        << ", " << other.var;
        std::cout << ");" << std::endl;
        return MyDouble(value / other.value);
    }

    // Overload the assignment operator
    MyDouble& operator=(const MyDouble& other) {
        std::cout << "assign(" << var 
        << ", " << other.var << ");" << std::endl;
        if (this != &other) {
            value = other.value;
        }
        return *this;
    }

    // MyDouble& operator=(double other) {
    //     std::cout << "assign(";
    //     SHOW(this);
    //     std::cout << ", " << "V(" << other << ")";
        
    //     std::cout << ");" << std::endl;
    //     value = other;
    //     return *this;
    // }

    ~MyDouble()
    {
        counter--;
    }

    // Define getter to retrieve the underlying double value
    double getValue() const {
        return value;
    }
};
int MyDouble::counter = 0;

// #define Var myVar
#define MyDouble MyDouble2



// namespace Proto {
//     template <class T=double, unsigned int C=1,
//     MemType MEM=MEMTYPE_DEFAULT,
//     unsigned char D=1, unsigned char E=1>
//     class LazyBoxData {
//         bool lazy = false;
//     public:
//         void setLazy() {
//             if (lazy == false)
//                 lazy = true;
//             else
//                 lazy = false;
//         }
//         T* operator[](unsigned int a_index) {
//             if(lazy == true) {
//                 setLazy();
//                 std::cout << "we are calling SPIRAL codegen here" << std::endl;
//             }
//             return T[a_index];
//         }

//     };
// }

// #define BoxData LazyBoxData