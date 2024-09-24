#include <iostream>
#include <string>
#define SHOW(a) std::cout << #a << std::endl;

// class MyDouble {
// private:
//     double value;
//     std::string var = "var";
//     static int counter; 
// public:

//     MyDouble() {
//         ++counter;
//         var += std::to_string(counter);
//         std::cout << "var(\"" << var << ", TReal\");" << std::endl;
//     }

//     MyDouble(double val) : value(val) {
//         ++counter;
//         var += std::to_string(counter);
//         std::cout << "assign(" << var 
//         << ", " << "V(" << val << ")";
//         std::cout << ");" << std::endl;
//     }

//     // Overload the + operator
//     MyDouble operator+(const MyDouble& other) const {
//         std::cout << "add(" << var << counter 
//         << ", " << other.var;
//         std::cout << ");" << std::endl;
//         return MyDouble(value + other.value);
//     }

//     // Overload the += operator (in-place addition)
//     MyDouble& operator+=(const MyDouble& other) {
//         std::cout << "assgin_add(" << var << counter 
//         << ", " << other.var;
//         std::cout << ");" << std::endl;
//         value += other.value;
//         return *this;
//     }

//     // Overload the -= operator (in-place subtraction)
//     MyDouble& operator-=(const MyDouble& other) {
//         std::cout << "assign_sub(" << var << counter 
//         << ", " << other.var;
//         std::cout << ");" << std::endl;
//         value -= other.value;
//         return *this;
//     }

//         // Overload the *= operator (in-place multiplication)
//     MyDouble& operator*=(const MyDouble& other) {
//         std::cout << "assign_mul(" << var << counter 
//         << ", " << other.var;
//         std::cout << ");" << std::endl;
//         value *= other.value;
//         return *this;
//     }

//     // Overload the /= operator (in-place division)
//     MyDouble& operator/=(const MyDouble& other) {
//         if (other.value == 0.0) {
//             // Handle division by zero as needed
//             // For example, throw an exception
//             throw std::runtime_error("Division by zero");
//         }
//         std::cout << "assign_div(" << var << counter 
//         << ", " << other.var;
//         std::cout << ");" << std::endl;
//         value /= other.value;
//         return *this;
//     }

//     // Overload the equality operator for comparison with MyDouble objects
//     bool operator==(const MyDouble& other) const {
//         return value == other.value;
//     }

//     // Overload the equality operator for comparison with regular doubles
//     bool operator==(double other) const {
//         return value == other;
//     }

//     // Overload the equality operator for comparison with regular ints
//     bool operator==(int other) const {
//         return value == other;
//     }

//     // Overload the equality operator for comparison with regular float
//     bool operator==(float other) const {
//         return value == other;
//     }

//     MyDouble operator-() const {
//         return MyDouble(-value);
//     }

//         // Overload multiplication with a scalar int
//     MyDouble operator*(int scalar) const {
//         return MyDouble(value * static_cast<double>(scalar));
//     }

//     // Overload multiplication with a scalar double
//     MyDouble operator*(double scalar) const {
//         return MyDouble(value * scalar);
//     }

//     // Overload the inequality operator for comparison with MyDouble objects
//     bool operator!=(const MyDouble& other) const {
//         return value != other.value;
//     }

//     // Overload operator* for double * Double
//     friend MyDouble operator*(const double& lhs, const MyDouble& rhs) {
//         return MyDouble(lhs * rhs.value);
//     }

//     // // Overload the - operator
//     MyDouble operator-(const MyDouble& other) const {
//         std::cout << "sub(" << var  
//         << ", " << other.var;
//         std::cout << ");" << std::endl;
//         return MyDouble(value - other.value);
//     }

//     // // Overload the * operator
//     MyDouble operator*(const MyDouble& other) const {
//         std::cout << "mul(" << var 
//         << ", " << other.var;
//         std::cout << ");" << std::endl;
//         return MyDouble(value * other.value);
//     }

//     // // Overload the / operator
//     MyDouble operator/(const MyDouble& other) const {
//         if (other.value == 0.0) {
//             // Handle division by zero as needed
//             // For example, throw an exception
//             throw std::runtime_error("Division by zero");
//         }
//         std::cout << "div(" << var  
//         << ", " << other.var;
//         std::cout << ");" << std::endl;
//         return MyDouble(value / other.value);
//     }

//     // Overload the assignment operator
//     MyDouble& operator=(const MyDouble& other) {
//         std::cout << "assign(" << var 
//         << ", " << other.var << ");" << std::endl;
//         if (this != &other) {
//             value = other.value;
//         }
//         return *this;
//     }

//     // MyDouble& operator=(double other) {
//     //     std::cout << "assign(";
//     //     SHOW(this);
//     //     std::cout << ", " << "V(" << other << ")";
        
//     //     std::cout << ");" << std::endl;
//     //     value = other;
//     //     return *this;
//     // }

//     ~MyDouble()
//     {
//         counter--;
//     }

//     // Define getter to retrieve the underlying double value
//     double getValue() const {
//         return value;
//     }
// };
// int MyDouble::counter = 0;



class MyDouble {
public:
    // Constructors
    MyDouble() : value(0.0) {
        // var += "var";
        ++counter;
        var += std::to_string(counter);
        std::cout << "let(" << var << " := var.fresh_t(\"" << var << "\", TReal)," << var << ")," << std::endl;
        // std::cout << "assign(" << var << ", V(" << value << ")))," << std::endl;
    }
    MyDouble(double val) : value(val) {
        ++counter;
        var += std::to_string(counter);
        std::cout << "let(" << var << " := var.fresh_t(\"" << var << "\", TReal)," << std::endl;
        std::cout << "chain(assign(" << var << ", V(" << value << "))))," << std::endl;

    }

    //Copy Constructor
    MyDouble(const MyDouble& other) : value(other.value) {
        ++counter;
        var += std::to_string(counter);
        std::cout << "let(" << var << " := var.fresh_t(\"" << var << "\", TReal)," << std::endl;
        std::cout << "chain(assign(" << var << "," << other.var << ")))," << std::endl;
    }

    // Subtraction of a template type from MyDouble
    template<typename T>
    MyDouble operator-(T other) const {
        // std::string new_var("var");
        //     new_var += std::to_string(counter);
        //     std::cout << "let(" << new_var << " := var.fresh_t(\"" << new_var << "\", TReal))," << std::endl;
        //     std::cout << "chain(assign(" << new_var << ", sub(" << var << ", " << other.var << ")))," << std::endl;
            
        //     MyDouble local;
        //     local.var.assign(new_var);
        //     return local;
        return MyDouble(this->value - static_cast<double>(other));
    }


   // Basic arithmetic operators
    MyDouble operator-(const MyDouble& other) const {
      std::string new_var("var");
            new_var += std::to_string(counter);
            std::cout << "let(" << new_var << " := var.fresh_t(\"" << new_var << "\", TReal)," << std::endl;
            std::cout << "chain(assign(" << new_var << ", sub(" << var << ", " << other.var << ")))," << std::endl;
            
            MyDouble local;
            local.var.assign(new_var);
            return local;
        // return MyDouble(this->value + other.value);
    }

    // Basic arithmetic operators
    MyDouble operator+(const MyDouble& other) const {
      std::string new_var("var");
            new_var += std::to_string(counter);
            std::cout << "let(" << new_var << " := var.fresh_t(\"" << new_var << "\", TReal)," << std::endl;
            std::cout << "chain(assign(" << new_var << ", add(" << var << ", " << other.var << ")))," << std::endl;
            
            MyDouble local;
            local.var.assign(new_var);
            return local;
        // return MyDouble(this->value + other.value);
    }

    MyDouble operator*(const MyDouble& other) const {
        std::string new_var("var");
            new_var += std::to_string(counter);
            std::cout << "let(" << new_var << " := var.fresh_t(\"" << new_var << "\", TReal)," << std::endl;
            std::cout << "chain(assign(" << new_var << ", mul(" << var << ", " << other.var << "))))," << std::endl;
            
            MyDouble local;
            local.var.assign(new_var);
            return local;
        return MyDouble(this->value * other.value);
    }

    MyDouble operator/(const MyDouble& other) const {
        if (other.value != 0.0) {
            ++counter;
            std::string new_var("var");
            new_var += std::to_string(counter);
            std::cout << "let(" << new_var << " := var.fresh_t(\"" << new_var << "\", TReal)," << std::endl;
            std::cout << "chain(assign(" << new_var << ", div(" << var << ", " << other.var << "))))," << std::endl;
            
            MyDouble local;
            local.var.assign(new_var);
            return local;
            // return MyDouble(this->value / other.value);
        } else {
            std::cerr << "Error: Division by zero." << std::endl;
            return MyDouble(0.0); // Return a default value
        }
    }

    // Assignment operators
    MyDouble& operator=(const MyDouble& other) {
        // std::cout << "HELLO????" << std::endl;
        if (this != &other) {
            this->value = other.value;
        }
        std::cout << "let(chain(assign(" << var << ", " << other.var << ")))," << std::endl;
        return *this;
    }

    // += operator
    MyDouble& operator+=(const MyDouble& other) {
        this->value += other.value;
        return *this;
    }

    // Comparison operators
    bool operator==(const MyDouble& other) const {
        return this->value == other.value;
    }

    bool operator!=(const MyDouble& other) const {
        return this->value != other.value;
    }

    // Pointer support
    MyDouble* operator&() {
        std::cout << "hello from *" << std::endl;
        // std::cout << "let(assign(" << var << ", " << this.var << "))," << std::endl;
        return this;
    }

    const MyDouble* operator&() const {
        // std::cout << "HELLO FROM *" << std::endl;
        return this;
    }

    // Conversion to double
    operator double() const {
        // std::cout << "HELLO FROM conversion" << std::endl;
        return value;
    }

    // Output operator
    friend std::ostream& operator<<(std::ostream& os, const MyDouble& myDouble) {
        std::cout << "hello from print" << std::endl;
        os << myDouble.value;
        return os;
    }

    ~MyDouble()
    {
        counter--;
    }

private:
    mutable double value;
    mutable std::string var = "var";
    static int counter; 
};
int MyDouble::counter = 0;

// #define myVar Var
#define double MyDouble

int main() {

    std::cout << "icode := [" <<std::endl;
    double gamma = 1.0;
    // SHOW(gamma);
    double input[4] = {3,3,3,3};
    //SHOW(input);
    double output[4] = {0,0,0,0};
    //SHOW(output);
    double rho = input[0];
    //SHOW(rho);
    // std::cout << rho << std::endl;
    double v2 = 0.0;
    // SHOW(v2);
    output[0] = rho;

    for(int i = 1; i <= 1; i++) {
        double v;
        // SHOW(v);
        v = input[i]/rho;
        // SHOW(v);
        output[i] = v;
        v2 += v*v;
    }
    output[3] = (input[3] - 0.5 * rho * v2) * (gamma - 1.0);
    std::cout << "];" << std::endl;
    return 0;
}