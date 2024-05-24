#ifndef Numerics_HH
#define Numerics_HH

#include "Vector.hh"
#include "Matrix.hh"

namespace Numerics
{
    /* Gives the index of the vector closest to the value*/
    int vectorIndex(const Vector<double>& vector, double value);

    /* Simpsons method for integration */
    double simpsons(const Vector<double>& variable,
        const Vector<double>& integrand);

    /* Enumeration for selecting interpolation type */
    enum struct InterpType { Linear, Log };
    
    /* Linear intepolation method in 1D */
    double interpolate1D_lin(const Vector<double>& sampleX,
        const Vector<double>& sampleY, double queryX);

    /* 1D linear interpolation in logarithm space */
    double interpolate1D_log(const Vector<double>& sampleX,
        const Vector<double>& sampleY, double queryX);

    /* 1D interpolation method */
    double interpolate1D(const Vector<double>& sampleX,
        const Vector<double>& sampleY, double queryX,
        const InterpType itype = InterpType::Linear);

    
    /* Linear intepolation method in 2D */
    double interpolate2D(const Vector<double>& sampleX,
        const Vector<double>& sampleY, Matrix<double> sampleZ,
        double queryPoint[2]);
}
#endif
