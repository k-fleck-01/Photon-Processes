#ifndef PhotonField_hh
#define PhotonField_hh

#include "Matrix.hh"
#include "Vector.hh"

class PhotonField
{
public:
    /* Override by returning true for isotropic or false 
       for nonisotroipic */
    virtual bool isIsotropic() const = 0;

    /* returns the dimensionality of the field. i.e 2 for isotropic
       or 3 for nonisotropic */
    virtual int fieldDimensions() const = 0;

    /* Following getter methods allow acces to the photon field 
       energy, angle and density variables. */
    virtual int getEnergyRes() const {return m_energyRes;}

    virtual int getAngleRes() const {return m_angleRes;}

    virtual const Vector<double>& getEnergy() const {return m_energy;}

    virtual const Vector<double>& getEnergyDensity() const {
        return m_energyDensity;}

    virtual const Vector<double>& getEnergydensity(int blockID) const {
        return getEnergyDensity();
    }
  
    virtual const Vector<double>& getTheta(int blockID) {return m_theta;}

    virtual const Vector<double>& getPhi(int blockID) {return m_phi;}

    virtual const Matrix<double>& getAngleDensity(int blockID)
        {return m_angleDensity[blockID];}

    virtual int getNumBlocks() const {return m_nBlocks;} 

protected:

    int m_nBlocks;
    int m_energyRes;
    int m_angleRes;
    Vector<double> m_energy;
    Vector<double> m_energyDensity;
    Vector<double> m_theta;
    Vector<double> m_phi;
    Vector<Matrix<double>> m_angleDensity;
};

#endif
