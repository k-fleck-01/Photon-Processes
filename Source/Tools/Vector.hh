#ifndef Vector_hh
#define Vector_hh

#include <iostream>
#include <cassert>
#include <iomanip>

#if USEHDF5
    #include "H5Cpp.h"
#endif


template <typename T>
class Vector
{
    private:
        int m_size;
        T *m_data;
    
    public:

        Vector():
        m_size(0), m_data(nullptr)
        {
        }

        Vector(int size):
        m_size(size)
        {
            m_data = new T [m_size];
        }

        Vector(const std::initializer_list<T> list):
        m_size(list.size())
        {
            m_data = new T [m_size];
            std::copy(list.begin(), list.end(), m_data);
        }

        // Copy constructor 
        Vector(const Vector &Vector)
        {
            m_size = Vector.m_size;
            m_data = new T [Vector.m_size];

            for (int i = 0; i < Vector.m_size; i++)
            {
                m_data[i] = Vector.m_data[i];
            }
        }

        // Defult destructor. This deletes the heap memory and avoids memory leaks
        ~Vector()
        {
            delete[] m_data;
        }

        Vector<T> deepCopy() const
        {
            Vector<T> copy(m_size);
            for (int i = 0; i < m_size; i++)
            {
                copy.m_data[i] = m_data[i];
            }
            return copy;
        }
        
        int size() const
        {
            return m_size;
        }

        T* begin() const
        {
            return m_data;
        }

        T* end() const
        {
            return &m_data[m_size - 1];
        }

        void empty()
        {
            for (int i = 0; i < m_size; i++)
            {
                m_data[i] = 0.0;
            }
        }

#if USEHDF5
        void save(const H5std_string fileName, const H5std_string dataName)
        {
        //  H5::Exception::dontPrint(); 
            H5::H5File* file = new H5::H5File(fileName, H5F_ACC_TRUNC);
            int dataRank = 1;
            hsize_t dataDim[1];
            dataDim[0] = m_size;
            H5::DataSpace dataSpace(dataRank, dataDim);
            H5::DataSet* dataSet = new H5::DataSet(
                file->createDataSet(dataName, H5::PredType::NATIVE_DOUBLE, dataSpace));
            dataSet->write(m_data, H5::PredType::NATIVE_DOUBLE);
            file->close();
            delete dataSet;
        }
#endif
        
        // Sums all the elements together
        T sum()
        {
            T sum(0);
            for (int i = 0; i < m_size; i++)
            {

                sum += m_data[i];
            }
            return sum;
        }

        // Appends a row to the bottom of the Vector
        void appendVector(const Vector<T> &vector)
        {
            //  assert(row.GetColumns == m_nColumn);
            T *dataNew = new T [m_size + vector.m_size];

            // fill the new array
            for (int i = 0; i < m_size; i++)
            {
                dataNew[i] = m_data[i];
            }

            for (int i = 0; i < vector.m_size; i++)
            {
                dataNew[i + m_size] = vector.m_data[i];
            }

            delete [] m_data;
            m_data = dataNew;
            m_size += vector.m_size;
        }

        void print() const
        {
            std::cout << *this;
        }

    public:

        T& operator()(int index)
        {
            assert(index >= 0 && index < m_size);
            return m_data[index];
        }

        T& operator()(int index) const
        {
            assert(index >= 0 && index < m_size);
            return m_data[index];
        }

        T& operator[](int index)
        {
            assert(index >= 0 && index < m_size);
            return m_data[index];
        }

        T& operator[](int index) const
        {
            assert(index >= 0 && index < m_size);
            return m_data[index];
        }

        Vector<T>& operator=(const Vector<T> &Vector)
        {
            // Self assigmnet gard
            if( this == &Vector)
            {
                return *this;
            }

            // Delete any data in new Vector is holding
            delete[] m_data;

            // Copy variables over
            m_size = Vector.m_size;
            if(Vector.m_data)
            {
                // allocate the array space
                m_data = new T [m_size];
                // Copy the data over
                for(int i = 0; i < m_size; i++)
                {

                    m_data[i] = Vector.m_data[i];
                }
            }else
            {
                m_data = 0;
            }

            return *this;
            // Copys a matirx to a new variable
        }

        // Prints Vector to the screen
        friend std::ostream& operator << (std::ostream& out, const Vector<T>& vector)
        {
            out << "[ ";
            for(int i = 0; i < vector.m_size; i++)
            {
                out << std::fixed << vector.m_data[i] << ", ";
                if(i == vector.m_size - 1)
                {
                    out << "]" << std::endl;
                }
            }
            return out;
        }
};
#endif