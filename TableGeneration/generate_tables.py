import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.integrate import simpson

from PhotonCrossSection import diff_cross

### Plotting settings
mpl.rcParams["font.family"] = "Helvetica"
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["text.latex.preamble"] = r"\usepackage{revsymb}"

### Global configurations
USE_LOG_RANGE = True
SHOW_CROSS_SECTION = False

if __name__ == "__main__":
    ### Define energy and angle grid
    GRIDSIZE = 250
    OMEGA_MIN = 5.0e-2
    OMEGA_MAX = 1.0e4

    if USE_LOG_RANGE:
        ### Logarithmic spacing chosen for omega = ZMF energy of each photon
        log_range = np.linspace(np.log10(OMEGA_MIN),
                                np.log10(OMEGA_MAX),
                                GRIDSIZE)
        energy = np.power(10.0, log_range)
    else:
        ### Linear spacing chosen for omega = ZMF energy of each photon
        energy = np.linspace(OMEGA_MIN, OMEGA_MAX, GRIDSIZE)
    
    s = 4.0 * energy**2 ### Mandelstam s
    angle = np.linspace(0.01, 3.13, GRIDSIZE) ### (diverges at 0 or pi)
    p_axis = np.linspace(0.0, 1.0, GRIDSIZE)
    angle_mesh, energy_mesh = np.meshgrid(angle, energy)
    
    ### Calculation of differential and total cross section
    d_sig = diff_cross(energy_mesh, angle_mesh)
    tot_sig = 2.0*np.pi * simpson(y=d_sig*np.sin(angle_mesh),
                                  x=angle)

    ### Invert CDF for MC sampling
    cdf = np.zeros_like(d_sig)
    cdf_inv = np.zeros_like(d_sig)
    for i, angle_array in enumerate(d_sig):
        data = angle_array * np.sin(angle)
        cdf[i, 0] = data[0]
        for j in range(1, len(data)):
            cdf[i, j] = cdf[i, j-1] + data[j]
        cdf[i, :] = cdf[i, :] / cdf[i, len(data) - 1]
        cdf_inv[i, :] = np.interp(p_axis, cdf[i, :], angle)

    ### Writting tabulations to HDF5 file
    hf = h5py.File('PhotonScatter_total.h5', 'w')
    hf.create_dataset('s', data=s)
    hf.create_dataset('sigma', data=tot_sig)

    hf = h5py.File('PhotonScatter_diff.h5', 'w')
    hf.create_dataset('s', data=s)
    hf.create_dataset('p', data=p_axis)
    hf.create_dataset('angle', data=cdf_inv)

    if SHOW_CROSS_SECTION:
        fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
        ax.plot_surface(angle_mesh, energy_mesh, d_sig, cmap= "plasma")
        ax.set_ylabel(r"$\omega/m_e$")
        ax.set_xlabel(r'$\theta$')
        ax.set_zlabel(r'$(1/\lambdabar_c^2)\; d \sigma/d \Omega$')

        fig, ax = plt.subplots()
        ax.loglog(energy, tot_sig)
        ax.set_xlabel(r"$\omega/m_e$")
        ax.set_ylabel(r'$\sigma / \lambdabar_c^2$')

        plt.show()
