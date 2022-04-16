import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from demag_functions import mesh_cuboid, apply_demag


meshf = 3 # mesh factor

# hard magnet
mag1 = (0,0,1000)
dim1 = (1,1,2)
cube1 = magpy.magnet.Cuboid(mag1, dim1, (0,0,0.5))
col1 = mesh_cuboid(cube1, (meshf,meshf,meshf*2))
#col1 = mesh_cuboid(cube1, (3,3,6))
xi1 = [0.5]*len(col1)

# soft magnet
mag2 = (0,0,0)
dim2 = (1,1,1)
cube2 = magpy.magnet.Cuboid(mag2, dim2, (0,0,0))
col2 = mesh_cuboid(cube2, (meshf,meshf,meshf)).move((1.5,0,0))
xi2 = [3999]*len(col2)
col2.rotate_from_angax(angle=45, axis='y', anchor=None)

# super collection
COL0 = col1 + col2
xi_vector = np.array(xi1 + xi2)

# apply demag to copy of super
COL1 = COL0.copy()

apply_demag(
    COL1,
    xi_vector,
    demag_store=False,
    demag_load=False)

# add sensors
sensors = [
    magpy.Sensor(
        position=np.linspace((-4,0,z), (6,0,z), 1001)
    ) for z in [-1,-3,-5]
]

magpy.show(COL1, sensors)


fig, [ax1,ax2,ax3] = plt.subplots(1,3,figsize=(12,5))

# load and plot FEM data coarse mesh
FEMdata = np.genfromtxt('FEMdata_test_softmag_01.csv', delimiter=',', skip_header=1, skip_footer=0).T
ax1.plot(FEMdata[1]*1000, color='k')
ax1.plot(FEMdata[2]*1000, color='k')
ax2.plot(FEMdata[3]*1000, color='k')
ax2.plot(FEMdata[4]*1000, color='k')
ax3.plot(FEMdata[5]*1000, color='k')
ax3.plot(FEMdata[6]*1000, color='k')


# load and plot FEM data fine mesh
FEMdata = np.genfromtxt('FEMdata_test_softmag_02.csv', delimiter=',', skip_header=1, skip_footer=0).T
ax1.plot(FEMdata[1]*1000, color='k')
ax1.plot(FEMdata[2]*1000, color='k')
ax2.plot(FEMdata[3]*1000, color='k')
ax2.plot(FEMdata[4]*1000, color='k')
ax3.plot(FEMdata[5]*1000, color='k')
ax3.plot(FEMdata[6]*1000, color='k')

# compute and plot results without demag
B0  = COL0.getB(*sensors)
ax1.plot(B0[:,0,0], color='r', ls=':')
ax1.plot(B0[:,0,2], color='r', ls=':')
ax2.plot(B0[:,1,0], color='gold', ls=':')
ax2.plot(B0[:,1,2], color='gold', ls=':')
ax3.plot(B0[:,2,0], color='c', ls=':')
ax3.plot(B0[:,2,2], color='c', ls=':')

# compute and plot demag results
B1  = COL1.getB(*sensors)
ax1.plot(B1[:,0,0], color='r', ls='--')
ax1.plot(B1[:,0,2], color='r', ls='--')
ax2.plot(B1[:,1,0], color='gold', ls='--')
ax2.plot(B1[:,1,2], color='gold', ls='--')
ax3.plot(B1[:,2,0], color='c', ls='--')
ax3.plot(B1[:,2,2], color='c', ls='--')

plt.tight_layout()
plt.show()
