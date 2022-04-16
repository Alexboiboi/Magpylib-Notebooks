import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from demag_functions import mesh_cuboid, apply_demag

# mesh factor (split up side length of a into a*meshf parts)
meshf = 3

# some low quality magnets with different parameters split up into cells
cube1 = magpy.magnet.Cuboid(magnetization=(0,0,1000), dimension=(1,1,1))
coll1 = mesh_cuboid(cube1, [meshf]*3)
coll1.move((-1.5,0,0))
xi1 = [0.3]*len(coll1) #mur=1.3

cube2 = magpy.magnet.Cuboid(magnetization=(900,0,0), dimension=(1,1,1))
coll2 = mesh_cuboid(cube2, [meshf]*3)
coll2.rotate_from_angax(-45, 'y').move((0,0,.2))
xi2 = [1.]*len(coll2) #mur=2.0

mx, my = 600*np.sin(30/180*np.pi), 600*np.cos(30/180*np.pi)
cube3 = magpy.magnet.Cuboid(magnetization=(mx,my,0), dimension=(1,1,2))
coll3 = mesh_cuboid(cube3, [meshf,meshf,meshf*2])
coll3.move((1.6,0,.5)).rotate_from_angax(30, 'z')
xi3 = [0.5]*len(coll3) #mu3=1.5

# collection of all cells
COLL = magpy.Collection(coll1, coll2, coll3)
xi_vector = np.array(xi1 + xi2 + xi3)

# sensor
sensor = magpy.Sensor(position=np.linspace((-4,0,-1), (4,0,-1), 301))

# compute field before demag
B0 = sensor.getB(COLL)

# apply demag
apply_demag(
    COLL,
    xi_vector,
    demag_store='500cells',
    demag_load=False)

# compute field after demag
B1 = sensor.getB(COLL)

# load ANSYS FEM data 
FEMdata = np.genfromtxt('FEMdata_test_cuboids_04.csv', delimiter=',', skip_header=1, skip_footer=0).T

fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

magpy.show(coll1, coll2, coll3, sensor, canvas=ax1)

# plot field from FE computation
ax2.plot(FEMdata[1]*1000, color='k', label='FEM (ANSYS)')
ax2.plot(FEMdata[2]*1000, color='k')
ax2.plot(FEMdata[3]*1000, color='k')

# plot field without demag
ax2.plot(B0[:,0], 'y', ls=':', label='Bx magpy - no interaction')
ax2.plot(B0[:,1], 'm', ls=':', label='By magpy - no interaction')
ax2.plot(B0[:,2], 'c', ls=':', label='Bz magpy - no interaction')

# plot field with demag
ax2.plot(B1[:,0], 'y', ls='--', label='Bx magpy - 500cell interaction')
ax2.plot(B1[:,1], 'm', ls='--', label='By magpy - 500cell interaction')
ax2.plot(B1[:,2], 'c', ls='--', label='Bz magpy - 500cell interaction')

ax2.set(
    title='B-field at sensor line [mT]',
    xlabel='position on line [mm]',
)
ax2.grid(color='.9')
ax2.legend()

plt.tight_layout()
plt.show()
