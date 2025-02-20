'''
author: Elko Gerville-Reache
date Created: 2025-02-10
last Modified: 2025-02-20
purpose: animates two galaxies merging with distinct cmaps where particles are represented as density points using plt.hexbin
usage: change the params in ANIMATION PARAMS, notably the path to the snapshot directory, number of particles, etc..

notes:
- requires the installation of numpy, matplotlib, celluloid, ffmpeg, and tqdm

'''
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from tqdm import tqdm
################################
# FUNCTIONS
################################
def loadData(string, i):
    # load data and return Nx3 position, Nx3 velocity, and Nx1 potential arrays
    data = np.load(string.format(i))

    return data[:,0:3], data[:,3:6], data[:,6:7]

def load2Galaxies(string, timesteps, N, NgalA, tempResolution):
    # number of snapshots
    Ntimesteps = int( (timesteps/tempResolution) + 1)
    # 3 dimensions
    Ncoord     = 3
    # number of particles in galaxyB
    NgalB      = N - NgalA
    # allocate arrays for data
    positionA   = np.zeros(shape = (Ntimesteps, NgalA, Ncoord) )
    velocityA   = np.zeros(shape = (Ntimesteps, NgalA, Ncoord) )
    potentialA  = np.zeros(shape = (Ntimesteps, NgalA, 1) )
    positionB   = np.zeros(shape = (Ntimesteps, NgalB, Ncoord) )
    velocityB   = np.zeros(shape = (Ntimesteps, NgalB, Ncoord) )
    potentialB  = np.zeros(shape = (Ntimesteps, NgalB, 1) )
    # loop through snapshots and load data
    for i in tqdm(range(Ntimesteps)):
        pos, vel, pot = loadData(string, i*tempResolution)
        positionA[i,:,:], velocityA[i,:,:], potentialA[i,:,:] = pos[0:NgalA], vel[0:NgalA], pot[0:NgalA]
        positionB[i,:,:], velocityB[i,:,:], potentialB[i,:,:] = pos[NgalA:N], vel[NgalA:N], pot[NgalA:N]

    return [positionA, velocityA, potentialA, positionB, velocityB, potentialB]

def computeTotalCOM(posA, posB, massA, massB):
    totalMass = np.sum(massA) + np.sum(massB)
    # broadcast masses along the snapshot axis
    massA = massA[np.newaxis, :, :]
    massB = massB[np.newaxis, :, :]
    # compute the total COM
    COM  = (np.sum(posA * massA, axis=1) + np.sum(posB * massB, axis=1)) / totalMass

    return COM

def computeCOM(pos, vel, mass):
    COM  = np.sum(pos*mass, axis = 0)/np.sum(mass)
    COMv = np.sum(vel*mass, axis = 0)/np.sum(mass)

    return COM, COMv

def centerOfMass(pos, vel, mass, timesteps, tempResolution):
    # number of snapshots
    Ntimesteps = int( (timesteps/tempResolution) + 1)
    # 3 dimensions
    Ncoord     = 3
    # allocate arrays for data
    COM  = np.zeros(shape = (Ntimesteps, 1, Ncoord) )
    COMv = np.zeros(shape = (Ntimesteps, 1, Ncoord) )
    # loop through snapshots and compute COM, COMv
    for i in range(0, Ntimesteps, 1):
        COM[i,:,:], COMv[i,:,:] = computeCOM(pos[i,:,:], vel[i,:,:], mass)

    return COM, COMv

def animateHexbin(galaxyA, galaxyB, start, stop, scale, gridRes, alpha, cmap = ['cividis', 'magma']):
    posA, velA, massA = galaxyA
    posB, velB, massB = galaxyB
    cmapA, cmapB = cmap
    # simulation box bounds
    extentMag = 100
    extentVec = [-extentMag, extentMag, -extentMag, extentMag]
    # figure params
    fig    = plt.figure(figsize = (10, 10))
    camera = Camera(fig) # define celluloid camera
    plt.rcParams['font.family'] = 'Courier New' # set font
    plt.minorticks_on()
    plt.tick_params(axis = 'both', length = 5, direction = 'in', which = 'both', right = True, top = True)
    plt.xlabel('X', size = 13)
    plt.ylabel('Y', size = 13)
    plt.xlim(-scale,scale)
    plt.ylim(-scale,scale)
    for i in range(start, stop):
        plt.hexbin(posA[i,:,0], posA[i,:,1], gridsize = gridRes, extent = extentVec, bins = 'log', cmap = cmapA)
        plt.hexbin(posB[i,:,0], posB[i,:,1], gridsize = gridRes, extent = extentVec, bins = 'log', cmap = cmapB, alpha = alpha)
        camera.snap()
    plt.close()

    return camera

#####################################################################################
# ANIMATION PARAMS: CHANGE THESE VALUES
# number of particles
N              = 23000
# number of particles in galaxyA
NgalA          = 13000
# number of timesteps
timesteps      = 7000
# number of time steps between each recorded snapshot of the simulation
tempResolution = 10
# path to data directory
path2Files     = 'disk_elliptcal/disk_ellip7_{}.npy'
path2Masses    = 'masses_ellip3.npy'
# animation start frame
animStart      = 0
# animation end frame
animEnd        = int(timesteps/tempResolution + 1)
# animation duration [seconds]
animLength     = 10
# animation name
animName       = 'galaxyHexAnim4.mp4'
# center of mass frame
comFrame = True

# plot params
#####################################################################################
# figure scale xlim ∈ [-scale, scale]; ylim ∈ [-scale, scale]
scale = 70
# grid resolution
gridRes = 200
# cmaps
cmap = ['cividis', 'magma']
# transparency of second galaxy
alpha = 0.85

#####################################################################################
# create animation
# load data as 2 separate galaxies
galaxyData    = load2Galaxies(path2Files, timesteps, N, NgalA, tempResolution)
posA, velA, potA, posB, velB, potB = galaxyData
# load masses and separate into corresponding arrays
mass          = np.load(path2Masses)
massA, massB  = mass[0:NgalA], mass[NgalA:N]
# group by galaxy
galaxyA       = [posA, velA, massA]
galaxyB       = [posB, velB, massB]
# if true, convert to COM frame of reference by subtracting COM from positions
if comFrame == True:
    # compute center of mass
    print('computing center of mass')
    COM = computeTotalCOM(posA, posB, massA, massB)
    # convert to center of mass frame
    for i in range(COM.shape[0]):
            posA[i, :, :] -= COM[i]
            posB[i, :, :] -= COM[i]
print('generating animation, sozz pls b patient ₊˚♡₊')
camera = animateHexbin(galaxyA, galaxyB, animStart, animEnd, scale, gridRes, alpha, cmap = cmap)
animation = camera.animate() # compile snapshots
animation.save(animName, fps = animEnd/animLength)
plt.close() #Stop the empty plot from displaying
exit()
