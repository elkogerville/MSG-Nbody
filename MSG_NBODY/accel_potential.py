from numba import njit
@njit
def accel_potential(position, mass, Sf, mT):
    '''calculates the gravitational acceleration acting onto each particle as well as the potential of each particle
    ---------------------------------------------------------------------------------------------------------------------------------
    position [NumPy array]: N x 3 array containing the x,y,z position of each particle in the simulation
    mass [NumPy array]: N x 1 array containing the mass of each particle in the simulation
    Sf [float]: softening length, this value is precomputed by the program and doesn't require user input
    mT [NumPy array]: N x N array of every mass-mass product, this value is precomputed by the program and doesn't require user input
    OUTPUT [NumPy array]: N x 3 array containing the gravitational acceleration onto each particle in the simulation
    '''
    import numpy as np
    G = 1
    # seperate position into x,y,z components
    x = position[:, 0:1]
    y = position[:, 1:2]
    z = position[:, 2:3]
    # calculate particle-particle separations
    delx = x.T - x
    dely = y.T - y
    delz = z.T - z
    delr = np.sqrt(delx**2 + dely**2 + delz**2 + Sf**2)
    
    # calculate acceleration
    g = delr**(-3)
    accelx = G * (delx * g) @ mass
    accely = G * (dely * g) @ mass
    accelz = G * (delz * g) @ mass
    
    # calculate (N x N) particle-particle potential matrix
    potential_N = (G * mT) / delr
    potential_sum = np.sum(potential_N, axis = 0).reshape(position.shape[0], 1)
    # subtract diagonal elements
    potential = potential_sum - np.diag(potential_N).reshape(position.shape[0], 1)
    # divide by mass of each particle
    potential = potential / mass
    
    # return (N x 3) acceleration matrix and (N x 1) potential matrix
    return np.hstack((accelx, accely, accelz)), potential
