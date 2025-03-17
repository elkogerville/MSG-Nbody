import numpy as np
from numba import njit

@njit()
def compute_accel_potential(pos, mass, accel, softening_sq, N):
    '''
    computes the gravitational acceleration and potential for each particle due to all others
    using softened Newtonian gravity
    Parameters
    ----------
    pos: np.ndarray[np.float64]
        Nx3 array containing the [x, y, z] positions of all particles
    mass: np.ndarray[np.float64]
        Nx1 array containing the mass of each particle
    accel: np.ndarray[np.float64]
        Nx3 array to store the computed gravitational acceleration [ax, ay, az] for each particle
    softening_sq: float
        square of softening length to prevent division by zero and to define the simulation resolution
        sqrt(softening) is the closest encounter the simulation can resolve
    N: integer
        number of particles in simulation
    Returns
    -------
    accel: np.ndarray[np.float64]
        Nx3 array of the computed gravitational acceleration [ax, ay, az] for each particle
    potential: np.ndarray[np.float64]
        Nx1 array of the gravitational potential experienced by each particle due to all other particles
    '''
    G = 1
    # seperate position into x,y,z components
    x = pos[:, 0][:, np.newaxis]
    y = pos[:, 1][:, np.newaxis]
    z = pos[:, 2][:, np.newaxis]
    # calculate particle-particle seperations
    delx = x.T - x
    dely = y.T - y
    delz = z.T - z
    delr = np.sqrt(delx**2 + dely**2 + delz**2 + softening_sq)
    inv_r3 = delr**(-3)
    # calculate acceleration
    accel[:, 0] = ( G * np.dot((delx * inv_r3), mass) )[:,0]
    accel[:, 1] = ( G * np.dot((dely * inv_r3), mass) )[:,0]
    accel[:, 2] = ( G * np.dot((delz * inv_r3), mass) )[:,0]
    
    # calculate (N x N) particle-particle potential matrix
    potential_N = (G * mass.T) / delr
    potential_sum = np.sum(potential_N, axis = 0).reshape(N, 1)
    # subtract diagonal elements
    potential = potential_sum - np.diag(potential_N).reshape(N, 1)
    
    # return (N x 3) acceleration matrix and (N x 1) potential matrix    
    return accel, potential
