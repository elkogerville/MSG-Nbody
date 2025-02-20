@njit()
def accel_potential(pos, mass, accel, softening):
    '''calculates the gravitational acceleration acting onto each particle as well as the potential of each particle
    ---------------------------------------------------------------------------------------------------------------------------------
    pos [NumPy array]: N x 3 array containing the x,y,z position of each particle in the simulation
    mass [NumPy array]: N x 1 array containing the mass of each particle in the simulation
    accel [NumPy array]: N x 3 array to store the acceleration due to gravity of each particle
    softening [float]: softening length, this value is precomputed by the program and doesn't require user input
    OUTPUT [NumPy array]: N x 3 array containing the gravitational acceleration onto each particle in the simulation
    '''
    G = 1
    # seperate position into x,y,z components
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    # calculate particle-particle seperations
    delx = x.T - x
    dely = y.T - y
    delz = z.T - z
    delr = np.sqrt(delx**2 + dely**2 + delz**2 + softening**2)
    g = delr**(-3)
    # calculate acceleration
    accel[:, 0:1] = G * np.dot((delx * g), mass)
    accel[:, 1:2] = G * np.dot((dely * g), mass)
    accel[:, 2:3] = G * np.dot((delz * g), mass)
    
    # calculate (N x N) particle-particle potential matrix
    potential_N = (G * mass.T) / delr
    potential_sum = np.sum(potential_N, axis = 0).reshape(pos.shape[0], 1)
    # subtract diagonal elements
    potential = potential_sum - np.diag(potential_N).reshape(pos.shape[0], 1)
    
    # return (N x 3) acceleration matrix and (N x 1) potential matrix    
    return accel, potential
