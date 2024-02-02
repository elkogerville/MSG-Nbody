def load_phase(initial_phase):
    '''load initial conditions and separate into position, velocity, and mass arrays
    --------------------------------------------------------------------------------------
    initial_phase [NumPy array]: initial N x 7 array of positions, velocities, and masses
    OUTPUT [NumPy array]: N x 3 position array, N x 3 velocity array, and N x 1 mass array
    '''
    import numpy as np
    # read initial conditions into Python
    phase_space = np.loadtxt(initial_phase)
    # index the initial N x 7 array into respective arrays and return
    return phase_space[:,0:3], phase_space[:,3:6], phase_space[:,6:7]

def scale_phase(pos, vel, mas, R, M):
    '''scale position, velocity, and mass by scalars R and M
    --------------------------------------------------------------------------
    pos, vel, mas [NumPy array]: initial position, velocity, and mass arrays
    R, M [float]: quantities to scale the positions, velocities, and masses by
    OUTPUT [NumPy array]: scaled position, velocity, and mass arrays
    '''
    import numpy as np
    G = 1 
    # scale position, velocity, and mass by scalar quantities. the velocities are scaled proportionally to a circular orbit velocity
    pos = pos * R
    vel = vel * np.sqrt(G*M/R)
    mas = mas * M
    return pos, vel, mas

# rotate perturber galaxy
def rotate_disk(pos, vel, deg, axis):
    '''rotate positions and velocities of galaxy about a specified axis
    -------------------------------------------------------------------------------
    pos, vel [NumPy array]: N x 3 position and velocity arrays
    deg [float]: degrees to rotate by
    axis [string]: set this argument to 'x', 'y', or 'z' to rotate around that axis
    OUTPUT [NumPy array]: rotated position and velocity arrays
    '''
    import numpy as np
    # convert degrees to radians
    rad = np.deg2rad(deg)
    # define sin and cosine functions
    sin, cos = np.sin, np.cos
    # define x, y, z rotation matrices
    rx = np.array([[1, 0, 0],[0, cos(rad), -sin(rad)], [0, sin(rad), cos(rad)]])
    ry = np.array([[cos(rad), 0, sin(rad)],[0, 1, 0], [-sin(rad), 0, cos(rad)]])
    rz = np.array([[cos(rad), -sin(rad), 0],[sin(rad), cos(rad), 0], [0, 0, 1]])
    # ensure position and velocity arrays are appropriate shapes for dot product
    N = pos.shape[0]
    pos, vel = pos.reshape(N,3), vel.reshape(N,3)
    # rotate around specified axis based on user input
    if axis == 'x' or axis == 'X':
        # dot product of positions and velocities with x rotation matrix
        rotated_pts = pos @ rx
        rotated_vel = vel @ rx
    if axis == 'y' or axis == 'Y':
        # dot product of positions and velocities with y rotation matrix
        rotated_pts = pos @ ry
        rotated_vel = vel @ ry
    if axis == 'z' or axis == 'Z':
        # dot product of positions and velocities with z rotation matrix
            rotated_pts = pos @ rz
            rotated_vel = vel @ rz
    return rotated_pts, rotated_vel

def escape_v(x, y, z, M):
    '''calculate the escape velocity of the satellite galaxy
    -----------------------------------------------------------------------------------------------------------------------
    x, y, z [float]: x, y, z position of the center of mass of the satellite galaxy
    M [float]: total mass of host galaxy
    OUTPUT [float]: magnitude of the escape velocity of the satellite galaxy at the specified distance from the host galaxy
    '''
    import numpy as np
    G = 1
    # find length of vector between origin and 
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.sqrt(2*G*M/r)

def merger_init(pos1, vel1, mas1, pos2, vel2, mas2):
    '''merge the position, velocity, and mass arrays of two different galaxies into contiguous position, velocity, and mass arrays
    ------------------------------------------------------------------------------------------------------------------------------
    pos1, vel1, mas1 [NumPy array]: position, velocity, and mass arrays of the first galaxy
    pos2, vel2, mas2 [NumPy array]: position, velocity, and mass arrays of the second galaxy
    OUTPUT [NumPy array]: contiguous position, velocity, and mass arrays containing both galaxies
    '''
    import numpy as np
    # append the second galaxy to the first
    pos = np.append(pos1, pos2, axis = 0)
    vel = np.append(vel1, vel2, axis = 0)
    mas = np.append(mas1, mas2, axis = 0)
    # create contiguous arrays
    positions = np.ascontiguousarray(pos)
    velocities = np.ascontiguousarray(vel)
    masses = np.ascontiguousarray(mas)

    return positions, velocities, masses

