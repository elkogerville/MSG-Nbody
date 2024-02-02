import numpy as np
from tqdm import tqdm
from numba import njit
import matplotlib.pyplot as plt
"""
this program integrates the orbits of a 10:1 face on minor merger of a hernquist
spherical galaxy and a disk galaxy.
"""
###########
# FUNCTIONS
###########
def load_phase(initial_phase):
    """load initial position, velocity, and mass phase space arrays"""
    phase_space = np.loadtxt(initial_phase)
    print('total initial mass of galaxy: {}'.format(np.sum(phase_space[:,6:7])))
    return phase_space[:,0:3], phase_space[:,3:6], phase_space[:,6:7]
def timestep_loader(string, timestep):
    data = np.load(string.format(timestep))
    N = data.shape[0]
    data = data.reshape(N,7)
    return data[:, 0:3], data[:, 3:6], data[:, 6:7]
def display(galaxies, savefig = False, scale = 100):
    """galaxies = list of galaxy initial conditions"""
    fig, ax = plt.subplots(1,2, figsize = (10,5))
    ax[0].minorticks_on()
    ax[1].minorticks_on()
    ax[0].tick_params(axis = 'both', length = 5, direction = 'in', which = 'both', right = True, top = True)
    ax[1].tick_params(axis = 'both', length = 5, direction = 'in', which = 'both', right = True, top = True)
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['font.family'] = 'Courier New'
    plt.rcParams['mathtext.default'] = 'regular'
    color = ['darkslateblue', 'purple', 'mediumslateblue', 'orchid', 'black']
    for i, g in enumerate(galaxies):
        print('number of particles: {}'.format(g.shape))
        ax[0].scatter(g[:,0], g[:,1], s = .11, c = color[i])
        ax[1].scatter(g[:,0], g[:,2], s = .11, c = color[i])
    ax[0].set_xlim(-scale, scale)
    ax[0].set_ylim(-scale, scale)
    ax[1].set_xlim(-scale, scale)
    ax[1].set_ylim(-scale, scale)
    ax[0].set_xlabel(r'X', size = 17)
    ax[1].set_xlabel(r'X', size = 17)
    ax[0].set_ylabel(r'Y', size = 17)
    ax[1].set_ylabel(r'Z', size = 17)
    plt.tight_layout()
    if savefig is not False:
        file_name = input('filename for image (png): ')
        plt.savefig(file_name, dpi = 1000, format = 'png')
    plt.show()

@njit
def accel_potential(position, mass, Sf, mT):
    """this function returns the gravitational acceleration from orbiting masses"""
    G = 1
    # seperate position into x,y,z components
    x = position[:, 0:1]
    y = position[:, 1:2]
    z = position[:, 2:3]
    # calculate particle-particle seperations
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

def MSGnbody(gal_pos, gal_vel, mass, dt, timesteps, string, **kwargs):
    """string format: 'directory/filename{}.npy' """
    import numpy as np
    from tqdm import tqdm
    
    # save every 10 timesteps
    mod = 10
    if 'mod' in kwargs:
        mod = kwargs['mod']
    
    # quantize timesteps
    timesteps = int(timesteps)
    def get_softening(N):
        N5 = N/1e5
        return 0.017 * (N5)**(-0.23)
    # calculate softening parameter:    
    soft_param = get_softening(gal_pos.shape[0])
    # calculate particle-particle mass products
    mass_product = mass*mass.T
    # calculate initial accelerations 
    gal_accel, gal_potential = accel_potential(gal_pos, mass, soft_param, mass_product)
    
    # save initial conditions
    phase_space0 = np.hstack((gal_pos, gal_vel, gal_potential))
    np.save(string.format(0), phase_space0)
    np.save('masses26k.npy', mass)
    
    # simulation loop
    print('simulation running....  /ᐠ –ꞈ –ᐟ\<[pls be patient]')  
    for n in tqdm(range(1, timesteps+1)):       
        # GALAXIES
        # 1/2 kick
        gal_vel += gal_accel * dt/2.0

        # drift
        gal_pos += gal_vel * dt 
        
        # update accelerations
        gal_accel, gal_potential = accel_potential(gal_pos, mass, soft_param, mass_product)
        
        # update velocities
        gal_vel += gal_accel * dt/2.0
        
        # store phase space from timestep into array
        if n % mod == 0:
            phase_space = np.hstack((gal_pos, gal_vel, gal_potential))
            np.save(string.format(n), phase_space)
    
    return 'simulation complete [yay!!! (ﾐΦ ﻌ Φﾐ)✿ *ᵖᵘʳʳ*]'

###############################################################################################################
# MAIN SIMULATION LOOP AND PARAMETERS
###############################################################################################################
# remember to change following items: spherical/disk initial conditions,
# mass file name, timestep file name, perturber galaxy initial pos and vel,
# number of timesteps, dt, snapshot directory, mod number

# load spherical galaxy initial conditions N = 13000, M = 1, a = 1
pos_sphr, vel_sphr, mas_sphr = load_phase('phase13k.txt')
# scale mass and position of galaxy
def scale_phase(pos, vel, mas, R, M):
    """scale position, velocity, and mass by scalars R and M"""
    G = 1
    pos = pos * R
    vel = vel * np.sqrt(G*M/R)
    mas = mas * M
    return pos, vel, mas

pos_sphr, vel_sphr, mas_sphr = scale_phase(pos_sphr, vel_sphr, mas_sphr, R = 2, M = 10)
print('final hernquist galaxy mass: {}'.format(np.sum(mas_sphr)))

# load disk galaxy initial conditions
# initial unormalized agama disk masses
pos_disk, vel_disk, mas_disk = load_phase('model_disk_final13k')
# normalize mass of disk galaxy
initial_mass = np.sum(mas_disk)
mas_disk = mas_disk / initial_mass
pos_disk = pos_disk / initial_mass
print('final disk galaxy mass: {}'.format(np.sum(mas_disk)))

# add radial velocity to perturber using escape speed 
def escape_v(x, y, z, M):
    G = 1
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.sqrt(2*G*M/r)
# print escape velocity
print('escape velocity: {}'.format(escape_v(45,10,0,10)))

# move perturber to intial position and give it initial velocity
# -45 in the z direction with intial velocity of +0.5 in z direction
#              x     y     z
pos_disk += [45.0, 10.0,  0.0]
vel_disk += [-0.5, 0.0,   0.0]

# merge initial phase space arrays into contigous arrays
def merger_init(pos1, vel1, mas1, pos2, vel2, mas2):
    pos = np.append(pos1, pos2, axis = 0)
    vel = np.append(vel1, vel2, axis = 0)
    mas = np.append(mas1, mas2, axis = 0)
    positions = np.ascontiguousarray(pos)
    velocities = np.ascontiguousarray(vel)
    masses = np.ascontiguousarray(mas)
    del pos, vel, mas, pos1, vel1, mas1, pos2, vel2, mas2

    return positions, velocities, masses

positions, velocities, masses = merger_init(pos_sphr, vel_sphr, mas_sphr, pos_disk, vel_disk, mas_disk)
print(positions.shape, velocities.shape, masses.shape, np.sum(masses))
# display initial conditions
display([positions], savefig = False, scale = 50)

# RUN NBODY SIMULATION
MSGnbody(positions, velocities, masses, 0.05, 7000, '26kmerger/26kmerger_radial_edgeon_{}.npy', mod = 5)
