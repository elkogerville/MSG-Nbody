def load_galaxies(directory, timesteps, N, Ngal1):
    '''loops through each simulation snapshot file and loads it into a NumPy array
    ---------------------------------------------------------------------------------------------------------------------
    directory [string]: directory where snapshots are stored on computer ex: 'guide_snapshots/test_case_snapshots_{}.npy'
        ensure that '{}' is in the filename as a placeholder for the timestep number
    timesteps [integer]: number of timesteps
    N [integer]: total number of particles
    Ngal1 [integer]: total number of particles in galaxy one
    OUTPUT [NumPy array]: position, velocity, and potential arrays for each galaxy with shapes (timesteps x N x 3)
    '''
    mod = 10
    # number of timesteps counting timestep 0
    Ntimesteps = int((timesteps + 1)/mod) + 1
    N, Ngal1 = int(N), int(Ngal1)
    # number of particles in second galaxy
    Ngal2 = N - Ngal1
    # number of position/velocity dimensions
    Ncoord = 3
    # create arrays for position, velocity, and potential
    POS_gal1, VEL_gal1 = np.zeros(shape = (Ntimesteps, Ngal1, Ncoord)), np.zeros(shape = (Ntimesteps, Ngal1, Ncoord))
    POT_gal1 = np.zeros(shape = (Ntimesteps, Ngal1, 1))
    POS_gal2, VEL_gal2 = np.zeros(shape = (Ntimesteps, Ngal2, Ncoord)), np.zeros(shape = (Ntimesteps, Ngal2, Ncoord))
    POT_gal2 = np.zeros(shape = (Ntimesteps, Ngal2, Ncoord))
    # function for reading in data
    def load_data(directory, i):
        '''returns position, velocity, and potential at given timestep'''
        data = np.load(directory.format(i))
        return data[:,0:3], data[:,3:6], data[:,6:7]
    for i in tqdm(range(0, Ntimesteps, 1)):
        # load data from timestep
        pos, vel, pot = load_data(directory, i*mod)
        # seperate data into seperate galaxies
        pos_g1, vel_g1, pot_g1 = pos[0:Ngal1], vel[0:Ngal1], pot[0:Ngal1]
        pos_g2, vel_g2, pot_g2 = pos[Ngal1:N], vel[Ngal1:N], pot[Ngal1:N]
        # save data into array
        POS_gal1[i,:,:], POS_gal2[i,:,:] = pos_g1, pos_g2
        VEL_gal1[i,:,:], VEL_gal2[i,:,:] = vel_g1, vel_g2
        POT_gal1[i,:,:], POT_gal2[i,:,:] = pot_g1, pot_g2
    return POS_gal1, VEL_gal1, POT_gal1, POS_gal2, VEL_gal2, POT_gal2

def display(galaxies, scale = 100, savefig = False):
    """display a simulation snapshot using matplotlib
    ----------------------------------------------------------------------------------------
    galaxies [list of arrays]: list of galaxy positions; e.i. galaxies = [host_pos, sat_pos]
    scale [float]: positive and negative axes limits
    savefig [boolean]: set to True to save figure to directory as a png
    OUTPUT [matplotlib figure]: displays the positions of the galaxies in a plot
    """
    # setup figure with 2 subplots, one for plotting x,y projection, the other for plotting x,z projection
    fig, ax = plt.subplots(1,2, figsize = (10,5))
    # format axes, minorticks, fonts, and plot colors
    ax[0].minorticks_on()
    ax[1].minorticks_on()
    ax[0].tick_params(axis = 'both', length = 5, direction = 'in', which = 'both', right = True, top = True)
    ax[1].tick_params(axis = 'both', length = 5, direction = 'in', which = 'both', right = True, top = True)
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams['font.family'] = 'Courier New'
    plt.rcParams['mathtext.default'] = 'regular'
    color = ['darkslateblue', 'purple', 'mediumslateblue', 'orchid', 'black']
    # plot each array in the galaxies list
    for i, g in enumerate(galaxies):
        # plot x,y projection
        ax[0].scatter(g[:,0], g[:,1], s = .11, c = color[i])
        # plot x,z projection
        ax[1].scatter(g[:,0], g[:,2], s = .11, c = color[i])
    # set axes limits
    ax[0].set_xlim(-scale, scale)
    ax[0].set_ylim(-scale, scale)
    ax[1].set_xlim(-scale, scale)
    ax[1].set_ylim(-scale, scale)
    # set axes labels
    ax[0].set_xlabel(r'X', size = 17)
    ax[1].set_xlabel(r'X', size = 17)
    ax[0].set_ylabel(r'Y', size = 17)
    ax[1].set_ylabel(r'Z', size = 17)
    plt.tight_layout()
    # if savefig is True, save figure to directory
    if savefig is not False:
        # promt user to input file name
        file_name = input('please input filename for image (ex: myimage.png): ')
        plt.savefig(file_name, dpi = 300, format = 'png')
    plt.show()
    
def get_Energy(vel, pot):
    '''computes the relative Energy, epsilon, based on the velocity and potential'''
    epsilon = pot - (1/2)*(vel[:,0:1]**2 + vel[:,1:2]**2 + vel[:,2:3]**2)
    return epsilon

def plot_Ne(Energies, labels, savefig = False, bin_m = -3,  bin_M = .1):
    # setup figure
    plt.figure(figsize = (5,5))
    plt.rcParams['font.family'] = 'Courier New'
    plt.minorticks_on()
    plt.tick_params(axis = 'both', length = 5, direction = 'in', which = 'both', right = True, top = True)
    plt.rcParams['axes.linewidth'] = 0.6
    ls = ['solid', '--', 'dotted', 'dashdot']
    color = ['k', 'darkslateblue', 'mediumslateblue', 'purple']
    
    # normalized simulated data
    for i, g in enumerate(Energies):
        hist, edges = np.histogram(g, bins = np.logspace(bin_m, bin_M, 65))
        center = (edges[1:] + edges[:-1])/2
        plt.step(center, hist/np.max(hist), color = color[i], lw = .6, ls = ls[i], label = labels[i])
    
    plt.xlabel('E', size = 13)
    plt.ylabel('N(E)', size = 13)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    if savefig is True:
        file_name = input('please input filename for image (ex: myimage.png): ')
        plt.savefig(file_name, dpi = 300, format = 'png')
    plt.show()
