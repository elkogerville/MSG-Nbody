def MSGnbody(gal_pos, gal_vel, mass, dt, timesteps, directory, **kwargs):
    '''integrates the motion of N particles under their mutual gravitational attraction through space and time
    --------------------------------------------------------------------------------------------------------------------------
    gal_pos [NumPy array]: N x 3 position array of all particles
    gal_vel [NumPy array]: N x 3 velocity array of all particles
    mass [NumPy array]: N x 1 mass array of all particles
    dt [float]: size of timestep, recommended dt = 0.01
    timesteps [integer]: number of timesteps
    directory [string]: directory path where snapshots are to be saved, including the filename
        best practice is to create a folder within the code directory; the filename should have '{}' before the .npn extension
        example: directory = 'folder/filename{}.npy' 
    **kwargs: mod [integer]: by default, simulation saves every 10 timesteps but can be changed by passing the mod argument
    OUTPUT [.npn file]: the program saves a simulation snapshot of the positions, velocities, and potential every 10 timesteps
    '''  
    ##################
    # simulation setup
    ##################
    import numpy as np
    # save every 10 timesteps
    mod = 10
    if 'mod' in kwargs:
        mod = kwargs['mod']
    # quantize mod
    mod = int(mod)
    # quantize timesteps
    timesteps = int(timesteps)
    N = gal_pos.shape[0]
    # enforce appropriate array dimensions
    if (gal_pos.shape != (N, 3)) & (gal_vel.shape != (N, 3)) & (mass.shape != (N, 3)):
        print('ERROR: please ensure gal_pos and gal_vel have shapes (N x 3) and mass has shape (N x 1) where N is the total number of particles. gal_pos shape: {} '.format(gal_pos.shape), 'gal_vel shape: {}'.format(gal_vel.shape), 'mass shape: {}'.format(mass.shape), '\n /ᐠ_ ꞈ _ᐟ\ <(fix it...)')
        raise SystemExit
    
    # calculate softening parameter: 
    def get_softening(N):
        '''calculate the softening length as a function of N based on Dehnen (2001)'''
        N5 = N/1e5
        return 0.017 * (N5)**(-0.23)
    soft_param = get_softening(gal_pos.shape[0])
    
    # calculate particle-particle mass products
    mass_product = mass*mass.T
    # calculate initial accelerations 
    gal_accel, gal_potential = accel_potential(gal_pos, mass, soft_param, mass_product)
    
    # save initial conditions
    phase_space0 = np.hstack((gal_pos, gal_vel, gal_potential))
    np.save(directory.format(0), phase_space0)
    np.save('masses.npy', mass)
    
    #################
    # simulation loop
    #################
 
    print('simulation running....  /ᐠ –ꞈ –ᐟ\<[pls be patient]')  
    for n in tqdm(range(1, timesteps+1)):       
        # 1/2 kick
        gal_vel += gal_accel * dt/2.0

        # drift
        gal_pos += gal_vel * dt 
        
        # update accelerations
        gal_accel, gal_potential = accel_potential(gal_pos, mass, soft_param, mass_product)
        
        # update velocities
        gal_vel += gal_accel * dt/2.0
        
        # store phase space from timestep as snapshot saved to directory
        if n % mod == 0:
            phase_space = np.hstack((gal_pos, gal_vel, gal_potential))
            np.save(directory.format(n), phase_space)
    
    return 'simulation complete [yay!!! (ﾐΦ ﻌ Φﾐ)✿ *ᵖᵘʳʳ*]'
