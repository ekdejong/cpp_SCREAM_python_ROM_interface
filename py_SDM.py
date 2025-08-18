def compute_coll_SDM(numbin, dt, MDdD_bin, NDdD_bin):
    """
    simple fake code for test only

    Within this function, one should call the true SDM processes modules.
    """

    import numpy as np
    import sys

    SDM_MDdD_bin = np.zeros((numbin))
    SDM_NDdD_bin = np.zeros((numbin))

    ### if dsd_hlp = MDdD_bin, then dsd_hlp is a reference to MDdD_bin (in my understanding, it's like a pointer). That means changing dsd_hlp would also change MDdD_bin too. That means in py_SDM lines 147-150, values = 0. solution: dsd_hlp must be referring to a new reference. e.g., dsd_hlp = MDdD_bin[:]; or see below.
    dsd_hlp      = np.zeros((numbin))
    ndsd_hlp     = np.zeros((numbin))
    
    dsd_hlp[:]  = MDdD_bin
    ndsd_hlp[:] = NDdD_bin

    SDM_deltat = 5.0               # unit: sec
    if dt % SDM_deltat == 0:
        iter = int(dt/SDM_deltat)  # number of iteration; time step of SDM = dt/iter
    else:
        print('dt is not completely divisible by SDM_deltat. Check!')
        sys.exit()
    
    for ii in range(iter):
        # print('iter= ', ii)
        
        ### fake code block
        for bb in range(numbin-1):
            dsd_hlp[bb+1] += dsd_hlp[bb]*0.1
            dsd_hlp[bb] *= 0.9

            ndsd_hlp[bb+1] += ndsd_hlp[bb]*0.2*0.2*0.2
            ndsd_hlp[bb] *= 0.8
        ### fake code block
        
    SDM_MDdD_bin = dsd_hlp
    SDM_NDdD_bin = ndsd_hlp

    return SDM_MDdD_bin, SDM_NDdD_bin

    