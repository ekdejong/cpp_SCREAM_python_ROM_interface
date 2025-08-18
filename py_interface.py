def SDM_interface(qc_in, nc_in, qr_in, nr_in, muc_in, mur_in, qsmall):
    """
    Description: A python interface that links to SDM python.
    Here reads in cloud variables from C++, creates assumed gamma size distributions based on the bulk properties, put into bins, and passes the bin distribution to SDM calculations, and retrieves the calculated rain tendency rate terms due to collision-coalescence and passes them back to C++.

    input: 
    qc_in: cloud liquid water content, kg/m3
    nc_in: cloud liquid droplet number concentration, 1/m3
    qr_in: rain water content, kg/m3
    nr_in: raindrop number concentration, 1/m3
    muc_in: spectral width of assumed gamma size distribution function for cloud liquid water, 
    mur_in: spectral width of assumed gamma size distribution function for rain,
    qsmall: low bound threshold.

    output:
    qc_tend_out: time rate of change of cloud liquid water due to collision-coalescence for a given time step, kg/m3/s
    nc_tend_out: time rate of change of cloud liquid droplet number due to collision-coalescence for a given time step, 1/m3/s
    qr_tend_out: time rate of change of rain due to collision-coalescence for a given time step, kg/m3/s
    nr_tend_out: time rate of change of raindrop number due to collision-coalescence for a given time step, 1/m3/s
    """

    import math
    import numpy as np
    import sys
    import py_SDM

    rhow = 1000.0   # liquid water density, unit: kg/m3

    ### flags controlling plotting
    make_plot_c     = False
    make_plot_r     = True
    make_plot_total = True
    

    ### a scalar
    Nc = nc_in
    Qc = qc_in
    muc= muc_in

    Nr = nr_in
    Qr = qr_in
    mur= mur_in
    
    ### convert bulk aggregated properties to bin necessary for SDM calculations
    
    ### create bin
    # bin, edge number
    numbin = 80
    numedg = 80 + 1

    # bin edge mass, size
    q_edg  = np.zeros((numedg))    # unit: mg
    r_edg  = np.zeros((numedg))    # unit: um
    
    # bin width
    lSCAL  = 2.0                  # controlling grid mesh resolution
    lalpha = 2.0                  # mass doubling (2), tripling (3), quadrupling (4), etc. every lscal bin(s)
    ax     = lalpha**(1.0/lSCAL)  # bin discretization

    # fill in each bin
    mass_min = 1e-9               # unit: mg
    
    for ii in range(numedg):
        # 1st bin
        if ii == 0:
            q_edg[ii] = mass_min*0.5*(ax+1.)                               # unit: mg
            r_edg[ii] = 1000.*np.exp(np.log(3.0*q_edg[ii]/(4.0*np.pi))/3.0)  # unit: um
        # 2nd bin and beyond
        else:
            q_edg[ii] = q_edg[ii-1] * ax
            r_edg[ii] = 1000.*np.exp(np.log(3.0*q_edg[ii]/(4.0*np.pi))/3.0)

    # bin center mass, size
    q_cen = np.zeros((numbin))    # unit: mg
    r_cen = np.zeros((numbin))    # unit: um
    D_cen = np.zeros((numbin))    # unit: meter
    deltad= np.zeros((numbin))    # unit: meter

    q_cen[:] = ((q_edg + np.roll(q_edg, 1))/2.0)[1:]
    r_cen[:] = 1000.*np.exp(np.log(3.0*q_cen[:]/(4.0*np.pi))/3.0)
    D_cen[:] = r_cen[:]*2.0*1e-6  # diameter
    deltad[:]= (r_edg - np.roll(r_edg,1))[1:] *2.0*1e-6

    ### assumed gamma size distribution
    lambdac = 0.0
    N0c     = 0.0
    lambdar = 0.0
    N0r     = 0.0
    gamma_psdc_bin = np.zeros((numbin))
    gamma_psdr_bin = np.zeros((numbin))
    NDdD_bin       = np.zeros((numbin))
    MDdD_bin       = np.zeros((numbin))

    if qc_in > qsmall:
        lambdac = (np.pi*rhow*Nc*math.gamma(muc+4)/(6.0*Qc*math.gamma(muc+1)))**(1.0/3.0)                # unit: meter -1
        #lambdac = np.max([(muc+1.)*2.5e4,np.min([lambdac,(muc+1.)*1.0e6])])
        N0c     = Nc*lambdac**(muc+1)/math.gamma(muc+1)   # unit: m^(-u-4)
        print('cloud liquid N0, lambda= ', N0c, lambdac)
    
        gamma_psdc_bin[:] = N0c*(D_cen[:]**muc)*np.exp(-lambdac*D_cen[:])   # unit: m-4

        ####################################################################
        if make_plot_c:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            
            ### bin
            dum_NDdD_bin = gamma_psdc_bin[:]*deltad[:]        # unit: m-3
            dum_MDdD_bin = dum_NDdD_bin[:]*q_cen[:]*1e-6      # unit:kg m-3

            ### gamma PSD
            powlaw_coeff = np.linspace(start=-9,stop=-1,num=100,endpoint=True)
            dd           = 10.0**powlaw_coeff 

            deltadd = (dd - np.roll(dd,1))[1:]
            centerdd= ((dd + np.roll(dd,1))/2.0)[1:]

            gamma_psdc   = N0c*(centerdd[:]**muc)*np.exp(-lambdac*centerdd[:])      # unit: m-4-u
            dum_NDdDc    = gamma_psdc[:]*deltadd[:]                                 # unit: m-3
            dum_massDc   = dum_NDdDc[:]*(np.pi/6.0*centerdd[:]**3*rhow)             # unit: kg/m3. Assumption: spherical

            # line number and mass (true value)
            v,lr_left = find_nearest(centerdd,D_cen[0])
            v,lr_righ = find_nearest(centerdd,D_cen[-1])
            
            total_nl = np.sum(dum_NDdDc[lr_left:lr_righ+1])*1.0e-6  # unit: cm-3
            total_ml = np.sum(dum_massDc[lr_left:lr_righ+1])*1.0e-3 # unit: g cm-3
            
            # bin number and mass 
            total_nl_bin = np.sum(dum_NDdD_bin*1.0e-6)
            total_ml_bin = np.sum(dum_MDdD_bin*1.0e-3)
            
            dpi = 300
            fig = plt.figure(figsize=(1800/dpi,1200/dpi), dpi=dpi) ##,constrained_layout=True)
            spec= gridspec.GridSpec(ncols=2, nrows=2,figure=fig,left=0.1, bottom=0.1,right=0.95,top=0.95)
            spec.update(wspace=0.4,hspace=0.45)
            
            ax = []

            ax.append(fig.add_subplot(spec[0, 0]))
            # ax[-1].set_ylim(0,20)
            ax[-1].set_yscale('linear')
            ax[-1].set_ylabel('Number [cm$^{-3}$]',fontsize=10)
            ax[-1].set_xscale('log')
            #ax[-1].set_xlim(1.0e-10,1.0e4)
            ax[-1].set_xlabel('Diameter [\u03bcm]', fontsize=10)
            ax[-1].tick_params(axis='y', rotation=0,labelsize=7)
            ax[-1].tick_params(axis='x', rotation=0,labelsize=7)
            ax[-1].plot(centerdd*1.0e6,dum_NDdDc*1.0e-6,alpha=0.8,lw=1,linestyle='-',c='k',)
            ax[-1].bar(D_cen*1.0e6,dum_NDdD_bin*1.0e-6,width=deltad*1.0e6,edgecolor='k',linewidth=0.5)
            ax[-1].axvline(x=D_cen[0]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')
            ax[-1].axvline(x=D_cen[-1]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')
            ax[-1].text(0.1,0.9,'Given total Nc: '+str("{:.2e}".format(Nc*1.0e-6))+'cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.8,'sum total gamma Nc: '+str("{:.2e}".format(total_nl))+'cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.7,'sum total bin Nc: '+str("{:.2e}".format(total_nl_bin))+'cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.6,'RelDiff: '+str("{:.2f}".format(abs(Nc*1.0e-6-total_nl_bin)/(Nc*1.0e-6)*100.))+'%',fontsize=6,transform = ax[-1].transAxes)


            ax.append(fig.add_subplot(spec[1, 0]))
            # ax[-1].set_ylim(0,20.0e-8)
            ax[-1].set_yscale('linear')
            ax[-1].set_ylabel('Mass [g cm$^{-3}$]',fontsize=10)
            ax[-1].set_xscale('log')
            #ax[-1].set_xlim(1.0e-10,1.0e4)
            ax[-1].set_xlabel('Diameter [\u03bcm]', fontsize=10)
            ax[-1].tick_params(axis='y', rotation=0,labelsize=7)
            ax[-1].tick_params(axis='x', rotation=0,labelsize=7)
            ax[-1].plot(centerdd*1.0e6,dum_massDc*1.0e-3,alpha=0.8,lw=1,linestyle='-',c='k',)
            ax[-1].bar(D_cen*1.0e6,dum_MDdD_bin*1.0e-3,width=deltad*1.0e6,edgecolor='k',linewidth=0.5)
            ax[-1].axvline(x=D_cen[0]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')
            ax[-1].axvline(x=D_cen[-1]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')
            ax[-1].text(0.1,0.9,'Given total mc: '+str("{:.2e}".format(Qc*1.0e-3))+'g cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.8,'sum total gamma mc: '+str("{:.2e}".format(total_ml))+'g cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.7,'sum total bin mc: '+str("{:.2e}".format(total_ml_bin))+'g cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.6,'RelDiff: '+str("{:.2f}".format(abs(Qc*1.0e-3-total_ml_bin)/(Qc*1.0e-3)*100.))+'%',fontsize=6,transform = ax[-1].transAxes)

            plt.savefig('PSDc_comparison.png')
            plt.close()
        ####################################################################


    if qr_in > qsmall:
        lambdar = (np.pi*rhow*Nr*math.gamma(mur+4)/(6.0*Qr*math.gamma(mur+1)))**(1.0/3.0)                # unit: meter -1
        #lambdar = np.max([1./0.005,np.min([lambdar,1.0e5])])
        N0r     = Nr*lambdar**(mur+1)/math.gamma(mur+1)   # unit: m^(-u-4)
        print('rain N0r, lambda= ', N0r, lambdar)
        
        gamma_psdr_bin[:] = N0r*(D_cen[:]**mur)*np.exp(-lambdar*D_cen[:])   # unit: m-4

        ####################################################################
        if make_plot_r:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            
            ### bin
            dum_NDdD_bin = gamma_psdr_bin[:]*deltad[:]        # unit: m-3
            dum_MDdD_bin = dum_NDdD_bin[:]*q_cen[:]*1e-6      # unit:kg m-3

            ### gamma PSD
            powlaw_coeff = np.linspace(start=-9,stop=-1,num=100,endpoint=True)
            dd           = 10.0**powlaw_coeff 

            deltadd = (dd - np.roll(dd,1))[1:]
            centerdd= ((dd + np.roll(dd,1))/2.0)[1:]

            gamma_psdr   = N0r*(centerdd[:]**mur)*np.exp(-lambdar*centerdd[:])      # unit: m-4-u
            dum_NDdDr    = gamma_psdr[:]*deltadd[:]                                 # unit: m-3
            dum_massDr   = dum_NDdDr[:]*(np.pi/6.0*centerdd[:]**3*rhow)             # unit: kg/m3. Assumption: spherical

            # line number and mass (true value)
            v,lr_left = find_nearest(centerdd,D_cen[0])
            v,lr_righ = find_nearest(centerdd,D_cen[-1])
            
            total_nl = np.sum(dum_NDdDr[lr_left:lr_righ+1])*1.0e-6  # unit: cm-3
            total_ml = np.sum(dum_massDr[lr_left:lr_righ+1])*1.0e-3 # unit: g cm-3
            
            # bin number and mass 
            total_nl_bin = np.sum(dum_NDdD_bin*1.0e-6)
            total_ml_bin = np.sum(dum_MDdD_bin*1.0e-3)
            
            dpi = 300
            fig = plt.figure(figsize=(1800/dpi,1200/dpi), dpi=dpi) ##,constrained_layout=True)
            spec= gridspec.GridSpec(ncols=2, nrows=2,figure=fig,left=0.1, bottom=0.1,right=0.95,top=0.95)
            spec.update(wspace=0.4,hspace=0.45)
            
            ax = []

            ax.append(fig.add_subplot(spec[0, 0]))
            # ax[-1].set_ylim(0,10)
            ax[-1].set_yscale('linear')
            ax[-1].set_ylabel('Number [L$^{-1}$]',fontsize=10)
            ax[-1].set_xscale('log')
            #ax[-1].set_xlim(1.0e-10,1.0e4)
            ax[-1].set_xlabel('Diameter [\u03bcm]', fontsize=10)
            ax[-1].tick_params(axis='y', rotation=0,labelsize=7)
            ax[-1].tick_params(axis='x', rotation=0,labelsize=7)
            ax[-1].plot(centerdd*1.0e6,dum_NDdDr*1.0e-6*1e3,alpha=0.8,lw=1,linestyle='-',c='k',)
            ax[-1].bar(D_cen*1.0e6,dum_NDdD_bin*1.0e-6*1e3,width=deltad*1.0e6,edgecolor='k',linewidth=0.5)
            ax[-1].axvline(x=D_cen[0]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')
            ax[-1].axvline(x=D_cen[-1]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')
            ax[-1].text(0.1,0.9,'Given total Nr: '+str("{:.2e}".format(Nr*1.0e-6))+'cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.8,'sum total gamma Nr: '+str("{:.2e}".format(total_nl))+'cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.7,'sum total bin Nr: '+str("{:.2e}".format(total_nl_bin))+'cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.6,'RelDiff: '+str("{:.2f}".format(abs(Nr*1.0e-6-total_nl_bin)/(Nr*1.0e-6)*100.))+'%',fontsize=6,transform = ax[-1].transAxes)


            ax.append(fig.add_subplot(spec[1, 0]))
            # ax[-1].set_ylim(0,2.0e-8)
            ax[-1].set_yscale('linear')
            ax[-1].set_ylabel('Mass [g cm$^{-3}$]',fontsize=10)
            ax[-1].set_xscale('log')
            #ax[-1].set_xlim(1.0e-10,1.0e4)
            ax[-1].set_xlabel('Diameter [\u03bcm]', fontsize=10)
            ax[-1].tick_params(axis='y', rotation=0,labelsize=7)
            ax[-1].tick_params(axis='x', rotation=0,labelsize=7)
            ax[-1].plot(centerdd*1.0e6,dum_massDr*1.0e-3,alpha=0.8,lw=1,linestyle='-',c='k',)
            ax[-1].bar(D_cen*1.0e6,dum_MDdD_bin*1.0e-3,width=deltad*1.0e6,edgecolor='k',linewidth=0.5)
            ax[-1].axvline(x=D_cen[0]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')
            ax[-1].axvline(x=D_cen[-1]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')
            ax[-1].text(0.1,0.9,'Given total mr: '+str("{:.2e}".format(Qr*1.0e-3))+'g cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.8,'sum total gamma mr: '+str("{:.2e}".format(total_ml))+'g cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.7,'sum total bin mr: '+str("{:.2e}".format(total_ml_bin))+'g cm$^{-3}$',fontsize=6,transform = ax[-1].transAxes)
            ax[-1].text(0.1,0.6,'RelDiff: '+str("{:.2f}".format(abs(Qr*1.0e-3-total_ml_bin)/(Qr*1.0e-3)*100.))+'%',fontsize=6,transform = ax[-1].transAxes)

            plt.savefig('PSDr_comparison.png')
            plt.close()
        ####################################################################

    if (qc_in > qsmall) or (qr_in > qsmall):
        ### add cloud and rain PSD into a single combined PSD
        # each bin total number
        NDdD_bin[:] = (gamma_psdc_bin[:]+gamma_psdr_bin[:])*deltad[:] # unit: m-3
        # each bin total mass
        MDdD_bin[:] = NDdD_bin[:] * q_cen[:] * 1e-6                   # unit:kg m-3

        ####################################################################
        if make_plot_total:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            ### gamma PSD
            powlaw_coeff = np.linspace(start=-9,stop=-1,num=100,endpoint=True)
            dd           = 10.0**powlaw_coeff 

            deltadd = (dd - np.roll(dd,1))[1:]
            centerdd= ((dd + np.roll(dd,1))/2.0)[1:]

            gamma_psdc   = N0c*(centerdd[:]**muc)*np.exp(-lambdac*centerdd[:])      # unit: m-4-u
            dum_NDdDc    = gamma_psdc[:]*deltadd[:]                                 # unit: m-3
            dum_massDc   = dum_NDdDc[:]*(np.pi/6.0*centerdd[:]**3*rhow)             # unit: kg/m3. Assumption: spherical

            gamma_psdr   = N0r*(centerdd[:]**mur)*np.exp(-lambdar*centerdd[:])      # unit: m-4-u
            dum_NDdDr    = gamma_psdr[:]*deltadd[:]                                 # unit: m-3
            dum_massDr   = dum_NDdDr[:]*(np.pi/6.0*centerdd[:]**3*rhow)             # unit: kg/m3. Assumption: spherical

            # line number and mass (true value)
            v,lr_left = find_nearest(centerdd,D_cen[0])
            v,lr_righ = find_nearest(centerdd,D_cen[-1])
            
            dpi = 300
            fig = plt.figure(figsize=(1800/dpi,1200/dpi), dpi=dpi) ##,constrained_layout=True)
            spec= gridspec.GridSpec(ncols=2, nrows=2,figure=fig,left=0.1, bottom=0.1,right=0.95,top=0.95)
            spec.update(wspace=0.4,hspace=0.45)
            
            ax = []

            ax.append(fig.add_subplot(spec[0, 0]))
            # ax[-1].set_ylim(0,20)
            ax[-1].set_yscale('linear')
            ax[-1].set_ylabel('Number [cm$^{-3}$]',fontsize=10)
            ax[-1].set_xscale('log')
            #ax[-1].set_xlim(1.0e-10,1.0e4)
            ax[-1].set_xlabel('Diameter [\u03bcm]', fontsize=10)
            ax[-1].tick_params(axis='y', rotation=0,labelsize=7)
            ax[-1].tick_params(axis='x', rotation=0,labelsize=7)
            ax[-1].plot(centerdd*1.0e6,(dum_NDdDc+dum_NDdDr)*1.0e-6,alpha=0.8,lw=1,linestyle='-',c='k',)
            ax[-1].plot(D_cen*1e6, NDdD_bin*1.0e-6, lw=1, linestyle=':', c='r')
            ax[-1].axvline(x=D_cen[0]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')
            ax[-1].axvline(x=D_cen[-1]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')


            ax.append(fig.add_subplot(spec[1, 0]))
            # ax[-1].set_ylim(0,2.0e-7)
            ax[-1].set_yscale('linear')
            ax[-1].set_ylabel('Mass [g cm$^{-3}$]',fontsize=10)
            ax[-1].set_xscale('log')
            #ax[-1].set_xlim(1.0e-10,1.0e4)
            ax[-1].set_xlabel('Diameter [\u03bcm]', fontsize=10)
            ax[-1].tick_params(axis='y', rotation=0,labelsize=7)
            ax[-1].tick_params(axis='x', rotation=0,labelsize=7)
            ax[-1].plot(centerdd*1.0e6,(dum_massDc+dum_massDr)*1.0e-3,alpha=0.8,lw=1,linestyle='-',c='k',)
            ax[-1].plot(D_cen*1.0e6,MDdD_bin*1.0e-3, lw=1, linestyle=':', c='r',)
            ax[-1].axvline(x=D_cen[0]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')
            ax[-1].axvline(x=D_cen[-1]*1.0e6, ymin=-10, ymax=1.0e2,linestyle=':',lw=1,color='grey')

            plt.savefig('PSD_total_comparison.png')
            plt.close()
        ####################################################################
        
        ### determine cloud liquid and rain cutoff size
        ### for simplicity, use a fixed radius threshold of 40 um to differentiate raindrops from cloud droplets for now (reference to 40 um: Gettelman et al 2021 JAMES; Geoffroy et al., 2014 ACP; Azimi et al. 2024 JAMES: 50um radius)
        val,cutoff_idx = find_nearest(r_cen, 40)
        # print('cutoff: ', val, ' ', cutoff_idx)
        
        #####################################################################
        ### call SDM emulator function, pass in "initial" PSD, return new PSD after the collision-coalescence processes with a timte step = 100 sec
        dt = 100.0                                             # unit: sec
        new_MDdD_bin, new_NDdD_bin = py_SDM.compute_coll_SDM(numbin, dt, MDdD_bin[:], NDdD_bin[:])
        #####################################################################
    
        ### derive tendencies
        ### cloud liquid 
        cld_dsd_nbf  = np.sum(NDdD_bin[0:cutoff_idx+1])
        #cld_dsd_nbf  = np.sum(MDdD_bin[0:cutoff_idx+1]/(q_edg[0:cutoff_idx+1] * 1e-6))
        cld_dsd_mbf  = np.sum(MDdD_bin[0:cutoff_idx+1])
        
        cld_dsd_naf  = np.sum(new_NDdD_bin[0:cutoff_idx+1])
        #cld_dsd_naf  = np.sum(new_MDdD_bin[0:cutoff_idx+1]/(q_edg[0:cutoff_idx+1] * 1e-6))
        cld_dsd_maf  = np.sum(new_MDdD_bin[0:cutoff_idx+1])
        
        ### rain
        rain_dsd_nbf = np.sum(NDdD_bin[cutoff_idx+1:numbin])
        #rain_dsd_nbf = np.sum(MDdD_bin[cutoff_idx+1:numbin]/(q_edg[cutoff_idx+1:numbin] * 1e-6))
        rain_dsd_mbf = np.sum(MDdD_bin[cutoff_idx+1:numbin])
    
        rain_dsd_naf = np.sum(new_NDdD_bin[cutoff_idx+1:numbin])
        #rain_dsd_naf =  np.sum(new_MDdD_bin[cutoff_idx+1:numbin]/(q_edg[cutoff_idx+1:numbin] * 1e-6))
        rain_dsd_maf = np.sum(new_MDdD_bin[cutoff_idx+1:numbin])
    
        ### total liquid for mass conservation test
        nliqtotbf = np.sum(NDdD_bin[0:numbin])
        qliqtotbf = np.sum(MDdD_bin[0:numbin])
    
        nliqtotaft= np.sum(new_NDdD_bin[0:numbin])
        qliqtotaft= np.sum(new_MDdD_bin[0:numbin])

        if (((qliqtotaft - qliqtotbf)/qliqtotbf) < 1e-10):
            pass
            ### mass is conserved.
        else:
            print('Mass is not conserved. Check!')
            sys.exit()
            
        qc_tend_out = (cld_dsd_maf - cld_dsd_mbf)/dt   # unit: kg m-3 s-1
        nc_tend_out = (cld_dsd_naf - cld_dsd_nbf)/dt   # unit: m-3 s-1
        qr_tend_out = (rain_dsd_maf - rain_dsd_mbf)/dt # unit: kg m-3 s-1
        nr_tend_out = (rain_dsd_naf - rain_dsd_nbf)/dt # unit: m-3 s-1

        if (qc_in < qsmall) and (qr_in > qsmall):
            qc_tend_out = 0.0
            qr_tend_out = 0.0
            
            nr_tend_out += nc_tend_out
            nc_tend_out = 0.0
    
    ### no cloud, skip all
    else:
        qc_tend_out = 0.0
        nc_tend_out = 0.0
        qr_tend_out = 0.0
        nr_tend_out = 0.0
    
    return qc_tend_out, nc_tend_out, qr_tend_out, nr_tend_out

def find_nearest(array, value):
    import numpy as np
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
