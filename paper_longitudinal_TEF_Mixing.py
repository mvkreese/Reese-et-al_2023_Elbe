# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:20:43 2022

@author: Nina Reese / nina.reese@io-warnemuende.de

**Data processing / analysis for GETM model output**

Combined TEF and Mixing plots, as two single figures, for two different months
in 2012 or 2013.
- Averaging over 2 spring-neap cycles (29 days, 12:48:00h), respectively
- 1 figure showing bulk values along x-axis, cross-channel (y) integrated
    --> TEF Q_in, Q_out, Q_in+Q_out, Q_r
    --> TEF s_in, s_out, s_in-s_out
    --> Mixing M(x) vs. s_in*s_out*Q_r
    ==> Fig. 7 in [4]
- 1 figure showing variables in x-S-space, cross-channel (y) integrated
    --> TEF Q(x,S)
    --> TEF q(x,S)
    --> y-integrated local mixing m_x(x,S)
    --> y-integrated local mixing derivative (1/2)*d(m_x)/dS
    --> y-integrated vertical diahaline velocity, u_dia,z(x,S)
    ==> Fig. 8 in [4]

Computes longitudinal Total Exchange Flow (TEF) variability from GETM output,
see MacCready (2011)[1] and Lorenz et al. (2019)[2].
The script applies the pyTEF package by Boergel and Lorenz (2022)[3].


[1] MacCready, P., 2011: Calculating estuarine exchange flow using isohaline coordinates.
    Journal of Physical Oceanography, 41 (6), 1116-1124, doi:10.1175/2011JPO4517.1,
    url: https://journals.ametsoc.org/view/journals/phoc/41/6/2011jpo4517.1.xml.
[2] Lorenz, M., Klingbeil, K., MacCready, P., and Burchard, H. (2019)
    Numerical issues of the Total Exchange Flow (TEF) analysis framework for
    quantifying estuarine circulation, Ocean Sci., 15, 601-614,
    https://doi.org/10.5194/os-15-601-2019
[3] pyTEF package for the Total Exchange Flow (TEF) analysis framework.
    https://florianboergel.github.io/pyTEF/ , version: Apr 4, 2022.
[4] N. Reese, U. Graewe, K. Klingbeil, X. Li, M. Lorenz, H. Burchard, 2023:
    Local mixing determines spatial structure of diahaline exchange flow in a
    mesotidal estuary – a study of extreme runoff conditions.
    J. Phys. Oceanogr., submitted.


LAST UPDATED:
    NR, 2023-02-21, 10:25h --- debugging (wrong Q_r)
    NR, 2023-01-30, 11:33h --- added tributaries to runoff --> Q_r=Q_r(x)
    NR, 2023-01-18, 15:17h --- debugging; added dividing salinity
    NR, 2023-01-13, 13:50h --- removed pyTEF binning; now uses GETM online binning
    NR, 2023-01-09, 13:15h --- minor plot fixes
    NR, 2022-12-19, 15:22h --- minor plot fixes
    NR, 2022-12-05, 11:39h
    NR, 2022-12-02, 15:27h
    NR, 2022-11-17, 15:45h
    NR, 2022-10-13, 09:39h
"""

#==============================================================================


import numpy as np
import pandas as pd
import xarray
import datetime
from time import perf_counter
#from pyTEF.tef_core import *
#from pyTEF.calc import sort_1dim
from pyTEF.calc import calc_bulk_values

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import TwoSlopeNorm #DivergingNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({       #Use LaTeX for plotting
    "text.usetex": True})


#==============================================================================

# Manual Input

exp = 'exp_2022-08-12'

#timeframes for the plots - defines two averaging periods that will be
#compared in the plots: ([start1, stop1], [start2, stop2])
timeframes = ( ['2012-09-01 00:00:00', '2012-09-30 12:48:00'],
               ['2013-06-01 00:00:00', '2013-06-30 12:48:00'])

# path to nc file containing river runoff forcing of setup:
river_path = '/silod6/reese/tools/getm/setups/elbe_realistic/rivers.nc4'

savefig = True #Figure will only be saved if True


#==============================================================================

# Before loop:

print(' \n' + 'Sehr geehrte Damen und Herren!\n' + 
      'Herzlich willkommen bei der Deutschen Bahn!\n' + 
      'Starting the script...')

t0 = perf_counter() #This is only here for my personal entertainment
cc = 0 #Just a counter for each month

#define figures and formatting:
fig1, ax1 = plt.subplots(3, 2, figsize=(7,7.5), tight_layout = True, sharex=True,
                         sharey='row')
fig2, ax2 = plt.subplots(5, 2, figsize=(8,8.2), tight_layout = False, sharex=True,
                         sharey=True)
fig2.subplots_adjust(
    top=0.9,
    bottom=0.1,
    left=0.08,
    right=0.95,
    hspace=0.12,
    wspace=0.02
)

print('')


#==============================================================================

# Start time loop here

for tf in timeframes:
    
    start = tf[0]
    stop  = tf[1] #should usually cover 2 spring-neap cycles
    
    print('\n====================================================')
    print('Considering time span from ' + start + 
          ' to ' + stop + '\n')

    startdate = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    date = startdate.strftime('%Y%m') + '01'
    month = startdate.strftime('%B')
    year = startdate.strftime('%Y')
    
    #Path to mean GETM output for TEF analysis and for model-km:
    tef_base = '../store/' + exp + '/OUT_182/' + date + '/'  
    tef_file_name = 'Elbe_TEF_mean_all.' + date + '.nc4'

    #Path to GETM output for mixing analysis:
    mixing_base = '../store/' + exp + '/OUT_182/' + date + '/' 
    mixing_file_name = 'Mixing_Mean_all.' + date + '.nc4'

    #Path to direct diahaline GETM output:
    dia_inst_base = '../store/' + exp + '/OUT_182/' + date + '/'  
    dia_inst_file_name = 'Elbe_dia_getm_all.' + date + '.nc4'
    
    
    #%%
    #==========================================================================
    #  LOAD Model-km and grid cell area
    #==========================================================================
    #Find distance along the thalweg cell-center line j=158 in km
    #with upstream end at 0km (Geesthacht) (i.e., compute model-km)
    #This is needed for plotting
    
    print('\t Computing model km...')
    
    path = tef_base + tef_file_name #full path to file
    dx_tot = xarray.open_mfdataset(path)['dxc'][:]
    dx_tot = np.asarray(dx_tot)
    dx = dx_tot[157,:] #just need dx along thalweg
    dx[454:509] = dx_tot[160,454:509] #fill Hamburg area with Norderelbe distances
    dx[452] = 351.272
    dx[453] = 356.421
    dx[509] = 256.211
    dx[510] = 219.797
    
    #distance along thalweg j=158 in km, with upstream end at 0km:
    x = np.cumsum(dx[::-1])[::-1]/1000
    
    # Stuff we will need for the mixing analysis later on:
    print('\t Loading grid cell area...')
    dA = xarray.open_mfdataset(path)['areaC']
    dA = np.asarray(dA)
    # Correctional settings for mixing computation:
    dA[np.isnan(dA)] = 0
    dx_tot[np.isnan(dx_tot)] = 0
    dA[dx_tot == 0] = 0
    dx_tot[dx_tot == 0] = 1
    
    
    #%%
    #==========================================================================
    #  LOAD DATA
    #==========================================================================
    #import getm output / load TEF stuff
    
    print('\t Loading TEF data...')
    path = tef_base + tef_file_name #full path to file
    
    # loading salt-binned variables:
    variables = [
        'salt_s',   #interface (!) salinity; g/kg
        'uu_s_mean',     #volume transport in local x-direction; m2/s
        'dyu',      #y-distance between u-points; m
        'Sfluxu_s_mean'  #salt flux in local x direction; (g/kg)*m3/s
    ]
    
    ds_Elbe = xarray.open_dataset(path) #load full data set for given month
    ds_Elbe = ds_Elbe[variables].load() #load variables
    
    print('\t Extracting data for requested time span...')
    #extract only data within start + stop date
    ds_Elbe = ds_Elbe.sel(time=slice(start, stop))

    dyu = ds_Elbe['dyu']
    salt_s = ds_Elbe['salt_s'] #salinity at interfaces
    uu_s_mean = ds_Elbe['uu_s_mean']
    Sfluxu_s_mean = ds_Elbe['Sfluxu_s_mean']
    
    delta_s = salt_s[1] - salt_s[0]
    salt_Q = salt_s                  #salinities where Q is defined (interfaces)
    salt_q = salt_s[:-1] + delta_s/2 #salinities where q is defined (centers)
    
    print('')
    print(ds_Elbe)
    print('')
    
    xlen = len(ds_Elbe['xic'])
    
    
    #%%
    #==========================================================================
    #  RIVER RUNOFF
    #==========================================================================
    #import observed river runoff data (which was also used in the model):
        
    print('\t Loading river runoff...')
    Qr_time = xarray.open_mfdataset(river_path)['time'][:] #loads full month
    Qr_time = Qr_time.loc[start:stop] #chooses data from start to stop
    
    # Read riverinfo to get names and x-indices of all tributaries
    rivers = pd.read_table('../riverinfo.dat', skiprows=1, comment='!',
                delimiter=' ', header=None, usecols=[1,3], names=['etac', 'name'])
    # remove duplicates (a given runoff might have multiple entrance points)            
    rivers = rivers.drop_duplicates()
    
    # Define runoff as function of along-channel position to include tributaries
    Q_r = np.zeros((len(x)))
    Q_r_x = 0 #runoff at a single position x along the channel
    
    for ii in np.arange(xlen-1,-1,-1):
    	# check when to load new runoff
    	# add new runoff to Q_r_x
    	if any(rivers['etac'].isin([ii+1])): #ii+1 because GETM indexing starts at 1, python at 0
    		r_name = rivers[rivers['etac'].isin([ii+1])]['name'].values[0]
    		Q_r_ri = xarray.open_mfdataset(river_path)[r_name][:] #loading runoff
    		Q_r_ri = Q_r_ri.loc[start:stop] #selecting time span
    		Q_r_ri = np.nanmean(Q_r_ri) #temporal averaging
    		Q_r_x += Q_r_ri
    
    	# Fill Q_r_x with respective sum of runoffs at a given position x
    	# A given tributary will be included at all positions x downstream of where it enters the main river
    	Q_r[ii] = Q_r_x
    
    Q_r[-1] = Q_r[-2] #last value would otherwise remain 0 / not be filled
    print('\t Average runoff at Neu Darchau: ' + str(Q_r[-1]) + ' m3/s')
    # Sepember 2012: avrg runoff at ~300 m3/s Neu Darchau, 350 m3/s Cuxhaven
    # June 2013: avrg runoff at 2500 m3/s Neu Darchau
    
    
    #%%
    #==========================================================================
    # TEF ANALYSIS: Q(S), q(s), Q_in, s_in, Q_out, s_out
    #==========================================================================
    #Compute TEF stuff from simulated data 
    
    print('\t Preparing TEF computations...')
    
    # Volume transport Q(S) for salinity classes s>=S
    Q = np.sum( np.cumsum(uu_s_mean[:,::-1,:,:]*dyu, axis=1)[:,::-1,:,:],
                axis=2 ) #cumulative sum in s, integration in y
    Q = np.nanmean( Q, axis=0 ) #temporal averaging
    
    # Salt transport Q_s(S) in salinity classes s>=S
    Q_s = np.sum( np.cumsum(Sfluxu_s_mean[:,::-1,:,:], axis=1)[:,::-1,:,:],
                  axis=2 ) #cumulative sum in s, integration in y, Q(S) is inflow for s>=S
    Q_s = np.nanmean( Q_s, axis=0 ) #temporal averaging
    
    # Volume transport q(S) per salinity class S
    # (removing 1st S-value of q (which should be 0), because q has dim S-1)
    q = np.sum( uu_s_mean*dyu / delta_s, axis=2)[:,1:,:]
    q = np.nanmean( q, axis=0 ) #temporal averaging
    
    
    #Initialise
    print('\t\t Initialising arrays...')
    Q_in = np.zeros((xlen)) #volume inflow
    Q_out = np.zeros((xlen)) #volume outflow
    Qs_in = np.zeros((xlen)) #salt inflow
    Qs_out = np.zeros((xlen)) #salt outflow
    s_in = np.zeros((xlen)) #inflow salinity
    s_out = np.zeros((xlen)) #outflow salinity
    s_div = np.zeros((xlen)) #dividing salinity between inflow and outflow layer
    
    
    print('\t\t Beginning longitudinal analysis; choosing transect lines...' + '\n')
    
    #loop through along-channel (x) indices (i.e., through transects)
    for ii in range(xlen):
        print('----------')
        print('Round #' + str(ii+1).zfill(3) + ' of ' + str(xlen))
        
        Q_transect = Q[:,ii]
        Q_transect_s = Q_s[:,ii]
        q_transect = q[:,ii]
        
        thresh = 0.01*abs(np.nanmax(Q_transect)) #default: 0.01
        
        # Moving on to bulk values
        print('\t Computing bulk values...')
        bulk_vol_s = calc_bulk_values(salt_Q, Q_transect, Q_thresh=thresh)
        bulk_salt_s = calc_bulk_values(salt_Q, Q_transect_s, index = bulk_vol_s.index)
        
        #Fill arrays
        if len(bulk_vol_s.Qin.values) == 1:
            Q_in[ii] = bulk_vol_s.Qin.values #volume inflow
        elif len(bulk_vol_s.Qin.values) > 1:
            Q_in[ii] = np.nansum(bulk_vol_s.Qin.values)
            print(Q_in[ii])
        else:
            Q_in[ii] = np.nan
            
        if len(bulk_salt_s.Qin.values) == 1:
            Qs_in[ii] = bulk_salt_s.Qin.values
        elif len(bulk_salt_s.Qin.values) > 1:
            Qs_in[ii] = np.nansum(bulk_salt_s.Qin.values)
        else:
            Qs_in[ii] = np.nan  
            
        if len(bulk_vol_s.Qout.values) == 1:
            Q_out[ii] = bulk_vol_s.Qout.values #volume outflow
        elif len(bulk_vol_s.Qout.values) > 1:
            Q_out[ii] = np.nansum(bulk_vol_s.Qout.values)
            print(Q_out[ii])
        else:
            Q_out[ii] = np.nan
            
        if len(bulk_salt_s.Qout.values) == 1:
            Qs_out[ii] = bulk_salt_s.Qout.values
        elif len(bulk_salt_s.Qout.values) > 1:
            Qs_out[ii] = np.nansum(bulk_salt_s.Qout.values)
        else:
            Qs_out[ii] = np.nan
            
        s_in[ii]  = Qs_in[ii] / Q_in[ii]
        s_out[ii] = Qs_out[ii] / Q_out[ii]
        s_div[ii] = np.nanmean(bulk_vol_s.divval) #mean in case we have >2 layers
    
    
    
    #%%
    #==========================================================================
    #==========================================================================
    # PLOTTING
    #==========================================================================
    #==========================================================================
    
    print('\n' + '\t Beginning the plots...')
    
    #%%
    #==========================================================================
    # (a) 2D Longitudinal Knudsen and TEF
    #========================================================================== 
    
    #First: Plot of Q_in, Q_out, and Q_in+Q_out=Q_r
    #   (in case of volume conserv.; should be constant along x!!!)
    
    print('\t\t Q_in, Q_out, Q_r...')
    
    ax = ax1[0,cc]
    
    ax.hlines(0, x[0], x[-1], colors='k', linestyles=':', linewidth=1, alpha=0.5)
    ax.plot(x, Q_in, color='k', alpha=1, label='$Q_{\mathrm{in}}$')
    ax.plot(x, Q_out, color='grey', alpha=1, label='$Q_{\mathrm{out}}$')
    ax.plot(x, Q_in+Q_out, color='gainsboro', alpha=1,
            label='$Q_{\mathrm{in}} + Q_{\mathrm{out}}$')
    ax.plot(x, -Q_r, color='orange', linestyle='-',
              label='$-Q_{\mathrm{r}}$')
    ax.set_ylim([-4600, 2100])
    ax.set_title(month + ' ' + year)
    
    if cc==0:
        ax.set_ylabel('$Q$ (m$^3$/s)', fontsize=12)
        ax.legend(loc=3, ncol=2)
        ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelleft=True, labeltop=True)
    elif cc==1:
    	ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelright=True, labeltop=True)

    
    
    #%% Next: Plot of s_in, s_out, delta_s
    
    print('\t\t s_in, s_out...')
    
    ax = ax1[1,cc]
    
    ax.plot(x, s_in, color='k', alpha=1, label='$S_{\mathrm{in}}$')
    ax.plot(x, s_out, color='grey', alpha=1, label='$S_{\mathrm{out}}$')
    ax.plot(x, s_in-s_out, color='gainsboro', alpha=1, label='$\Delta S$')
    ax.set_ylim([-1, 25])
    
    if cc==0:
        ax.set_ylabel('$S$ (g/kg)', fontsize=12)
        ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelleft=True)
    elif cc==1:
        ax.legend(loc=5)
        ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                       left=True, right=True, labelright=True)
    
    
    
    #%%
    #==========================================================================
    #  (b) 3D Longitudinal TEF and Mixing
    #==========================================================================
    
    # First: Q(x,S)
    print('\t\t Q(x,S)...')
    
    # Coordinates for plotting:
    S, X = np.meshgrid(salt_Q, x)
    X = X.T
    S = S.T
    
    #For plotting, set transport in empty salinity classes from 0 to NaN
    Q[np.where(Q==0)] = np.nan
    
    ax = ax2[0,cc]
    
    cax = ax.contourf(X, S, Q, levels=np.arange(-1200, 1700, 100),
                      cmap='twilight_shifted', norm=TwoSlopeNorm(0), extend='both')
    CS = ax.contour(X, S, Q, levels=np.arange(-1200, 1700, 400), colors='k')
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([30,0])
    ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True)
    
    if cc==0:
	    ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelleft=True, labeltop=True)
    
    elif cc==1:
        ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelright=True, labeltop=True)
        cbaxes = inset_axes(ax, width="80%", height="5%", loc=4)
        cbar = fig2.colorbar(cax, cax=cbaxes, orientation='horizontal',
		                 spacing='proportional', extend='both')
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.set_label('$Q(S)$ (m$^3$s$^{-1}$)')
    
    #ax.clabel(CS, inline=True, fontsize=6, fmt='%1.0f')
    ax.set_title(month + ' ' + year)
    
    
    #%% Next: q(x,S)
    
    print('\t\t q(x,S)...')
    
    # Coordinates for plotting:
    S, X = np.meshgrid(salt_q, x)
    X = X.T
    S = S.T
    
    #For plotting, set transport in empty salinity classes from 0 to NaN
    q[np.where(q==0)] = np.nan
    
    ax = ax2[1,cc]
    
    cax = ax.contourf(X, S, q, levels=np.linspace(-600,600,31),
                       cmap='twilight_shifted', norm=TwoSlopeNorm(0), extend='both') #'PuOr_r'
    ax.plot(x, s_in, color='darkred', linewidth=1.5, alpha=1, label='$S_{\mathrm{in}}$')
    ax.plot(x, s_out, color='w', linewidth=1.5, alpha=1, label='$S_{\mathrm{out}}$')
    ax.plot(x, (s_in+s_out)/2, color='k', linewidth=1, alpha=1,
            label='$(S_{\mathrm{in}}+S_{\mathrm{out}})/2$')
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([30,0])
    
    if cc==0:
        ax.legend(facecolor='grey', loc=4, fontsize=9)
        ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelleft=True)
	    
    elif cc==1:
        cbaxes = inset_axes(ax, width="80%", height="5%", loc=4)
        cbar = fig2.colorbar(cax, cax=cbaxes, orientation='horizontal',
		                 spacing='proportional', extend='both')
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.set_label('$q(S)$ (m$^3$s$^{-1}$(g/kg)$^{-1}$)')
        ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelright=True)
    
    
    
    #%% Do some clean-up before moving on to mixing...
    
    print('\t\t Clearing memory...')
    ds_Elbe = None; del ds_Elbe
    Q = None; del Q
    q = None; del q
    
    
    
    #%% Mixing along the channel m(x,S)
    # =========================================================================
    # LOAD Mixing data
    # =========================================================================
    
    print('\n\t Starting mixing analysis...')
    
    print('\t\t Loading ' + mixing_file_name + ' data...')
    
    path = mixing_base + mixing_file_name #full path to file
    
    salt_s = xarray.open_mfdataset(path)['salt_s'][:]  #salinity classes
    hpmS_s_mean = xarray.open_mfdataset(path)['hpmS_s_mean'] #physical mixing
    hpmS_s_mean = hpmS_s_mean.loc[start:stop]
    hnmS_s_mean = xarray.open_mfdataset(path)['hnmS_s_mean'] #numerical mixing
    hnmS_s_mean = hnmS_s_mean.loc[start:stop]
    
    print('\t\t Temporal averaging...')
    hpmS_s_mean = np.nanmean(hpmS_s_mean[:,:,:,:],axis=0) #Physical Mixing
    hnmS_s_mean = np.nanmean(hnmS_s_mean[:,:,:,:],axis=0) #Numerical Mixing
    
    print('\t\t Converting arrays...')
    hpmS_s_mean = np.asarray(hpmS_s_mean)
    hnmS_s_mean = np.asarray(hnmS_s_mean)
    salt_s = np.asarray(salt_s)
    delta_s = salt_s[1]-salt_s[0] #salinity class step size
    
    
    #%%
    # =========================================================================
    # Integration only along Y; Division by ds for variables per salinity class
    # =========================================================================
    
    # This for the longitudinal mixing distribution analysis
    
    print('\t\t Integration along Y...')
    #This is only integrated in y (etac), not in x:
    mms_phy_x = np.sum(hpmS_s_mean*dA/dx_tot,axis=1)/delta_s
    mms_num_x = np.sum(hnmS_s_mean*dA/dx_tot,axis=1)/delta_s
    
    mms_total_x = mms_phy_x+mms_num_x
    mask = np.ones(np.shape(mms_total_x))
    mask[mms_total_x==0] = np.nan
    
    
    print('\t Plotting mixing...')
    
    #coordinates for plotting:
    S, X = np.meshgrid(salt_s, x)
    X = X.T
    S = S.T  
    
    ax = ax2[2,cc]
    levels=np.linspace(0.0,3.2,num=60)
    
    cax = ax.contourf(X, S, mms_total_x*mask, cmap='magma_r', levels=levels,
                              extend='both')
    ax.plot(x, (s_in+s_out)/2, color='k', linewidth=1.5,
             label='$(S_{\mathrm{in}}+S_{\mathrm{out}})/2$')
    
    #finish plot
    if cc==0:
        ax.legend(facecolor='grey', loc=4, fontsize=9)
        ax.set_ylabel('$S$ (g/kg)')
        ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelleft=True)
    
    elif cc==1:
	    cticks = np.linspace(0, 3, 7)
	    cbaxes = inset_axes(ax, width="80%", height="5%", loc=4)
	    cbar = fig2.colorbar(cax, cax=cbaxes, orientation='horizontal',
		                 extend='both', ticks=cticks)
	    cbar.ax.tick_params(labelsize=8)
	    cbar.ax.xaxis.set_label_position('top')
	    cbar.ax.xaxis.set_ticks_position('top')
	    cbar.set_label('$m_{\mathrm{x}}(x,S)$ (m$^2$(g/kg)s$^{-1}$)')
	    ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelright=True)
    
                           
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([30,0])
    
    
    #%%
    # =========================================================================
    # Y-integrated S-derivative of m(x,y,S)
    # =========================================================================
    
    # This for the x-S-distribution of 1/2 dS(m(x,S))
    # (to compare to y-integrated udia,z)
    
    print('\t Computing Y-integrated S-derivative of m_xy...')
    
    #left-sided derivative:
    # dsm = ( mms_total_x[1:,:] - mms_total_x[:-1,:] ) / (2*delta_s) #dim(salt_s-1, x)
    
    #symmetric derivative:
    dsm = ( (mms_total_x*mask)[2:,:] - (mms_total_x*mask)[:-2,:] ) / (4*delta_s) #dim(salt_s-2, x)
    
    print('\t Plotting...')
    #coordinates for plotting:
    salt_s_dsm = salt_s[1:-1] #salinities where S-derivative is located
    S, X = np.meshgrid(salt_s_dsm, x)
    X = X.T
    S = S.T
    
    ax = ax2[3,cc]
    levels=np.linspace(-0.2,0.2,num=41)
    
    cax = ax.contourf(X, S, -dsm, levels=levels,
                       cmap='twilight_shifted', norm=TwoSlopeNorm(0), extend='both')
    ax.plot(x, (s_in+s_out)/2, color='k', linewidth=1.5,
            label='$(S_{\mathrm{in}}+S_{\mathrm{out}})/2$')
    
    #finish plot
    if cc==0:
    	ax.legend(facecolor='grey', loc=4, fontsize=9)
    	ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelleft=True)
    
    elif cc==1:
	    cbaxes = inset_axes(ax, width="80%", height="5%", loc=4)
	    cbar = fig2.colorbar(cax, cax=cbaxes, orientation='horizontal',
		                 extend='both')
	    cbar.ax.tick_params(labelsize=8)
	    cbar.ax.xaxis.set_label_position('top')
	    cbar.ax.xaxis.set_ticks_position('top')
	    cbar.set_label('$-1/2 \partial_S m_{\mathrm{x}}(x,S)$ (m$^2$s$^{-1}$)')
	    ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelright=True)
    

    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([30,0])
    
    print('\t Clearing memory...')
    dsm = None; del dsm
    
    
    
    
    
    #%%
    #==========================================================================
    # Next: M(x) for fig1
    #==========================================================================
    
    print('\t Computing M(x)...')
    # Compute total mixing as function of x by integrating m(x,S) over all
    # salinity classes
    
    print('\t\t Integration along S...')
    hpmS_s_mean[np.isnan(hpmS_s_mean)] = 0
    hnmS_s_mean[np.isnan(hnmS_s_mean)] = 0
    MMs_phy_x = np.sum(hpmS_s_mean, axis=0)
    MMs_num_x = np.sum(hnmS_s_mean, axis=0)
    
    print('\t\t Clearing memory...')
    hpmS_s_mean = None; del hpmS_s_mean
    hnmS_s_mean = None; del hnmS_s_mean
    
    print('\t\t Integration along y...')
    MMs_phy_x = np.sum(MMs_phy_x*dA, axis=0)
    MMs_num_x = np.sum(MMs_num_x*dA, axis=0)
    
    # Integrating over x to get mixing inside estuarine volume bounded by transect
    # at location x
    print('\t\t Integration along x...')
    MMs_phy_x = np.cumsum(MMs_phy_x[::-1], axis=0)[::-1]
    MMs_num_x = np.cumsum(MMs_num_x[::-1], axis=0)[::-1]
    MMs_total_x = MMs_num_x + MMs_phy_x
    
    
    print('\t Plotting M(x)...')
    ax = ax1[2,cc]
    
    ax.hlines(0, x[0], x[-1], colors='k', linestyles=':', linewidth=1, alpha=0.5)
    ax.plot(x, s_in*s_out*Q_r, color='orange', linestyle='--', alpha=1,
            label='$S_{\mathrm{in}}S_{\mathrm{out}}Q_{\mathrm{r}}$')
    ax.plot(x, MMs_total_x, color='k', alpha=1, label='$M_{\mathrm{T}}(x)$')
    ax.plot(x, MMs_phy_x, color='grey', alpha=1, label='$M_{\mathrm{T,phy}}(x)$')
    ax.plot(x, MMs_num_x, color='gainsboro', alpha=1, label='$M_{\mathrm{T,num}}(x)$')
    ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True)
    ax.set_ylim([-0.2*10**5, 4.1*10**5])
    
    if cc==0:
        ax.set_ylabel('$M_{\mathrm{T}}(x)$ ((m$^2$/s)(g/kg)$^3$)', fontsize=12)
        ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelleft=True)
    elif cc==1:
        ax.legend(loc=7, ncol=1)
        ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                       left=True, right=True, labelright=True)
    
    
    print('\t Clearing memory...')
    mms_phy_x = None; del mms_phy_x
    mms_num_x = None; del mms_num_x
    mms_total_x = None; del mms_total_x
    
    
    #%%
    #==========================================================================
    # Last: u_dia(x,y,S)dy for fig2
    #==========================================================================
    
    # Plot the y-integral of u_dia for longitudinal analysis
    
    print('\t Loading ' + dia_inst_file_name + ' data...')
    path = dia_inst_base + dia_inst_file_name #full path to file
    w_dia_getm = (xarray.open_mfdataset(path)['wdia_s_mean']).loc[start:stop] #[time, salt_s, etac, xic]
    
    print('\t\t Temporal averaging...')
    w_dia_getm = np.nanmean(w_dia_getm[:,:,:,:],axis=0)
    
    print('\t\t Integration in y...')
    w_dia_getm[np.isnan(w_dia_getm)] = 0
    w_dia_getm = np.sum(w_dia_getm[:,:,:]*dA/dx_tot, axis=1)
    
    S, X = np.meshgrid(salt_s, x)
    X = X.T
    S = S.T
    
    
    print('\t Plotting u_dia,z...')
    ax = ax2[4,cc]
    
    levels=np.linspace(-0.2,0.2,num=41)
    
    cax = ax.contourf(X, S, -w_dia_getm*mask, levels=levels,
                       cmap='twilight_shifted', norm=TwoSlopeNorm(0), extend='both')
    ax.plot(x, (s_in+s_out)/2, color='k', linewidth=1.5,
            label='$(S_{\mathrm{in}}+S_{\mathrm{out}})/2$')
    
    #finish plot
    if cc==0:
    	ax.legend(facecolor='grey', loc=4, fontsize=9)
    	ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelleft=True)
    
    elif cc==1:
	    cbaxes = inset_axes(ax, width="80%", height="5%", loc=4)
	    cbar = fig2.colorbar(cax, cax=cbaxes, orientation='horizontal',
		                 extend='both')
	    cbar.ax.tick_params(labelsize=8)
	    cbar.ax.xaxis.set_label_position('top')
	    cbar.ax.xaxis.set_ticks_position('top')
	    cbar.set_label('$-\int u_{\mathrm{dia,z}}^{\mathrm{S}}$d$y$ (m$^2$s$^{-1}$)')
	    ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True, labelright=True)
                           
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([30,0])
    ax.set_yticks([0, 10, 20, 30])
    ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True)
    
    print('\t Clearing memory...')
    w_dia_getm = None; del w_dia_getm
    
    
    
    print('')
    cc += 1


#%%
#==============================================================================
# AFTER TIME LOOP
#==============================================================================

print('\n' + 'Finishing plots...')

# add letters to each subfigure
letters = ['(a)', '(b)', '(c)', '(d)', '(e)',
           '(f)', '(g)', '(h)', '(i)', '(j)']
positions = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]

cc = 0
for ax in ax1.flatten():
    anchored_text = AnchoredText( letters[cc], loc=1,
                                  frameon=False, prop=dict(color='k', zorder=1000),
                                  pad=0.02)
    ax.add_artist(anchored_text)
    ax.set_xlim([ x[78], x[254] ])
    cc +=1    
cc = 0
for ax in ax2.flatten():
    anchored_text = AnchoredText( letters[cc], loc=positions[cc],
                                  frameon=False, prop=dict(color='k', zorder=1000),
                                  pad=0.02)
    ax.add_artist(anchored_text)
    ax.set_xlim([ x[78], x[254] ])
    ax.grid(True, zorder=-1)
    cc +=1


#Common x-axis for each figure:
    
for fig in [fig1, fig2]:
    # Add an invisible axis for shared x- and y-labels:
    axlbl = fig.add_subplot(frameon=False)
    # Hide ticks and tick label of the big axes
    axlbl.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    axlbl.grid(False)
    axlbl.set_xlabel('$x$ (Elbe model-km)')

# # Add an invisible axis for shared x- and y-labels:
# axlbl1 = fig1.add_subplot(111, frameon=False)
# # Hide ticks and tick label of the big axes
# axlbl1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# axlbl1.grid(False)
# axlbl1.set_xlabel('$x$ (Elbe model-km)')

# # Add an invisible axis for shared x- and y-labels:
# axlbl2 = fig2.add_subplot(frameon=False)
# # Hide ticks and tick label of the big axes
# axlbl2.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
# axlbl2.grid(False)
# axlbl2.set_xlabel('$x$ (Elbe model-km)')


# Smaller fontsize in legends:
plt.rc('legend',fontsize=6)


# Saving figure:
if savefig:
    fig1.savefig('plots/paper' + '/Knudsen_TEF' + '.pdf')
    fig1.savefig('plots/paper' + '/Knudsen_TEF' + '.png', dpi=300)
    fig2.savefig('plots/paper' + '/Long_TEF_mixing' + '.png', dpi=600)
plt.show()

#Yes, the only purpose the following line serves is for my stupid
# Deutsche Bahn joke. You're welcome.
t1 = perf_counter() - t0

print('Done!\n' + 
      'We have reached our destination with a delay of {t} minutes.'.format(t = int(np.round(t1/60))) )
print('Thank you for travelling with Deutsche Bahn!')

