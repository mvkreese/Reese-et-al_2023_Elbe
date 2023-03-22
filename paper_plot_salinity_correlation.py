# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:26:59 2022

@author: Nina Reese / nina.reese@io-warnemuende.de

**Data processing / analysis for GETM model output**

This script loads model and observational salinity data for given stations
in the Elbe estuary (tidal Elbe setup @Nina Reese).
	- interpolates model data from depth-varying coordinates
	  to a custom depth above ground
	- applies low-pass filter to data to remove M2 tidal variability
	  (cutoff frequency can be set manually)
	- plots & computes correlation of observed and simulated salinity
      (both instantaneous and low-pass filtered)
      
==> Fig. 5 in [1]

[1] N. Reese,U. Graewe, K. Klingbeil, X. Li, M. Lorenz, H. Burchard, 2023:
    Local mixing determines spatial structure of diahaline exchange flow in a
    mesotidal estuary – a study of extreme runoff conditions.
    J. Phys. Oceanogr., submitted.

HISTORY:
2023-03-16 NR: Script clean-up; added some documentation
2022-03 NR: Added depth interpolation to the old version (plot_salinity.py)
	 and renamed to plot_salinity_depth-interp.py
2022-03-14 NR: Improved functionality, added commentary.

"""


#==============================================================================

import numpy as np
import os
import xarray
from datetime import datetime
from datetime import timedelta
from scipy import signal
from scipy.interpolate import griddata
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.offsetbox import AnchoredText


#==============================================================================
# MANUAL INPUT
#==============================================================================

'''
- start: Date at which to start plotting
- stop: Date at which to stop plotting
- stations: names of the stations to load, as used in the file names (!)
- exp: name of experiment/directory from which the model data should be used

All stations available:
stations = ['Steinriff', 'Cuxhaven', 'Neufeldreede', 'Krummendeich']
'''

# Set analysis interval
start = '2013-01-01 00:00:00'
stop = '2013-12-31 23:55:00'

# Choose stations for analysis
stations = ['Steinriff', 'Cuxhaven', 'Neufeldreede',
            'Krummendeich']
# Labels by which to refer to the stations in the plot:
stations_labels = ['(a) LZ4a Steinriff', '(b) Cuxhaven AL',
                   '(c) LZ2a Neufeldreede', '(d) LZ1b Krummendeich']

# Choose model data to use
exp = 'exp_2022-08-12' #experiment handle
conf = '182' #configuration (for parallelisation; needed to find directory path)

# Choose which functions to use:
low_pass = True #Also compute lowpass-filtered data (i.e., semidiurnal tides removed)
plot_full = True #Plot results, i.e., salinity
save_fig = True #Save plot. Only applies for plot_results=True.
log = False #If True, a log file will be created in the output directory

# Low-pass filter settings
filter_cutoff = 30 #cutoff period for the low-pass filter (h)

# Set interpolation depth
int_d = 2.5 #depth above ground to which model data will be interpolated. DATA: 2.5m

# Info about observation data files
miss_val = '' #Value that marks missing data in the observation files
skip_header = 19 #Header rows to skip in observation files
skip_footer = 3 #Footer rows to skip in observation files



#==============================================================================
#%%  FUNCTION DEFINITONS
#==============================================================================

def find_datedirs(start, stop):
    
    """
    Finds all monthly directories of the form YYYYmm01 that will have to
    be loaded to get all data from start to stop time.
    Also returns epoch variable, which is required for time conversion for
    observational data.
    
    INPUT:
        start: [scalar, dtype=str] str of the form YYYY-mm-dd HH:MM:SS
            defining the start of the time interval for which data should
            be loaded
        stop: [scalar, dtype=str] str of the form YYYY-mm-dd HH:MM:SS
            defining the end of the time interval for which data should
            be loaded
    OUTPUT:
        datedirs: [list, dtype=str] list of all directories of the form
            YYYYmm01 that are covered by the time interval start:stop
        epoch: [scalar, datetime obj] datetime of the first day of the
            month defined in "start" at 00:00:00 hrs
    """
    
    start_year = int(start[0:4])
    start_month = int(start[5:7])
    stop_year = int(stop[0:4])
    stop_month = int(stop[5:7])
    datedirs = []
    
    if start_year==stop_year:
        if start_month==stop_month:
            datedirs.append(start[0:4] + start[5:7] + '01')
        else:
            for ii in range(start_month, stop_month+1):
                datedirs.append(start[0:4] + str(ii).zfill(2) + '01')
    else:
        for jj in range(start_year, stop_year+1):
            if jj==start_year:
                for ii in range(start_month, 13):
                    datedirs.append(str(jj) + str(ii).zfill(2) + '01')
            elif jj==stop_year:
                for ii in range(1, stop_month+1):
                    datedirs.append(str(jj) + str(ii).zfill(2) + '01')
            else:
                for ii in range(1, 13):
                    datedirs.append(str(jj) + str(ii).zfill(2) + '01')
    
    epoch = datetime(start_year,start_month,1,0,0)
    
    return datedirs, epoch


#==============================================================================

def vert_interp(salt, hn, H, int_d=2.5):
    
    """
    Vertically interpolates GETM model output for salinity at a given
    single location (i.e., horizontal grid cell) onto a given single
    depth value.
    
    INPUT:
        salt: [np.ndarray, dtype=float] array of dimension [time, z]
            model output salinity at a given horizontal grid cell
            in time and vertical dimension z
        hn: [np.ndarray, dtype=float] array of dimension [time, z]
            vertical sigma layer thickness at a given horizontal grid cell
            in time and vertical dimension z, in m
        H: [scalar, dtype=float]
            water depth below z=0 at the given horizontal grid cell in m
        int_d: [scalar, dtype=float]
            desired interpolation depth above ground (i.e., above z=H)
            in m
    OUTPUT:
        S_interp: [np.ndarray, dtype=float] array of dimension [time]
            Salinity interpolated onto int_d at all time steps
    """
    
    print('Interpolating in vertical direction...')

    if H <= int_d: 
        raise Exception("Error: Station too shallow!") #exit w/ error message
    
    depth_interp = H-int_d #interpolation depth is 2.5m above ground
    
    S_interp = np.zeros((np.shape(salt)[0])) #initiate interpolated salt array
    
    #loop through time dimension
    for ii in range(np.shape(salt)[0]):
        
        #depth at center of each sigma layer
        depths = np.cumsum(hn[ii, ::-1]) - hn[ii, ::-1]/2
        
        if (max(depths)-depth_interp) <= 0:
            #if 2.5m above ground is dry/not flooded, set salt to nan
            S_interp = np.nan
        else:
            #vertical interpolation onto desired interpolation depth
            S_interp[ii] = griddata(depths, salt[ii,:], depth_interp,
                                    method='cubic')
        
    return S_interp


#==============================================================================
#%%  START EXECUTION
#==============================================================================

prntlst = ["Moin!", "\n", "===============================", "\n",
	"Data Info:", "start = " + start, "stop = " + stop,
	"Model data from experiment #" + exp,
	"Lowpass filter cutoff at T = " + str(filter_cutoff) + " hrs",
	"\n", "Stations considered: " + str(stations), "\n",
	"===============================", "\n"]


if log:
    f = open('plots/salinity/log_' + str(start[:10]) + '_' + str(stop[:10])
             + exp + '.txt', 'w')  
    for line in prntlst:
        print(line)
        f.write(line)
        f.write("\n")
    f.close()

else:
    for line in prntlst:
        print(line)



#==============================================================================
#%%  LOAD DATA, DO THE COMPUTATIONS
#==============================================================================

# A simple counter for loop through stations:
cc = 0

datedirs, epoch = find_datedirs(start, stop)

#==============================================================================

fig, axs = plt.subplots(2,2, figsize=(8,8), sharex=True, sharey=True)

for nn in range(len(stations)):
    print("Considering station " + stations[nn] + "...")

    #(a) Model data
        
    mod_time_lst = []
    mod_S_lst = []
    hn_lst = []
    H = 0.0
    
    for datedir in datedirs: #load monthly data
        path = ('/silod6/reese/tools/getm/setups/elbe_realistic/store/'
                + exp + '/OUT_' + conf + '/' + datedir + '/')
        directories = os.listdir(path)
        directories = [d for d in directories if ('SST' in d)] #remove gauges from list (similar station names)
        ii = np.where([(stations[nn] in d) for d in directories])[0] #should only be 1 directory per loop, but in case it isn't... we load 'em all
        directories = [directories[i] for i in ii]
        directories = [path + d for d in directories]
        mod_S = xarray.open_mfdataset(directories)['salt'][:,:,0,0]
        mod_time = xarray.open_mfdataset(directories)['time'][:]
        hn = xarray.open_mfdataset(directories)['hn'][:,:,0,0] #sigma layer thickness
        H = xarray.open_mfdataset(directories)['bathymetry'][:] #bathy depth at station location
        
        mod_time_lst.append(mod_time) #add monthly data to total list
        mod_S_lst.append(mod_S)
        hn_lst.append(hn)
    
    mod_time = xarray.concat(mod_time_lst, dim='time') #concatenate data into one xarray
    mod_S = xarray.concat(mod_S_lst, dim='time')
    hn = xarray.concat(hn_lst, dim='time')
    mod_time = np.asarray(mod_time.loc[start:stop]) #chop off parts that are not contained within [start,stop]
    mod_S = np.asarray(mod_S.loc[start:stop])
    hn = np.asarray(hn.loc[start:stop])
    H = np.asarray(H)[0]
            
    mod_S = vert_interp(mod_S, hn, H, int_d)
    
    if low_pass: #apply a low-pas filter to remove M2 tidal cycle
        b, a = signal.butter(3, 1/(3600*filter_cutoff)/(1/1800/2), 
                             btype='low', analog=False)
        mod_S_lp = signal.filtfilt(b, a, mod_S)



#==============================================================================    
    
    #(b) Measured data
    
    base = '/silod6/reese/tools/getm/setups/elbe_realistic/analysis/observations/salt/'
    modeldirs = os.listdir(base)
    ii = np.where([(stations[nn] in m) for m in modeldirs])[0][0]
    file_name = modeldirs[ii]
    obs_path = base + file_name
    
    f = open(obs_path, 'r') # 'r' = read
    obs_data = np.genfromtxt(f, skip_header=skip_header, skip_footer=skip_footer, delimiter='\t', dtype=str)
    f.close()
    
    mestart = datetime.strptime(start, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1) #shift from UTC to UTC+1
    mestart = mestart.strftime("%Y-%m-%d %H:%M:%S")
    mestop = datetime.strptime(stop, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1) #shift from UTC to UTC+1
    mestop = mestop.strftime("%Y-%m-%d %H:%M:%S") #"2013-12-11 08:25:00" #(last data point in 2013 for Brunsbuettel)
    start_ind = np.where([(mestart in m) for m in obs_data[:,0]])[0][0] #first index of the start date
    stop_ind = np.where([(mestop in m) for m in obs_data[:,0]])[0][0] + 1 #first index of the stop date
    obs_data = obs_data[start_ind:stop_ind,:]
    obs_data[obs_data[:,1]==miss_val,1] = '-999' #missing values
    obs_S = obs_data[:,1].astype(float)
    obs_t_str = obs_data[:,0]
    obs_t_str = [ (datetime.strptime(mt, '%Y-%m-%d %H:%M:%S')
                    - timedelta(hours=1)) for mt in obs_t_str ] #-1h due to shift from UTC+1 to UTC
    obs_time = obs_t_str
    
    #remove missing values (NaN) from data:
    kk = np.where(obs_S==-999)[0]
    obs_S = np.delete(obs_S, kk)
    obs_time = np.delete(obs_time, kk)
    
    # Convert UTC datetime to seconds since the Epoch:
    obs_time_cu = [ ((mt - epoch).total_seconds()) for mt in obs_time ]
    obs_time_cu = np.array(obs_time_cu)
    
    
    #remove temporal loops from measured gauge data:
    mt_loops = signal.argrelmin(obs_time_cu, order=1)
    obs_time = np.delete(obs_time, mt_loops)
    obs_S = np.delete(obs_S, mt_loops)
    
    #find some data gaps for plotting
    delta_t = obs_time_cu[1:] - obs_time_cu[:-1]
    idx = np.where(delta_t>305)[0]+1
    
    if low_pass: #apply a low-pass filter to remove M2 tidal cycle
        b, a = signal.butter(3, 1/(3600*filter_cutoff)/(1/300/2), 
                 btype='low', analog=False)
        obs_S_lp = signal.filtfilt(b, a, obs_S)
        obs_S_lp = np.insert(obs_S_lp, idx, np.nan)
    
    #insert nan values at idx to avoid connecting lines between blocks of missing data
    #(for plotting)
    obs_S = np.insert(obs_S, idx, np.nan)
    obs_time = np.insert(obs_time, idx, obs_time[idx])
        
        
        
        
#%% Interpolating observations onto model time stamps

    obs_time_sse = [ ((mt - epoch).total_seconds()) for mt in obs_time ]
    obs_time_sse_unique = list(dict.fromkeys(obs_time_sse))
    obs_time_sse = (np.array(obs_time_sse)).astype(float)
    #choose only first index where certain timestamp appears:
    idx = [np.where(obs_time_sse==mtsu)[0][0] for mtsu in obs_time_sse_unique]
    obs_time_sse = obs_time_sse[idx]
    obs_S_lp_sse = obs_S_lp[idx]
    obs_S_sse = obs_S[idx]
    
    mod_time_sse = [ np.timedelta64(mt - np.datetime64(epoch), 's') for mt in mod_time ]
    mod_time_sse = (np.array(mod_time_sse)).astype(float)

    mtsse = [mtsse for mtsse in mod_time_sse if (mtsse in obs_time_sse)]
    ids = [np.where(obs_time_sse==m)[0][0] for m in mtsse]
    ids_mod = [np.where(mod_time_sse==m)[0][0] for m in mtsse]
    obs_S_lp_interp = obs_S_lp[ids]
    obs_S_interp = obs_S[ids]   
    mod_S = mod_S[ids_mod]
    mod_S_lp = mod_S_lp[ids_mod]

#==============================================================================
#%%  PLOTTING: Salinity (correlation between model and observations)
#==============================================================================

    ax = axs.flatten()[nn]
    ax.scatter(obs_S_interp, mod_S, s=2, color='k', alpha=0.4, label='tidal')
    ax.scatter(obs_S_lp_interp, mod_S_lp, s=2, color='royalblue', #mediumblue
               alpha=1, label='low-pass')

    # Compute Pearson's Correlation
    corr, _ = pearsonr(obs_S_interp[~np.isnan(obs_S_interp)],
                       mod_S[~np.isnan(obs_S_interp)])
    print('Pearsons correlation: %.3f' % corr)
    print('R^2: %.3f' % corr**2)
    
    corr_lp, _ = pearsonr(obs_S_lp_interp[~np.isnan(obs_S_lp_interp)],
                          mod_S_lp[~np.isnan(obs_S_lp_interp)])
    print('Pearsons correlation of low-pass filtered data: %.3f' % corr_lp)
    print('R^2: %.3f' % corr_lp**2)

    ax.set_title(stations_labels[nn])
    ax.set_xlim([0, 32])
    ax.set_ylim([0, 32])
    ticks = np.arange(0,35,5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params('both', direction='in', bottom=True, top=True,
                       left=True, right=True)
    ax.set_aspect('equal')

    xpoints = ax.get_xlim()
    ypoints = ax.get_xlim()
    ax.plot(xpoints, ypoints, linestyle='-', color='grey', lw=0.5)
    
    text = ('$R^2$ (tidal): {corr2:.3f} \n$R^2$ (low-pass): {corr2lp:.3f}')
    at = AnchoredText(text.format(corr2=corr**2, corr2lp=corr_lp**2),
                      loc=4, frameon=True, pad=0.1, borderpad=1)
    at.patch.set_boxstyle('round')
    at.patch.set_facecolor('wheat')
    at.patch.set_alpha(0.5)
    ax.add_artist(at)

    
    print('-------------')
    cc += 1
    

#add legend to lower right axis
lgnd = axs[0,0].legend(loc=2, scatterpoints=1)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]

# add a big axes, hide frame
k = fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel('Observed $S$ (g/kg)', fontsize=12)
plt.ylabel('Simulated $S$ (g/kg)', fontsize=12)


if save_fig:
    fig.savefig('plots/paper/salinity_correlation.png', dpi=600)
    fig.savefig('plots/paper/salinity_correlation.pdf')
plt.show()

    
print("Done!")


