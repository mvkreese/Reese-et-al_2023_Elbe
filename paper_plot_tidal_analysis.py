#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 08:46:00 2022

@author: Nina Reese / nina.reese@io-warnemuende.de

**Data processing / analysis for GETM model output**

This script loads model and observational tidal elevations for given gauges
in the Elbe estuary (tidal Elbe setup presented in [1]).
	- computes tidal M2 and M4 amplitudes and M2 phases at given gauge stations
      using Pytides [2]
    - for model data, also computes tidal M2 and M4 amplitudes and M2 phases
      at intermediate locations between gauges to provide a continuous analysis
	- applies low-pass filter to time series data to remove (M2) tidal
      variability (cutoff frequency can be set manually)
	- plots the resulting amplitudes and phases (at gauges and continuous),
      as well as the elevation time series at each gauge (both instantaneous
      and low-pass filtered)
==> Fig. 4 in [1]
      

[1] N. Reese,U. Graewe, K. Klingbeil, X. Li, M. Lorenz, H. Burchard, 2023:
    Local mixing determines spatial structure of diahaline exchange flow in a
    mesotidal estuary – a study of extreme runoff conditions.
    J. Phys. Oceanogr., submitted.

[2] Pytides routine for tidal prediction and analysis by Sam Cox
    (sam.cox@cantab.net).
    Available on GitHub: https://github.com/sam-cox/pytides


HISTORY:
2023-03-10 NR: Minor fixes
2023-02-21 NR: Just some updating
2022-08-18 NR: Fixing minor issues, replacing interpolated intermediate
    locations for tidal analysis with hand-picked grid cells. This ensures
    that each location is well inside the navigational channel and not on land.
2022-08-17 NR: Replacing station-wise tidal analysis of model results with a
    more continuous analysis (i.e., including points inbetween stations).
    Fixing minor issues
2022-08-16 NR: Creating this script as a new, cleaned-up version of the old
    plot_gauge_elevations.py
    --> Major rewrites, removal of old functions that are no longer used, etc.
      
"""

#==============================================================================

import numpy as np
import os
import xarray
from datetime import datetime
from datetime import timedelta
from pandas import to_datetime
from scipy import signal
from pytides2.tide import Tide

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

#==============================================================================
# MANUAL INPUT
#==============================================================================

'''
- start: Date at which to start computations & plotting
- stop: Date at which to stop
- stations: names of the stations to load, as used in the file names (!)
- exp: name of experiment/directory from which the model data should be loaded

All stations available:
stations = ['Cux', 'Brunsbuettel', 'Brokdorf', 'Glueckstadt',
             'Stadersand', 'Schulau', 'StPauli',
             'Zollenspieker', 'Geesthacht']
'''

# Set analysis interval
start = '2012-08-01 00:00:00'
stop = '2013-12-31 23:59:00' #1-minute resolved observational data available

# Choose stations for analysis
stations = ['Cux', 'Brunsbuettel', 'Brokdorf', 'Glueckstadt',
            'Stadersand', 'Schulau', 'StPauli', 'Zollenspieker']
# Labels by which to name the stations in the plot:
# (list has to be same length and order as stations)
station_labels = ['Cuxh. Sth.', 'Brunsbüttel', 'Brokdorf', 'Glückstadt',
            'Stadersand', 'Schulau', 'St. Pauli', 'Zollenspieker']

# Choose model data to use
exp = 'exp_2022-08-12' #experiment handle
conf = '182' #configuration (for parallelisation; needed to find directory path)

# Choose which functions to use:
    
#Load model data if True:
do_model = True
# do "continuous" tidal analysis for modelled data at every 10 x-indices
# inbetween gauges if True [only used if do_model==True]:
do_continuous = True 
#Load observational data if True:
do_meas = True
#Plot results, i.e., tidal elevations vs. time, if True:
plot_results = True
#Save temporal plot if True [only used if plot_results==True]:
save_fig = False
#save plot showing tidal amplitudes and phases if True:
save_tidal_fig = True
#Create a log file in the output directory if True:
log = True

# Settings for the low-pass filter:
filter_cutoff = 30 #cutoff period for the low-pass filter (h)
sampling_period_mod = 5*60 #sampling period of the modelled data (s)
sampling_period_obs = 60 #sampling period of the observational data (s)
order = 3 #order of the butterworth low-pass filter

# Info about observational data files:
skip_header = 20 #3 #Header rows to skip in observation files
skip_footer = 3 #0 #Footer rows to skip in observation files
miss_val = '' #-777 #Value that marks missing data in the observations



#==============================================================================
#%%  FUNCTION DEFINITONS
#==============================================================================

def find_datedirs(start, stop):
    
    """
    Lists all monthly directories of the form YYYYmm01 that will have to
    be loaded to get all _simulation_ data from start to stop time.
    Also returns epoch variable, which is required for time conversion for
    _observational_ data.
    
    INPUT:
        start: [scalar, dtype=str] str of the form YYYY-mm-dd HH:MM:SS
            defining the start of the time interval for which data should
            be loaded
        stop: [scalar, dtype=str] str of the form YYYY-mm-dd HH:MM:SS
            defining the end of the time interval for which data should
            be loaded
    OUTPUT:
        datedirs: [list, dtype=str] list of all directories of the form
            YYYYmm01 that are inside the time interval start:stop
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


def tidal_analysis(elv, time):
    
    """
    Applies the Pytides routine [1] to a given surface elevation time
    series and returns tidal amplitude and phase for the M2 and M4
    tidal constituents
    
    INPUT:
        elv: [np.ndarray, dtype=float] arr of dimension (N)
            Surface elevation time series at a given location
        time: [np.ndarray, datetime] arr of dimension (N)
            Corresponding time of surface elevations elv, time is
            given as datetime objects
    OUTPUT:
        M2_amp_cc, M2_phas_cc, M4_amp_cc, M4_phas_cc: [scalar, dtype=float]
            tidal amplitude (amp) and phase (phas) of the M2 and M4 tidal
            constituent, respectively
            
    [1] Pytides routine for tidal prediction and analysis by Sam Cox
        (sam.cox@cantab.net).
        Available on GitHub: https://github.com/sam-cox/pytides
    """
    
    demeaned = elv - np.nanmean(elv)
    demeaned[np.isnan(demeaned)] = 0
    tide = Tide.decompose(demeaned, time)
    constituent = np.asarray( [c.name for c in tide.model['constituent']] )
        
    amp = tide.model['amplitude']
    phas = tide.model['phase']
    
    M2_amp_cc = amp[np.where(constituent=='M2')[0][0]]
    M2_phas_cc = phas[np.where(constituent=='M2')[0][0]]
    M4_amp_cc = amp[np.where(constituent=='M4')[0][0]]
    M4_phas_cc = phas[np.where(constituent=='M4')[0][0]]
    S2_amp_cc = amp[np.where(constituent=='S2')[0][0]]
    S2_phas_cc = phas[np.where(constituent=='S2')[0][0]]
    
    return [M2_amp_cc, M2_phas_cc, M4_amp_cc, M4_phas_cc,
            S2_amp_cc, S2_phas_cc]



#==============================================================================
#%%  START EXECUTION
#==============================================================================

prntlst = ["Welcome to this new and cleaned-up tidal analysis!",
   "\n", "===============================", "\n",
	"Data Info:", "start = " + start, "stop = " + stop,
	"Model data from experiment #" + exp,
	"Lowpass filter cutoff at T = " + str(filter_cutoff) + " hrs",
	"\n", "Stations considered: " + str(stations), "\n",
	"===============================", "\n"]

print('\n')

if log:
    f = open('plots/tidal_elevs/log_' + str(start[:10])
             + '_' + str(stop[:10]) + exp + '.txt', 'w+')  
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

nn = len(stations) #number of stations
cc = 0             #counter

#initialise tidal amplitude and phase lists
M2_amp_mod = []   #modelled M2 tidal amplitude
M2_amp_obs = []   #observed M2 tidal amplitude
M2_phas_mod = []  #modelled M2 tidal phase
M2_phas_obs = []  #observed M2 tidal phase
M4_amp_mod = []   #same for M4 tide
M4_amp_obs = []
M4_phas_mod = []
M4_phas_obs = []
S2_amp_mod = []   #same for S2 tide
S2_phas_mod = []
S2_amp_obs = []
S2_phas_obs = []

datedirs, epoch = find_datedirs(start, stop)


#==============================================================================

# Loop through stations
for station in stations:
    
    print('*** Gauge: ' + station + ' ***')
    
    #Load the files belonging to the station:
    #(a) Model data
    if do_model:
        
        print('Loading simulation data...')
        
        time_mod_lst = []
        elv_mod_lst = []
        
        for datedir in datedirs:
            path = ('/silod6/reese/tools/getm/setups/elbe_realistic/store/' + exp +
                    '/OUT_' + conf + '/' + datedir + '/')
            directories = os.listdir(path)
            #remove salt stations from list (similar station names):
            directories = [d for d in directories if not ('SST' in d)]
            ii = np.where([(station in d) for d in directories])[0]
            directories = [directories[i] for i in ii]
            directories = [path + d for d in directories]
            time_mod = xarray.open_mfdataset(directories)['time'][:]  #time
            elv_mod = xarray.open_mfdataset(directories)['elev'][:,0,0] #elevations

            time_mod_lst.append(time_mod)
            elv_mod_lst.append(elv_mod)
        
        print('\t Processing...')
        time_mod = xarray.concat(time_mod_lst, dim='time')
        elv_mod = xarray.concat(elv_mod_lst, dim='time')

        elv_mod = np.asarray(elv_mod.loc[start:stop])
        time_mod = np.asarray(time_mod.loc[start:stop])

        #convert from datetime64 to datetime.datetime:
        time_mod = [to_datetime(t) for t in time_mod]
        time_mod = np.asarray([t.to_pydatetime() for t in time_mod])

        #low-pass filter
        print('\t Low-pass filtering...')
        cutoff = 1/(3600*filter_cutoff) #cutoff frequency of the filter (1/s)
        fs = 1/(sampling_period_mod) #sampling frequency of the data (1/s)
        nyq = fs/2 #Nyquist frequency
        normal_cutoff = cutoff/nyq #normalised cutoff frequency
        b, a = signal.butter(order, normal_cutoff,
                     btype='low', analog=False)
        elv_filt_mod = signal.filtfilt(b, a, elv_mod)
        
    #==========================================================================    
    
    #(b) Observational data
    if do_meas:
        
        print('Loading observations...')
        base = ('/silod6/reese/tools/getm/setups/elbe_realistic/' +
                'analysis/observations/gauges/')
        modeldirs = os.listdir(base)
        ii = np.where([(station in m) for m in modeldirs])[0][0]
        file_name = modeldirs[ii]
        meas_path = base + file_name
        
        f = open(meas_path, 'rb') # 'rb' = read binary
        meas_data = np.genfromtxt(f, skip_header=skip_header,
                                  skip_footer=skip_footer, delimiter='\t',
                                  dtype=str)
        f.close()
        
        print('\t Processing...')
        # shift from UTC to UTC+1
        mestart = datetime.strptime(start, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1)
        mestart = mestart.strftime("%Y-%m-%d %H:%M:%S")
        # shift from UTC to UTC+1
        mestop = datetime.strptime(stop, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1)
        mestop = mestop.strftime("%Y-%m-%d %H:%M:%S")
        start_ind = np.where([(mestart in m) 
                    for m in meas_data[:,0]])[0][0] #first index at start date
        stop_ind = np.where([(mestop in m)
                    for m in meas_data[:,0]])[0][0] + 1 #first index at stop date
        
        # extract data inside interval [start, stop]
        meas_data = meas_data[start_ind:stop_ind,:]
        
        # take care of missing values
        meas_data[meas_data[:,1]==miss_val,1] = '-999' #missing values
        meas_elv = np.asarray(meas_data[:,1]).astype(float) / 100 #convert from cm to m
        meas_elv[meas_elv==(-999/100)] = np.nan   #set missing values to NaN
        
        # convert time of observations to str
        meas_t_str = meas_data[:,0]
        meas_t_str = [ (datetime.strptime(mt, '%Y-%m-%d %H:%M:%S')
                        - timedelta(hours=1)) for mt in meas_t_str ]
        
        # remove missing values (NaN) from data
        kk = np.where(np.isnan(meas_elv))[0]
        meas_elv = np.delete(meas_elv, kk)
        meas_t_str = np.delete(meas_t_str, kk)
        
        # Convert UTC datetime to seconds since the Epoch:
        meas_time = [ ((mt - epoch).total_seconds()) for mt in meas_t_str ]
        meas_time = np.array(meas_time)
        
        # remove temporal loops from measured gauge data:
        mt_loops = signal.argrelmin(meas_time, order=1)
        meas_time = np.delete(meas_time, mt_loops)
        meas_elv = np.delete(meas_elv, mt_loops)
        meas_t_str = np.delete(np.asarray(meas_t_str), mt_loops) 
        
        # low-pass filter
        print('\t Low-pass filtering...')
        cutoff = 1/(3600*filter_cutoff) #cutoff frequency of the filter (1/s)
        fs = 1/(sampling_period_obs) #sampling frequency of the data (1/s)
        nyq = fs/2 #Nyquist frequency
        normal_cutoff = cutoff/nyq #normalised cutoff frequency
        b, a = signal.butter(order, normal_cutoff, 
                     btype='low', analog=False)
        elv_filt_meas = signal.filtfilt(b, a, meas_elv)
        
        # find data gaps (for plotting)
        delta_t = meas_time[1:] - meas_time[:-1]
        idx = np.where(delta_t>65)[0]+1
        # insert NaN at gap locations so gaps will be plotted as such
        meas_time = np.insert(meas_time, idx, meas_time[idx])
        meas_t_str = np.insert(meas_t_str, idx, meas_t_str[idx])
        meas_elv = np.insert(meas_elv, idx, np.nan)
        elv_filt_meas = np.insert(elv_filt_meas, idx, np.nan)


#==============================================================================
#%%  TIDAL ANALYSIS: M2, M4 tidal phase and amplitude
#============================================================================== 

    print('Doing the pytides...')
    
    print('\t Simulation...')
    [M2_amp_mod_cc, M2_phas_mod_cc,
     M4_amp_mod_cc, M4_phas_mod_cc,
     S2_amp_mod_cc, S2_phas_mod_cc] = tidal_analysis(elv_mod, time_mod)

    print('\t Observations...')
    [M2_amp_obs_cc, M2_phas_obs_cc,
     M4_amp_obs_cc, M4_phas_obs_cc,
     S2_amp_obs_cc, S2_phas_obs_cc] = tidal_analysis(meas_elv, meas_t_str)
        
    # Append to lists
    print('Adding to lists...')
    M2_amp_mod.append(M2_amp_mod_cc)
    M2_amp_obs.append(M2_amp_obs_cc)
    M2_phas_mod.append(M2_phas_mod_cc)
    M2_phas_obs.append(M2_phas_obs_cc)
    M4_amp_mod.append(M4_amp_mod_cc)
    M4_amp_obs.append(M4_amp_obs_cc)
    M4_phas_mod.append(M4_phas_mod_cc)
    M4_phas_obs.append(M4_phas_obs_cc)
    S2_amp_mod.append(S2_amp_mod_cc)
    S2_amp_obs.append(S2_amp_obs_cc)
    S2_phas_mod.append(S2_phas_mod_cc)
    S2_phas_obs.append(S2_phas_obs_cc)
    
    print('')
    print('Amplitudes:')
    print('\t M2 (mod): ' + str(M2_amp_mod[cc]))
    print('\t M2 (obs): ' + str(M2_amp_obs[cc]))
    print('\t M4 (mod): ' + str(M4_amp_mod[cc]))
    print('\t M4 (obs): ' + str(M4_amp_obs[cc]))
    print('\t S2 (mod): ' + str(S2_amp_mod[cc]))
    print('\t S2 (obs): ' + str(S2_amp_obs[cc]))
    
    print('Phases:')
    print('\t M2 (mod): ' + str(M2_phas_mod[cc]))
    print('\t M2 (obs): ' + str(M2_phas_obs[cc]))
    print('\t M4 (mod): ' + str(M4_phas_mod[cc]))
    print('\t M4 (obs): ' + str(M4_phas_obs[cc]))
    print('\t S2 (mod): ' + str(S2_phas_mod[cc]))
    print('\t S2 (obs): ' + str(S2_phas_obs[cc]))
    
    
    # Fill log:
    f = open('plots/paper/log_tidal_' + str(start[:10])
              + '_' + str(stop[:10]) + exp + '.txt', 'a')
    
    f.write('Results of tidal analysis for ' + station + ':')
    f.write('\n')
    
    f.write('Amplitudes:');  f.write("\n")
    f.write('\t M2 (mod): ' + str(M2_amp_mod[cc])); f.write("\n")
    f.write('\t M2 (obs): ' + str(M2_amp_obs[cc])); f.write("\n")
    f.write('\t M4 (mod): ' + str(M4_amp_mod[cc])); f.write("\n")
    f.write('\t M4 (obs): ' + str(M4_amp_obs[cc])); f.write("\n")
    f.write('\t S2 (mod): ' + str(S2_amp_mod[cc])); f.write("\n")
    f.write('\t S2 (obs): ' + str(S2_amp_obs[cc])); f.write("\n")
    
    f.write('Phases:');       f.write("\n")
    f.write('\t M2 (mod): ' + str(M2_phas_mod[cc])); f.write("\n")
    f.write('\t M2 (obs): ' + str(M2_phas_obs[cc])); f.write("\n")
    f.write('\t M4 (mod): ' + str(M4_phas_mod[cc])); f.write("\n")
    f.write('\t M4 (obs): ' + str(M4_phas_obs[cc])); f.write("\n")
    f.write('\t S2 (mod): ' + str(S2_phas_mod[cc])); f.write("\n")
    f.write('\t S2 (obs): ' + str(S2_phas_obs[cc])); f.write("\n")
    
    f.close()



#==============================================================================
#%%  PLOTTING: Tidal elevations (temporal variation)
#==============================================================================                   
   
    if plot_results:
        fig, ax = plt.subplots(figsize=(4.5,3), tight_layout=True)
        if do_model:
            ax.plot(time_mod, elv_mod-np.nanmean(elv_mod), color='k',
                    alpha=0.3, label='Mod')
            ax.plot(time_mod, elv_filt_mod-np.nanmean(elv_mod),
                    color='k', label='Mod: low-pass')
            
        if do_meas:
            ax.plot(meas_t_str, meas_elv-np.nanmean(meas_elv),
                    color='orange', alpha=0.5, label='Obs')
            ax.plot(meas_t_str, elv_filt_meas-np.nanmean(meas_elv),
                    color='chocolate', label='Obs: low-pass')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Rel. surface elevation (m)')
        ax.set_title('Tidal Range at Gauge ' + station)
        ax.set_ylim([-3.2,3])
        ax.legend(ncol=4, fontsize=8, loc=8)
        plt.xticks(rotation = 45)
        plt.grid(True, which='both')
        
        if ( save_fig ):
            fig.savefig('plots/tidal_elevs/Tidal_elev_' + station + '_'
                        + start[:10] + '_' + stop[:10] + exp + '.pdf')
            fig.savefig('plots/tidal_elevs/Tidal_elev_' + station + '_'
                        + start[:10] + '_' + stop[:10] + exp + '.png', dpi=200)

    #append to log:
    print('\n')
    print('====')
    print('\n')
    
    f = open('plots/tidal_elevs/log_' + str(start[:10])
              + '_' + str(stop[:10]) + exp + '.txt', 'a')
    f.write("\n"); f.write('===='); f.write("\n")
    f.write("\n")
    f.close()
                    
    cc += 1


#==============================================================================
#%%  Tidal analysis inbetween gauges
#==============================================================================

# grid locations (x,y) that are closest to the gauges
# in the order given in gauges
gauges = ['Cux', 'Osteriff', 'Brunsbuettel', 'Brokdorf', 'Glueckstadt',
                'Stadersand', 'Schulau', 'Seemannshoeft', 'StPauli', 
                'Zollenspieker', 'Geesthacht']
grid_locs = np.array(([88, 144, 171, 205, 236, 318, 392, 436, 459, 563, 612],
                      [144, 153, 166, 167, 170, 150, 159, 152, 163, 160, 153]))   

# x-indices of gauges that have been used for tidal analysis of *observations*:
stations_used = [(g in stations) for g in gauges]
obs_x = grid_locs[0, stations_used]


if (do_model and do_continuous):
    
    print('Making tidal analysis for intermediate setup locations...')
    print('(This may take a while... Just sit back and relax!)')
    
    # x-indices of grid cells every 10 cells, located inbetween gauges Cuxhaven
    # and Geesthacht
    ix = np.arange(90, 620, 10)
        
    # corresponding y-indices that are well inside the navigational channel
    # (picked manually)
    iy = np.array([ 150, 155, 157, 159, 159, 159, 159, 159, 166, 165, 
                    165, 163, 163, 163, 159, 159, 159, 159, 159, 159,
                    159, 159, 156, 152, 152, 152, 152, 152, 155, 155,
                    157, 157, 157, 157, 157, 158, 160, 162, 162, 162,
                    162, 162, 162, 158, 158, 158, 158, 158, 158, 158,
                    160, 160, 155 ])

    # x-indices every 10 grid cells that lie inbetween the outermost and
    # innermost gauges which are used for obervations
    interp_x = np.arange(10*np.ceil(min(obs_x/10)),
                         10*np.floor(max(obs_x/10))+10,
                         10, dtype=int)
    inds = [np.where(ix==xx)[0][0] for xx in interp_x]
    #corresponding y-indices, chosen from iy above
    interp_y = iy[inds]
    
    # Loop through all locations to start tidal analysis:
    for ll in range(len(interp_x)):
        
        print('----------')
        print('Round #' + str(ll+1).zfill(2) + ' of ' + str(len(interp_x)))
    
        #load data for given grid point:
        
        time_mod_lst = []
        elv_mod_lst = []
        
        for datedir in datedirs:
            path = ('/silod6/reese/tools/getm/setups/elbe_realistic/store/'
                    + exp + '/OUT_' + conf + '/' + datedir + '/')
            directories = os.listdir(path)
            #remove salt stations from list (similar station names):
            directories = [d for d in directories]
            ii = np.where([('2D_elv' in d) for d in directories])[0]
            directories = [directories[i] for i in ii]
            directories = [path + d for d in directories]
            time_mod = xarray.open_mfdataset(directories)['time'][:]  #time
            #elevations:
            elv_mod = xarray.open_mfdataset(directories)['elev'][:,interp_y[ll],
                                                                 interp_x[ll]]

            time_mod_lst.append(time_mod)
            elv_mod_lst.append(elv_mod)
            
        time_mod = xarray.concat(time_mod_lst, dim='time')
        elv_mod = xarray.concat(elv_mod_lst, dim='time')

        elv_mod = np.asarray(elv_mod.loc[start:stop])
        time_mod = np.asarray(time_mod.loc[start:stop])

        #convert from datetime64 to datetime.datetime:
        time_mod = [to_datetime(t) for t in time_mod]
        time_mod = np.asarray([t.to_pydatetime() for t in time_mod])
        
        #Tidal analysis:
        [M2_amp_mod_cc, M2_phas_mod_cc,
         M4_amp_mod_cc, M4_phas_mod_cc,
         S2_amp_mod_cc, S2_phas_mod_cc] = tidal_analysis(elv_mod, time_mod)

        # Append to lists
        print('')
        print('Adding to lists...')
        M2_amp_mod.append(M2_amp_mod_cc)
        M2_phas_mod.append(M2_phas_mod_cc)
        M4_amp_mod.append(M4_amp_mod_cc)
        M4_phas_mod.append(M4_phas_mod_cc)
        S2_amp_mod.append(S2_amp_mod_cc)
        S2_phas_mod.append(S2_phas_mod_cc)


#==============================================================================
#%%  PLOTTING: Preparation
#==============================================================================

print('Preparing plot...')

#Computing along-channel distance in km (for plotting)
#Path to GETM output in 'Mean' file:
startdate = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
date = startdate.strftime('%Y%m') + '01'
mean_base = '../store/' + exp + '/OUT_182/' + date + '/' 
mean_file_name = 'Mean_all.' + date + '.nc4'
path = mean_base + mean_file_name #full path to file
dx_tot = xarray.open_mfdataset(path)['dxc'][:]
dx = np.asarray(dx_tot)[157,:] #just need dx along thalweg
dx[454:509] = dx_tot[160,454:509] #fill Hamburg area with Norderelbe distances
dx[452] = 351.272 #manual distances where Norderelbe connects with main channel
dx[453] = 356.421
dx[509] = 256.211
dx[510] = 219.797
dx[np.isnan(dx)] = 0 #missing cells do not contribute to total length
#distance along thalweg j=157 in km, with upstream end at 0km:
x = np.cumsum(dx[::-1])[::-1]/1000


if do_continuous:
    if do_model:
        #add shared gauge locations to model x-values:
        mod_x = np.append(obs_x, interp_x)
        #then sort by mod_x values:
        inds = mod_x.argsort()
        mod_x = mod_x[inds]
        mod_x = x[mod_x]
        M2_amp_mod = np.asarray(M2_amp_mod)[inds]
        M4_amp_mod = np.asarray(M4_amp_mod)[inds]
        S2_amp_mod = np.asarray(S2_amp_mod)[inds]
        M2_phas_mod = np.asarray(M2_phas_mod)[inds]
        M4_phas_mod = np.asarray(M4_phas_mod)[inds]
        S2_phas_mod = np.asarray(S2_phas_mod)[inds]
    if do_meas:
        M2_phas_obs = np.asarray(M2_phas_obs)
        S2_phas_obs = np.asarray(S2_phas_obs)
else:
    if do_model:
        mod_x = obs_x
        M2_phas_mod = np.asarray(M2_phas_mod)
        S2_phas_mod = np.asarray(S2_phas_mod)
    if do_meas:
        M2_phas_obs = np.asarray(M2_phas_obs)
        S2_phas_obs = np.asarray(S2_phas_obs)
        
obs_x = x[obs_x]
xlims = [max(obs_x)+10, min(obs_x)-10]
xticks = np.arange(0,175,25)



#==============================================================================
#%%  PLOTTING: Tidal amplitudes and phases
#==============================================================================


if (do_model and do_meas):
    
    fig, ax = plt.subplots(2,1, figsize=(7.5,5.5), sharex=True,
                           tight_layout=True)
    
    amps = [ [M2_amp_mod, M2_amp_obs],
             [M4_amp_mod, M4_amp_obs],
             [S2_amp_mod, S2_amp_obs] ]
    
    phases = [ [M2_phas_mod, M2_phas_obs],
               [M4_phas_mod, M4_phas_obs],
               [S2_phas_mod, S2_phas_obs] ]
    
    colors = ['k', 'grey', 'lightgrey']
    labels_mod = ['mod M$_2$', 'mod M$_4$', 'mod S$_2$']
    labels_obs = ['obs M$_2$', 'obs M$_4$', 'obs S$_2$']
    markers    = ['o', 'v', '^']
    
    cc = 0
    
    #==========================================================================
    
    
    #Plot the results - tidal amplitude
    print('Plotting tidal amplitude...')
    
    for amp in amps:
        amp_mod = amp[0]
        amp_obs = amp[1]
        
        ax[0].plot(mod_x, amp_mod, color=colors[cc], linestyle='-',
                   label=labels_mod[cc])
        ax[0].plot(obs_x, amp_obs, marker=markers[cc], color=colors[cc],
                   alpha=1, linestyle='', label=labels_obs[cc])
        
        cc += 1
   
    ax[0].set_xlim(xlims)
    ax[0].set_ylim([np.nanmin( ( 
                          np.nanmin(M2_amp_mod), np.nanmin(M4_amp_mod),
                          np.nanmin(S2_amp_mod),
                          np.nanmin(M2_amp_obs), np.nanmin(M4_amp_obs),
                          np.nanmin(S2_amp_obs)
                             ) )-0.3,
                    np.nanmax( (
                          np.nanmax(M2_amp_mod), np.nanmax(M4_amp_mod),
                          np.nanmax(S2_amp_mod),
                          np.nanmax(M2_amp_obs), np.nanmax(M4_amp_obs),
                          np.nanmax(S2_amp_obs)
                             ) )+0.3])
    
    for kk in range(len(stations)):
        ax[0].vlines(obs_x[kk], ax[0].get_ylim()[0],
                     ax[0].get_ylim()[1], color='grey',
                     linewidth=0.8, alpha=0.7)
    
    ax[0].set_xticks(xticks)    
    ax[0].set_xticks(obs_x, minor=True)
    ax[0].set_ylabel('Tidal Amplitude (m)')
    ax[0].set_title('(a) M$_2$, M$_4$, and S$_2$ Tidal Amplitude')
    ax[0].tick_params('both', which='both', direction='in', bottom=True,
                      top=True, left=True, right=True)
    
    #==========================================================================    
      
    
    #Plot the results - tidal phase
    print('Plotting tidal phase...')
    
    cc = 0
    for phas in phases:
        
        phas_mod = np.array(phas[0])
        phas_obs = np.array(phas[1])
        
        if cc < 2:
            #shift M2 and M4 from [0,360] to [-180,180]:
            phas_mod[phas_mod > 180] = phas_mod[phas_mod > 180] - 360
            phas_mod[phas_mod < -180] = phas_mod[phas_mod < -180] + 360
            phas_obs[phas_obs > 180] = phas_obs[phas_obs > 180] - 360
            phas_obs[phas_obs < -180] = phas_obs[phas_obs < -180] + 360
            
        ax[1].plot(mod_x, phas_mod, color=colors[cc], linestyle='-',
                  label=labels_mod[cc])
        ax[1].plot(obs_x, phas_obs, marker=markers[cc], color=colors[cc],
                   alpha=1, linestyle='', label=labels_obs[cc])
        
        cc += 1
    
    ax[1].set_xlim(xlims)
    ax[1].set_ylim([-180, 220])
    
    for kk in range(len(stations)):
        ax[1].vlines(obs_x[kk], ax[1].get_ylim()[0],
                     ax[1].get_ylim()[1], color='grey',
                     linewidth=0.8, alpha=0.7)
    
    ax[1].set_xticks(xticks)
    ax[1].set_xticks(obs_x, minor=True)
    ax[1].set_xticklabels(station_labels, minor=True, rotation=90)
    ax[1].set_xlabel('$x$ (Elbe model-km)')
    ax[1].set_ylabel('Tidal Phase (°)')
    ax[1].set_title('(b) M$_2$, M$_4$, and S$_2$ Tidal Phase')
    ax[1].tick_params('both', which='both', direction='in', bottom=True,
                      top=True, left=True, right=True)
    plt.xticks(ha='right')
    
    #To get legend in correct order, sort handles and labels manually:
    handles, labels = ax[1].get_legend_handles_labels()
    seq = [0, 2, 4, 1, 3, 5]
    ax[1].legend([handles[idx] for idx in seq],[labels[idx] for idx in seq],
               ncol=2, loc=4, fontsize=8)
    
    if save_tidal_fig:
        fig.savefig('plots/paper/Tidal_amp_phas.pdf')
        fig.savefig('plots/paper/Tidal_amp_phas.png', dpi=300)
    
    plt.show()
            
        
#%%
    
print('')
print("Done! How would you rate your experience?")
print("(Just kidding. Have fun!)")
