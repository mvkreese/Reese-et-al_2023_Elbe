#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 09:43:00 2021

@author: Nina Reese / nina.reese@io-warnemuende.de

**Data processing / analysis for GETM model output**
Tidal Elbe setup @Nina Reese

Plots parameters related to model forcing and setup validation vs. time:
    
(a) Wind magnitude (daily average) and direction (7-day running mean) 10m above
    ground from meteorological model forcing. Spatiall averaged over the Elbe
    river mouth. 
(b) Summed-up total freshwater runoff (daily average) from all sources within
    the model  
(c) Simulated vs. observed temperature at a single station within the Elbe
    estuary 
(d) Simulated vs. observed surface and bottom salinity at station
    D4 Rhinplate-Nord 
(e) Simulated vs. observed salt stratification (bottom - surface) at station
    D4 Rhinplate-Nord
	- Model data: difference between bottom and surface sigma layer
	- Observational data: difference between two data sets (one for surface,
      one for bottom)
    
For panels (c)-(d):
	- applies low-pass filter to data to remove M2 tidal variability
	  (cutoff frequency can be set manually)
	- plots stratification vs. time (both instantaneous 
	  and low-pass filtered)
    
==> Fig. 3 in [1]

[1] N. Reese,U. Graewe, K. Klingbeil, X. Li, M. Lorenz, H. Burchard, 2023:
    Local mixing determines spatial structure of diahaline exchange flow in a
    mesotidal estuary – a study of extreme runoff conditions.
    J. Phys. Oceanogr., submitted.

LAST UPDATED:
    NR, 2023-02-06 11:46h
"""

#==============================================================================

import numpy as np
import pandas as pd
import os
import xarray
from datetime import datetime, timedelta, date
import scipy.signal as si
from scipy import signal
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter1d

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


#==============================================================================
# MANUAL INPUT
#==============================================================================

'''
- start: Date at which to start plotting
- stop: Date at which to stop plotting
- stations: names of the stations to load, as used in the file names (!)
- exp: name of experiment/directory from which the model data should be used
- config: configuration (for parallelisation; needed to find directory path)

All stations available:
tstation = ['Steinriff', 'Cuxhaven', 'Neufeldreede', 'Krummendeich']
sstation = ['Rhinplate']
'''

# Set analysis interval
start = '2012-08-01 00:00:00'
stop = '2013-12-31 23:55:00'

# Choose station for temperature analysis:
tstation = 'Cuxhaven'
int_d = 2.5 #depth above ground to which model temp data will be interpolated

# Choose station for stratification analysis
sstation = 'Rhinplate'

# Choose model data to use
exp = 'exp_2022-08-12' #experiment handle
conf = '182' #configuration

# Choose which functions to use:
do_model = True #Load model data, plot if plot_results=True
do_meas = True #Load measured data, plot if plot_results=True
low_pass = True #Also compute lowpass-filtered data (i.e., semidiurnal tides removed)
save_fig = True #Save plot. Only applies for plot_full=True.
log = True #If True, a log file will be created in the output directory

# Settings for analysis
filter_cutoff = 30 #cutoff period for the lowpass filter (h)
levels = [1, -1] #z-level (sigma level) to choose for bottom and surface, ranges from 0 (bottom) to 19 (surface)

# Info about observation data files
miss_val = '' #Value that marks missing data in the observation files
skip_header = 19 #Header rows to skip in observation files
skip_footer = 3 #Footer rows to skip in observation files


#==============================================================================
#%%  FUNCTION DEFINITONS
#==============================================================================

def find_datedirs(start, stop):
    
    """
    Computes all monthly directories of the form YYYYmm01 that will have to
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


def vert_interp(var, hn, H=2.5, int_d=2.5):
    
    print('Interpolating in vertical direction...')

    if H <= int_d: 
        raise Exception("Error: Station too shallow!") #exit w/ error message
    
    depth_interp = H-int_d #intepolation depth is 2.5m above ground
    v_interp = np.zeros((np.shape(var)[0])) #initiate interpolated variable array
    
    for ii in range(np.shape(var)[0]): #loop through time dimension
        depths = np.cumsum(hn[ii, ::-1]) - hn[ii, ::-1]/2 #depth at center of each sigma layer
        if (max(depths)-depth_interp) <= 0:
            v_interp = np.nan #if 2.5m above ground is dry/not flooded, set var to nan
        else:
            v_interp[ii] = griddata(depths, var[ii,:], depth_interp, method='cubic')
        
    return v_interp


#==============================================================================


def load_observations(station, base, start, stop, low_pass, filter_cutoff, 
		skip_header, skip_footer, loc='surf'):

    
    """
    Don't mind me simply not wanting to type things twice.
    Loads and processes observational data for a given station.
    
    INPUT:
        station: [scalar, dtype=str] str containing the name of the station
            for which data will be loaded
        base: [scalar, dtype=str] path to the directory in which the
            respective station data file is located
        start: [scalar, dtype=str] str of the form YYYY-mm-dd HH:MM:SS
            defining the start of the time interval for which data should
            be loaded
        stop: [scalar, dtype=str] str of the form YYYY-mm-dd HH:MM:SS
            defining the end of the time interval for which data should
            be loaded
        low_pass: [scalar, dtype=Boolean] if True, data will also be
            low-pass filtered w/ the cutoff frequency given by filter_cutoff
            to remove tidal variation
        filter_cutoff: [scalar, dtype=float] cutoff 'frequency' for the
            low-pass filter (given as period duration in hrs)
        skip_header: [scalar, dtype=int] no. of header rows to skip in
            the observation data files
        skip_footer: [scalar, dtype=int] no. of footer rows to skip in
            the observation data files
        loc: [scalar, dtype=str] str, either "surf" or "bott"
            defining whether surface ("surf") or bottom ("bott") data
            will be loaded
    OUTPUT:
        meas_time: [numpy array, dtype=datetime] 1D array (dim=(N))
            containing sampling time for each data point
        meas_S: [numpy array, dtype=float] 1D array (dim=(N))
            containing observed salinity for each data point
        S_filt_meas: [numpy array, dtype=float] 1D array (dim=(N))
            containing low-pass filtered, observed salinity for each data point.
            Cutoff frequency for the filter is given by filter_cutoff.
            If low_pass==False, a scalar of value 0 (float) will be returned instead.
    """

    mdirs = os.listdir(base) #directories for observational/measured data
    mdirs = [m for m in mdirs if loc in m]
    ii = np.where([station in m for m in mdirs])[0][0]
    file_name = mdirs[ii]
    meas_path = base + file_name
    
    f = open(meas_path, 'r') # 'r' = read
    meas_data = np.genfromtxt(f, skip_header=skip_header, skip_footer=skip_footer,
                              delimiter='\t', dtype=str)
    f.close()
    
    #string to datetime; shift from UTC to UTC+1:
    mestart = datetime.strptime(start, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1)
    mestart = mestart.strftime("%Y-%m-%d %H:%M:%S")
    #string to datetime; shift from UTC to UTC+1:
    mestop = datetime.strptime(stop, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1)
    mestop = mestop.strftime("%Y-%m-%d %H:%M:%S")
    #first index of the start date:
    start_ind = np.where([(mestart in m) for m in meas_data[:,0]])[0][0]
    #first index of the stop date:
    stop_ind = np.where([(mestop in m) for m in meas_data[:,0]])[0][0] + 1
    meas_data = meas_data[start_ind:stop_ind,:]
    meas_data[meas_data[:,1]==miss_val,1] = '-999'
    meas_S = meas_data[:,1].astype(float)
    meas_t_str = meas_data[:,0]
    #observational times with -1h due to shift from UTC+1 to UTC:
    meas_t_str = [ (datetime.strptime(mt, '%Y-%m-%d %H:%M:%S')
                        - timedelta(hours=1)) for mt in meas_t_str ]
    meas_time = meas_t_str
    
    #remove missing values (NaN) from data:
    kk = np.where(meas_S==-999)[0]
    meas_S = np.delete(meas_S, kk)
    meas_time = np.delete(meas_time, kk)
    
    # Convert UTC datetime to seconds since the Epoch:
    meas_time_cu = [ ((mt - epoch).total_seconds()) for mt in meas_time ]
    meas_time_cu = np.array(meas_time_cu)
    
    #remove temporal loops from measured gauge data:
    mt_loops = si.argrelmin(meas_time_cu, order=1)
    meas_time = np.delete(meas_time, mt_loops)
    meas_S = np.delete(meas_S, mt_loops)
    
    #find some data gaps for plotting
    delta_t = meas_time_cu[1:] - meas_time_cu[:-1]
    idx = np.where(delta_t>305)[0]+1
    
    S_filt_meas = 0 #initiate as 0 in case low_pass is False

    if low_pass: #apply a low-pass filter to remove M2 tidal cycle
        print("\t Low-pass filtering...")
        b, a = signal.butter(3, 1/(3600*filter_cutoff)/(1/300/2), 
                 btype='low', analog=False)
        S_filt_meas = signal.filtfilt(b, a, meas_S)
        S_filt_meas = np.insert(S_filt_meas, idx, np.nan)
    
    #insert nan values at idx to avoid connecting lines between blocks of missing data
    #(for plotting)
    meas_S = np.insert(meas_S, idx, np.nan)
    meas_time = np.insert(meas_time, idx, meas_time[idx])
    
    return meas_time, meas_S, S_filt_meas



#==============================================================================
#%% START EXECUTION
#==============================================================================

prntlst = ["Moin!", "\n", "===============================", "\n",
	"Data Info:", "start = " + start, "stop = " + stop,
	"Model data from experiment #" + exp,
	"Lowpass filter cutoff at T = " + str(filter_cutoff) + " hrs",
	"\n", "Stations considered: " + str(sstation) + ' (strat), ' 
    + str(tstation), " (temp)\n",
	"===============================", "\n"]

print('')

if log:
    f = open('plots/stratification/log_salt_strat_' + str(start[:10]) + '_'
             + str(stop[:10]) + exp + '.txt', 'w') 
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

#finding all date directories to loop through:
datedirs, epoch = find_datedirs(start, stop)
  
#Initialise figure
fig, ax = plt.subplots(5,1, figsize=(7,8.5), tight_layout=True, sharex=True)

#==============================================================================
#%% (1) WIND
#==============================================================================

print('\n\n===================')
print('WIND')
print('===================\n')

startdate = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
stopdate = datetime.strptime(stop, '%Y-%m-%d %H:%M:%S')
startyear = startdate.strftime('%Y')
stopyear = stopdate.strftime('%Y')

# Some initialisation:
wind_data = []
wind_data_ii = []
variables = ['time', 'u10', 'v10']

for year in range(int(startyear), int(stopyear)+1):

    #Path to meteo file containing wind forcing:
    wind_path = '/data/meteo/dwd/getm/data/DWD_' + str(year) + '.nc'
    
    print('\t Reading wind data for ' + str(year) + '...')
    wind_data_ii = xarray.open_dataset(wind_path)[variables]
    wind_data_ii = wind_data_ii.isel(lat=slice(28,37), lon=slice(215, 227))
    
    if year == startdate.year:
        wind_data = wind_data_ii
    else:
        wind_data = xarray.concat([wind_data, wind_data_ii],
                                  dim='time', data_vars='all')
  
print('\t Converting arrays...')
datetimeindex = wind_data.indexes['time'].to_datetimeindex()
wind_data['time'] = datetimeindex
wind_data = wind_data.sel(time=slice(start,stop))

wind_time = wind_data['time'].load()
u10 = wind_data['u10'].load()
v10 = wind_data['v10'].load()

print('\t Spatially averaging wind data from the Elbe mouth...')
u10 = np.nanmean(u10, axis=1) #average along y-axis
u10 = np.nanmean(u10, axis=1) #average along x-axis
v10 = np.nanmean(v10, axis=1) #average along y-axis
v10 = np.nanmean(v10, axis=1) #average along x-axis

print('\t Computing magnitude and angle, applying running mean...')
u_wind = np.sqrt(u10**2 + v10**2) #magnitude of wind
phi = np.arctan2(-v10, -u10) #angle between x- and y- wind components
#7-day running mean:
siderian_day = 86164.0989 #length of siderian day (s)
delta_t = np.round((wind_time[1]-wind_time[0]).astype(np.float32)/10**9) # time step size in s
filt_len = int(np.round(1*siderian_day/delta_t)) #index length of filter
u_wind = uniform_filter1d(u_wind, size=filt_len)
phi = uniform_filter1d(phi, size=7*filt_len)

#==============================================================================

print('\t Plotting...')

# Plotting wind magnitude
ax[0].plot(wind_time, u_wind, color='k')
ax[0].set_xlim([startdate, stopdate])
ax[0].set_ylim([0, 22])
yticks = [0, 5, 10]
minoryticks = np.linspace(0,10,5)
ax[0].set_yticks(yticks)
ax[0].set_yticks(minoryticks, minor=True)

#loop through points in time and plot wind direction for each point
trans = []
for ii in range(0, len(wind_time), 7):
 	trans = ax[0].transData.transform((mdates.date2num(wind_time[ii]), 12)) #transform time points from data to axis coordinates pt. 1
 	trans = ax[0].transAxes.inverted().transform(trans) #pt. 2
 	inset_ax = ax[0].inset_axes([trans[0]-0.15, trans[1], 0.3, 0.3]) #create inset coordinate system w/ origin at given time point
 	inset_ax.arrow(0, 0, np.cos(phi[ii]), np.sin(phi[ii]), color='k', alpha=0.3,
                width=0.001, length_includes_head=True, head_width=0.1, overhang=0) #plot line with an arrowhead
 	inset_ax.set_xlim([-1,1])
 	inset_ax.set_ylim([-1,1])
 	inset_ax.set_aspect('equal')
 	inset_ax.axis('off')

ax[0].grid(True, which='major', axis='both', zorder=-100)
ax[0].grid(True, which='minor', axis='both', linestyle='--', zorder=-100)


#==============================================================================
#%% (2) RUNOFF
#==============================================================================

print('\n\n===================')
print('RUNOFF')
print('===================\n')

print('Loading river data...')
river_path = '../rivers.nc'

# Read riverinfo to get names and x-indices   
rivers = pd.read_table('../riverinfo.dat', skiprows=1, comment='!',
            delimiter=' ', header=None, usecols=[1,3], names=['etac', 'name'])
# remove duplicates (a given runoff might have multiple entrance points)            
rivers = rivers.drop_duplicates()

# Load runoff for each tributary, add to total runoff
Q_r = 0
for r_name in rivers['name']:
	Q_r_ri = xarray.open_mfdataset(river_path)[r_name][:]
	Q_r_ri = Q_r_ri.loc[start:stop]
	Q_r += Q_r_ri

Q_r_time = Q_r.time
Q_r = Q_r.data

ax[1].plot(Q_r_time, Q_r, linestyle='-', color='k')


#==============================================================================
#%% (3) TEMPERATURE
#==============================================================================

print('\n\n===================')
print('TEMPERATURE')
print('===================\n')

#(a) Model data
if do_model:
    
    mod_time_lst = []
    mod_T_lst = []
    hn_lst = []
    H = 0.0
    
    for datedir in datedirs: #load monthly data
        path = '/silod6/reese/tools/getm/setups/elbe_realistic/store/' + exp + '/OUT_' + conf + '/' + datedir + '/'
        directories = os.listdir(path)
        directories = [d for d in directories if ('SST' in d)] #remove gauges from list (similar station names)
        ii = np.where([(tstation in d) for d in directories])[0] #should only be 1 directory per loop, but in case it isn't... we load 'em all
        directories = [directories[i] for i in ii]
        directories = [path + d for d in directories]
        mod_T = xarray.open_mfdataset(directories)['temp'][:,:,0,0]
        mod_time = xarray.open_mfdataset(directories)['time'][:]
        hn = xarray.open_mfdataset(directories)['hn'][:,:,0,0] #sigma layer thickness
        H = xarray.open_mfdataset(directories)['bathymetry'][:] #bathy depth at station location
        
        mod_time_lst.append(mod_time) #add monthly data to total list
        mod_T_lst.append(mod_T)
        hn_lst.append(hn)
    
    mod_time = xarray.concat(mod_time_lst, dim='time') #concatenate data into one xarray
    mod_T = xarray.concat(mod_T_lst, dim='time')
    hn = xarray.concat(hn_lst, dim='time')
    mod_time = np.asarray(mod_time.loc[start:stop]) #chop off parts that are not contained in [start,stop]
    mod_T = np.asarray(mod_T.loc[start:stop])
    hn = np.asarray(hn.loc[start:stop])
    H = np.asarray(H)[0]
    
    mod_T = vert_interp(mod_T, hn, H, int_d)
    
    if low_pass: #apply a low-pas filter to remove M2 tidal cycle
        b, a = signal.butter(3, 1/(3600*filter_cutoff)/(1/1800/2), 
                              btype='low', analog=False)
        T_filt_mod = signal.filtfilt(b, a, mod_T)
    
        
    print('\t Adding to plot...')
    ax[2].plot(mod_time, mod_T, linestyle='-', linewidth=0.7,
              color='k', alpha=0.3, label='Mod')
    if low_pass:
        ax[2].plot(mod_time, T_filt_mod, linestyle='-', linewidth=1,
                  color='k', alpha=1, label='Mod:\n low-pass')
    
#==============================================================================    

#(b) Measured data
if do_meas:
    base = '/silod6/reese/tools/getm/setups/elbe_realistic/analysis/observations/temperature/'
    modeldirs = os.listdir(base)
    ii = np.where([(tstation in m) for m in modeldirs])[0][0]
    file_name = modeldirs[ii]
    meas_path = base + file_name
    
    f = open(meas_path, 'r') # 'r' = read
    meas_data = np.genfromtxt(f, skip_header=19, skip_footer=3, delimiter='\t', dtype=str)
    f.close()
    
    mestart = datetime.strptime(start, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1) #shift from UTC to UTC+1
    mestart = mestart.strftime("%Y-%m-%d %H:%M:%S")
    mestop = datetime.strptime(stop, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1) #shift from UTC to UTC+1
    mestop = mestop.strftime("%Y-%m-%d %H:%M:%S") #"2013-12-11 08:25:00" #(last data point in 2013 for Brunsbuettel)
    start_ind = np.where([(mestart in m) for m in meas_data[:,0]])[0][0] #first index of the start date
    stop_ind = np.where([(mestop in m) for m in meas_data[:,0]])[0][0] + 1 #first index of the stop date
    meas_data = meas_data[start_ind:stop_ind,:]
    meas_data[meas_data[:,1]==miss_val,1] = '-999'
    meas_T = meas_data[:,1].astype(float)
    meas_t_str = meas_data[:,0]
    meas_t_str = [ (datetime.strptime(mt, '%Y-%m-%d %H:%M:%S')
                    - timedelta(hours=1)) for mt in meas_t_str ] #-1h due to shift from UTC+1 to UTC
    meas_time = meas_t_str
    
    #remove missing values (NaN) from data:
    kk = np.where(meas_T==-999)[0]
    meas_T = np.delete(meas_T, kk)
    meas_time = np.delete(meas_time, kk)
    
    # Convert UTC datetime to seconds since the Epoch:
    meas_time_cu = [ ((mt - epoch).total_seconds()) for mt in meas_time ]
    meas_time_cu = np.array(meas_time_cu)
    
    
    #remove temporal loops from measured gauge data:
    mt_loops = signal.argrelmin(meas_time_cu, order=1)
    meas_time = np.delete(meas_time, mt_loops)
    meas_T = np.delete(meas_T, mt_loops)
    
    #find some data gaps for plotting
    delta_t = meas_time_cu[1:] - meas_time_cu[:-1]
    idx = np.where(delta_t>305)[0]+1
    
    if low_pass: #apply a low-pas filter to remove M2 tidal cycle
        b, a = signal.butter(3, 1/(3600*filter_cutoff)/(1/300/2), 
                  btype='low', analog=False)
        T_filt_meas = signal.filtfilt(b, a, meas_T)
        T_filt_meas = np.insert(T_filt_meas, idx, np.nan)
    
    #insert nan values at idx to avoid connecting lines between blocks of missing data
    #(for plotting)
    meas_T = np.insert(meas_T, idx, np.nan)
    meas_time = np.insert(meas_time, idx, meas_time[idx])
    
    
    print('\t Adding to plot...')
    ax[2].plot(meas_time, meas_T, linestyle='-', linewidth=1,
            color='orange', alpha=0.5, label='Obs')
    if low_pass:
        ax[2].plot(meas_time, T_filt_meas, linestyle='-', linewidth=1,
            color='chocolate', alpha=1, label='Obs:\n low-pass')



#==============================================================================
#%% (4) SALINITY AND STRATIFICATION
#==============================================================================

print('\n\n===================')
print('STRATIFICATION')
print('===================\n')

#(a) Model data
if do_model:

    print("Loading model data...")
    
    mod_time_lst = []
    mod_S_surf_lst = []
    mod_S_bott_lst = []
    
    minx = 0 #lower xlim of plot x-axis
    maxx = 0 #upper xlim of plot x-axis

    #initialising some stuff
    mod_time_iim1 = []
    mod_S_surf_iim1 = []
    mod_S_bott_iim1 = []

    #finding startdate, stopdate and 1st date after startdate:
    dateints = np.array([int(dd) for dd in datedirs])
    startint = min(dateints)
    startp1int = min(dateints[dateints>startint])
    stopint = max(dateints)

    
    for datedir in datedirs: #load monthly data
    
        year = str(datedir[:4])
        month = str(datedir[4:6]).zfill(2)
        print('')
        print('\t ==============================\n')
        print('\t MONTH: ' + month + ' ' + year)

        path = ('/silod6/reese/tools/getm/setups/elbe_realistic/store/'
                + exp + '/OUT_' + conf + '/' + datedir + '/')
        directories = os.listdir(path)
        #remove gauges from list (similar station names):
        directories = [d for d in directories if ('SST' in d)]
        ii = np.where([(sstation in d) for d in directories])[0]
        directories = [directories[i] for i in ii]
        directories = [path + d for d in directories]
        mod_S_surf = xarray.open_mfdataset(directories)['salt'][:,-1,0,0] #0, -7, -3 #-2
        mod_S_bott = xarray.open_mfdataset(directories)['salt'][:,0,0,0] #0, -7, -3 #-2
        mod_time = xarray.open_mfdataset(directories)['time'][:]
        
        #mod_time = xarray.open_mfdataset(path + 'Mean*')['time'][:]
        #mod_S_surf = xarray.open_mfdataset(path + 'Mean*')['salt_mean'][:, -1, 152, 222] #152, 228 (~original)
        #mod_S_bott = xarray.open_mfdataset(path + 'Mean*')['salt_mean'][:, 0, 152, 222] #OR 156, 210 (downstream)

        mod_time_ii = mod_time
        mod_S_surf_ii = mod_S_surf
        mod_S_bott_ii = mod_S_bott
        
        if datedir==str(startint):
            minx = np.nanmin(mod_time_ii) #lower xaxis limit
            
        else:
            
            print('\t Concatenating arrays...')
            mod_time = xarray.concat([mod_time_iim1, mod_time_ii], 
                                     dim='time', data_vars='all')
            mod_S_surf = xarray.concat([mod_S_surf_iim1, mod_S_surf_ii], 
                                     dim='time', data_vars='all')
            mod_S_bott = xarray.concat([mod_S_bott_iim1, mod_S_bott_ii], 
                                     dim='time', data_vars='all')
    
            #chop off parts that are not contained in [start,stop]
            mod_time = np.asarray(mod_time.loc[start:stop])
            mod_S_surf = np.asarray(mod_S_surf.loc[start:stop])
            mod_S_bott = np.asarray(mod_S_bott.loc[start:stop])
            
            if low_pass: #apply a low-pass filter to remove M2 tidal cycle
                print("\t Low-pass filtering...")
                b, a = signal.butter(3, 1/(3600*filter_cutoff)/(1/1800/2), 
                                     btype='low', analog=False)
                S_filt_mod_surf = signal.filtfilt(b, a, mod_S_surf)
                S_filt_mod_bott = signal.filtfilt(b, a, mod_S_bott)
        
#==============================================================================
            
            print('\t Making plot additions...')
            if datedir==str(startp1int):
                sta = int(0)
                sto = int(np.round(len(mod_time_ii.time)/2))-1
                
                ax[4].plot(mod_time[sta:-sto],
                           mod_S_bott[sta:-sto]-mod_S_surf[sta:-sto],
                           linestyle='-', linewidth=0.7, color='k', alpha=0.3,
                           label='Mod')
                if low_pass:
                    ax[4].plot(mod_time[sta:-sto],
                               S_filt_mod_bott[sta:-sto]-S_filt_mod_surf[sta:-sto],
                               linestyle='-', linewidth=1, color='k',
                               alpha=1, label='Mod:\n low-pass')
                    ax[3].plot(mod_time[sta:-sto], S_filt_mod_surf[sta:-sto],
                               linestyle='-', linewidth=1, color='k', alpha=1,
                               label='Mod:\n low-pass surf')
                    ax[3].plot(mod_time[sta:-sto], S_filt_mod_bott[sta:-sto],
                               linestyle='--', linewidth=1, color='k', alpha=1,
                               label='Mod:\n low-pass bott')
                    
            if datedir==str(stopint):
                sta = int(np.round(len(mod_time_iim1.time)/2))-1
                
                ax[4].plot(mod_time[sta:], mod_S_bott[sta:]-mod_S_surf[sta:],
                           linestyle='-', linewidth=0.7, color='k', alpha=0.3)
                if low_pass:
                    ax[4].plot(mod_time[sta:],
                               S_filt_mod_bott[sta:]-S_filt_mod_surf[sta:],
                               linestyle='-', linewidth=1, color='k', alpha=1)
                    ax[3].plot(mod_time[sta:], S_filt_mod_surf[sta:],
                               linestyle='-', linewidth=1, color='k', alpha=1)
                    ax[3].plot(mod_time[sta:], S_filt_mod_bott[sta:],
                               linestyle='--', linewidth=1, color='k', alpha=1)
                maxx=max(mod_time)
            
            else:
                sta = int(np.round(len(mod_time_iim1.time)/2))-1
                sto = int(np.round(len(mod_time_ii.time)/2))-1
                
                ax[4].plot(mod_time[sta:-sto],
                           mod_S_bott[sta:-sto]-mod_S_surf[sta:-sto],
                           linestyle='-', linewidth=0.7, color='k', alpha=0.3)
                if low_pass:
                    ax[4].plot(mod_time[sta:-sto],
                               S_filt_mod_bott[sta:-sto]-S_filt_mod_surf[sta:-sto],
                               linestyle='-', linewidth=1, color='k', alpha=1)
                    ax[3].plot(mod_time[sta:-sto], S_filt_mod_surf[sta:-sto],
                               linestyle='-', linewidth=1, color='k', alpha=1)
                    ax[3].plot(mod_time[sta:-sto], S_filt_mod_bott[sta:-sto],
                               linestyle='--', linewidth=1, color='k', alpha=1)
                    
        print('\t Preparing next month...')
        mod_time_iim1 = mod_time_ii
        mod_S_bott_iim1 = mod_S_bott_ii
        mod_S_surf_iim1 = mod_S_surf_ii
    
    
#==============================================================================    

#(b) Measured data
if do_meas:

    print("\nLoading observational data...")

    base = ('/silod6/reese/tools/getm/setups/elbe_realistic/analysis/' +
            'observations/stratification/')
    
    [meas_time_surf, meas_S_surf, meas_S_surf_lp] = load_observations(
      sstation, base, start, stop, low_pass, filter_cutoff, skip_header,
      skip_footer, loc='surf')
    [meas_time_bott, meas_S_bott, meas_S_bott_lp] = load_observations(
      sstation, base, start, stop, low_pass, filter_cutoff, skip_header,
      skip_footer, loc='bott')

    print("\t Cleaning up data...")


    #now find and delete all data points that only exist for surface OR bottom:

    #remove repeated time stamps
    meas_time_surf_sse = np.array([ ((mt - epoch).total_seconds())
                                   for mt in meas_time_surf ])
    meas_time_surf_unique = list(dict.fromkeys(meas_time_surf_sse))
    
    #choose only first index where certain timestamp appears:
    idx = [np.where(meas_time_surf_sse==mtsu)[0][0]
           for mtsu in meas_time_surf_unique]
    meas_time_surf = meas_time_surf[idx]
    meas_S_surf = meas_S_surf[idx]
    meas_S_surf_lp = meas_S_surf_lp[idx]

    meas_time_bott_sse = np.array([ ((mt - epoch).total_seconds())
                                   for mt in meas_time_bott ])
    meas_time_bott_unique = list(dict.fromkeys(meas_time_bott_sse))
    
    #choose only first index where certain timestamp appears:
    idx = [np.where(meas_time_bott_sse==mtbu)[0][0]
           for mtbu in meas_time_bott_unique]
    meas_time_bott = meas_time_bott[idx]
    meas_S_bott = meas_S_bott[idx]
    meas_S_bott_lp = meas_S_bott_lp[idx]


    while(len(meas_time_surf) != len(meas_time_bott)):
        
        indices = [ii for ii in np.arange(len(meas_time_surf))
                   if not (meas_time_surf[ii] in list(meas_time_bott))]
    
        if len(indices)!=0:
            meas_S_surf = np.delete(meas_S_surf, indices)
            meas_S_surf_lp = np.delete(meas_S_surf_lp, indices)
            meas_time_surf = np.delete(meas_time_surf, indices)
            
        indices = [ii for ii in np.arange(len(meas_time_bott))
                   if not (meas_time_bott[ii] in list(meas_time_surf))]
        
        if len(indices)!=0:
            meas_S_bott = np.delete(meas_S_bott, indices)
            meas_S_bott_lp = np.delete(meas_S_bott_lp, indices)
            meas_time_bott = np.delete(meas_time_bott, indices)
    
#==============================================================================
    
    print('\t Adding to plot...\n')
    ax[4].plot(meas_time_surf, meas_S_bott-meas_S_surf, linestyle='-', linewidth=1,
           color='orange', alpha=0.5, label='Obs')
    if low_pass:
        ax[4].plot(meas_time_surf, meas_S_bott_lp-meas_S_surf_lp, linestyle='-',
                   linewidth=1,  color='chocolate', alpha=1, label='Obs:\n low-pass')
        ax[3].plot(meas_time_surf, meas_S_surf_lp, linestyle='-', linewidth=1,
                  color='chocolate', alpha=1, label='Obs:\n low-pass surf')
        ax[3].plot(meas_time_surf, meas_S_bott_lp, linestyle='--', linewidth=1,
                  color='chocolate', alpha=1, label='Obs:\n low-pass bott')
        

#==============================================================================
#%%  PLOTTING: Finish
#==============================================================================                   

print("\nFinishing plot...")

ax[0].set_ylabel('$|u_{10}|$ (m/s)')
ax[0].text(0.82, 0.87, ('(a) Wind'), fontweight='bold',
           transform=ax[0].transAxes)
ax[0].yaxis.set_label_coords(-.04, .23)
ax[0].tick_params('both', which='both', direction='in', bottom=True, top=True,
                        left=True, right=True)

ax[1].set_ylabel('$Q_{\mathrm{r}}$ (m$^3$/s)')
ax[1].text(0.82, 0.75, ('(b) Total\n      Runoff'), fontweight='bold',
           transform=ax[1].transAxes)
ax[1].set_ylim([0,4400])
ax[1].tick_params('both', which='both', direction='in', bottom=True, top=True,
               left=True, right=True, labelleft=True, labelright=False)

ax[2].set_ylabel('Temperature (°C)')
ax[2].text(0.82, 0.75, ('(c) Temp.\n      ' + tstation[:4] + '. AL'), fontweight='bold',
           transform=ax[2].transAxes)
ax[2].set_ylim([-1,26])
ax[2].legend(ncol=2, fontsize=8, loc=2, bbox_to_anchor=(0.15, 0.5, 0.5, 0.5))
ax[2].tick_params('both', which='both', direction='in', bottom=True, top=True,
                left=True, right=True, labelleft=True, labelright=False)

ax[3].set_ylabel('$S$ (g/kg)')
ax[3].text(0.82, 0.75, ('(d) Salinity\n      ' + sstation[:6] + '.'), fontweight='bold',
           transform=ax[3].transAxes)
ax[3].set_ylim([0,3.2])
ax[3].legend(ncol=2, fontsize=8, loc=9)
ax[3].tick_params('both', which='both', direction='in', bottom=True, top=True,
                left=True, right=True, labelleft=True, labelright=False)

ax[4].set_xlabel('Time')
ax[4].set_ylabel('$S_{\mathrm{bott}} - S_{\mathrm{surf}}$ (g/kg)')
ax[4].text(0.82, 0.75, ('(e) Strat.\n      ' + sstation[:6] + '.'), fontweight='bold',
           transform=ax[4].transAxes)
ax[4].set_ylim([-0.5,2.5])
ax[4].legend(ncol=2, fontsize=8, loc=9)
ax[4].tick_params('both', which='both', direction='in', bottom=True, top=True,
                left=True, right=True, labelleft=True, labelright=False)


if (int(startyear) <= 2012 and int(stopyear) >= 2012):
    # add a rectangle for September to all axes
    for aa in ax:
        ymin, ymax = aa.get_ylim()
        sta = date(2012, 9, 1)
        sto = date(2012, 10, 1)
        rect = Rectangle((sta, ymin), width=(sto-sta), height=(ymax-ymin), 
                         edgecolor=None, facecolor='k', alpha=0.1, zorder=-3)
        aa.add_artist(rect)
if (int(startyear) <= 2013 and int(stopyear) >= 2013):
    # add a rectangle for June
    for aa in ax:
        ymin, ymax = aa.get_ylim()
        sta = date(2013, 6, 1)
        sto = date(2013, 7, 1)
        rect = Rectangle((sta, ymin), width=(sto-sta), height=(ymax-ymin), 
                         edgecolor=None, facecolor='k', alpha=0.1, zorder=-3)
        aa.add_artist(rect)

if ( save_fig ):
    fig.savefig('plots/paper/Forcing_Stratification.pdf')
    fig.savefig('plots/paper/Forcing_Stratification.png', dpi=300)

plt.show()
    
print("\nDone!")
