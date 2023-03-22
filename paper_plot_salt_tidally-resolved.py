# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:02:41 2023

@author: Nina
"""

#==============================================================================

import numpy as np
#import pandas as pd
import os
import xarray
from datetime import datetime#, timedelta, date
#from datetime import timedelta
#from datetime import date
#import scipy.signal as si
#from scipy import signal
#from scipy.interpolate import griddata
#from scipy.ndimage import uniform_filter1d

import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
#import matplotlib.pylab as pl
#from matplotlib.lines import Line2D
#from matplotlib.patches import Rectangle

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
stations = ['Cuxhaven']
'''

# Set analysis interval
timeframes = ( ['2012-09-01 00:00:00', '2012-09-30 12:48:00'],
               ['2013-06-01 00:00:00', '2013-06-30 12:48:00'])

# Choose station for stratification analysis
sstation = 'Cuxhaven'

levels = [1, -1] #16, 19 z-level (sigma level), ranges from 0 (bottom) to 19 (surface)

# Choose model data to use
exp = 'exp_2022-08-12' #experiment handle
conf = '182' #configuration

#==============================================================================

springtides = ( [np.datetime64('2012-09-16'), np.datetime64('2012-09-30')],
                [np.datetime64('2013-06-08'), np.datetime64('2013-06-23')] )

neaptides =   ( [np.datetime64('2012-09-08'), np.datetime64('2012-09-22')],
                [#np.datetime64('2013-05-31'), 
                 np.datetime64('2013-06-16'), np.datetime64('2013-06-30')] )



#==============================================================================

print('')
print('Moin!')

# initialise figure 1
fig1, axs1 = plt.subplots(2,1, figsize=(10,4), tight_layout=True,
                        sharey=True)
fig2, axs2 = plt.subplots(2,1, figsize=(10,4), tight_layout=True,
                        sharey=True)

#simple counter:
cc = 0

for tf in timeframes:
    
    start = tf[0]
    stop  = tf[1] #should usually cover 2 spring-neap cycles
    
    print('\n====================================================')
    print('Considering time span from ' + start + 
          ' to ' + stop + '\n')
    
    startdate = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    stopdate = datetime.strptime(stop, '%Y-%m-%d %H:%M:%S')
    date = startdate.strftime('%Y%m') + '01'
    month = startdate.strftime('%B')
    year = startdate.strftime('%Y')
    
    #Path to mean GETM output for TEF analysis and for model-km:
    #mean_base = '../store/' + exp + '/OUT_182/' + date + '/'  
    #mean_file_name = 'Mean_all.' + date + '.nc4'
    
    path = ('/silod6/reese/tools/getm/setups/elbe_realistic/store/'
                    + exp + '/OUT_' + conf + '/' + date + '/')
    
    directories = os.listdir(path)
    #remove gauges from list (similar station names):
    directories = [d for d in directories if ('SST' in d)]
    ii = np.where([(sstation in d) for d in directories])[0]
    directories = [directories[i] for i in ii]
    directories = [path + d for d in directories]
    time = xarray.open_mfdataset(directories)['time'][:].loc[start:stop]
    salt = xarray.open_mfdataset(directories)['salt'][:,:,0,0].loc[start:stop]
    z = xarray.open_mfdataset(directories)['depth'][:,:,0,0].loc[start:stop]
    
    fig, tax = plt.subplots()
    tax.plot(z[5,:])
    
    time_plt = np.tile( time, (len(z[0,:]), 1) ).T #make time same dimension as z

    
    ax = axs1[cc]
    ax.contourf(time_plt, z, salt)
    ax.set_xlim([startdate, stopdate])
    
    ax = axs2[cc]
    ax.plot(time, salt[:,0]-salt[:,-1])
    ax.set_ylim([-1, 10])
    ax.set_xlim([startdate, stopdate])
    springs = springtides[cc]
    neaps = neaptides[cc]
    miny, maxy = ax.get_ylim()
    ax.vlines(neaps, miny, maxy, linestyle='--', label='neap')
    ax.vlines(springs, miny, maxy, linestyle='-', label='spring')
    
    #if cc==0:
    ax.plot(time, -z[:,-1]+7.3, color='r', label='$-z + 7.3$')
    
    ax.legend(loc=2)
    ax.grid()
    
    cc += 1
    

fig1.savefig('plots/paper' + '/salt_tidal_all' + '.png', dpi=400)
fig2.savefig('plots/paper' + '/salt_tidal_strat' + '.pdf')
fig2.savefig('plots/paper' + '/salt_tidal_strat' + '.png', dpi=400)
plt.show()
