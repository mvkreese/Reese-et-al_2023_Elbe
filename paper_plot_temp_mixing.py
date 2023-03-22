#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  24 14:35 2021

@author: Nina Reese / nina.reese@io-warnemuende.de

**Data processing / analysis for GETM model output**

Temporal mixing analysis in isohaline framework
Yields
    (a) line plot for mixing at a given salinity class vs. time
    (b) mixing per salinity class m(S,t) vs. time t and salinity class S
    (c) mixing normalised with the universal law of estuarine mixing [1], i.e.,
        m(S,t) / 2SQ_r, with the freshwater runoff Q_r and salinity class S

Script yields results presented in Fig. 10 in [2]

[1] Burchard, H., 2020: A Universal Law of Estuarine Mixing.
    J. Phys. Oceanogr., 50 (1), 81–93. 
    DOI: https://doi.org/10.1175/JPO-D-19-0014.1
[2] N. Reese,U. Graewe, K. Klingbeil, X. Li, M. Lorenz, H. Burchard, 2023:
    Local mixing determines spatial structure of diahaline exchange flow in a
    mesotidal estuary – a study of extreme runoff conditions.
    J. Phys. Oceanogr., submitted.

"""


import xarray
import numpy as np
import pandas as pd
#import datetime
from scipy.interpolate import griddata
from scipy import signal
from datetime import date as dt
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm #DivergingNorm
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredText
plt.rcParams.update({       #Use LaTeX for plotting
    "text.usetex": True})

# We expect to have mean of empty slices in here, so let's ignore that warning:
import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')


#==============================================================================
# Manual Input

# Choice of day is irrelevant. Month in stopdate is last month to be loaded.
startdate = '2012-08-01 00:00:00'
stopdate  = '2013-12-31 23:59:00' #'2014-01-01 00:00:00'

S_class = 13 #salinity classes used for line plot

exp = 'exp_2022-08-12'

savefig = True #Figure will only be saved if True


#==============================================================================
#%% FUNCTION DEFINITIONS
#==============================================================================

def secsinceepoch(dates, epoch='2012-02-01 00:00:00'):
    
    """
    Computes seconds passed since epoch.
    INPUT: dates [xarray.DataArray, dtype=datetime64[ns]] dimension (N) 
                --> contains the dates to be converted to seconds since epoch
            epoch [str of format YYYY-mm-dd HH:MM:SS] scalar
                --> date at which elapsed seconds are set to 0
                --> default is 2012-02-01 00:00:00
    OUTPUT: timestamp [np array, dtype=float] dimension (N)
                --> dates converted into seconds since epoch
    """
    
    epoch = np.datetime64(epoch)
    timestamp = (dates - epoch).astype('timedelta64[s]')
    return np.asarray(timestamp, dtype=float)

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
    
    return datedirs



#==============================================================================
#%% RUN PREPARATIONS
#==============================================================================

print('\n' + 'Moin!\n' + 'Starting script...')
print('')

# Create salt bins:   (NOTE: better read them from file at some point...)
salt_s = np.linspace(0,35,176) #salt bins
ds =(salt_s[1]-salt_s[0]) #width of each salt bin
si = np.where(salt_s == S_class)[0][0] #find index of salinity class for line plot

colors = pl.cm.bone(np.linspace(0.2,.9,2)) #Colours used for line plots
custom_lines = [Line2D([0], [0], color=colors[0], lw=2),
                Line2D([0], [0], color=colors[1], lw=2),
                Line2D([0], [0], color='orange', lw=2),
                Line2D([0], [0], color='magenta', lw=1, linestyle='--')] #will be used for plot legend
custom_labels = ['$ m = (m_{\mathrm{phy}} + m_{\mathrm{num}})$',
                 '$m_{\mathrm{num}}$', '$Q_{\mathrm{r}}$', 'Spring tides'] #will be used for plot legend


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
	Q_r_ri = Q_r_ri.loc[startdate:stopdate]
	Q_r += Q_r_ri

Q_r_time = Q_r.time
Q_r_time_sse = secsinceepoch(Q_r_time) #converting to seconds since epoch for interpolation later on
Q_r = Q_r.data


#==============================================================================
#%% SPRING- AND NEAPTIDE DATES
#==============================================================================

springtides = [np.datetime64('2012-07-03'), np.datetime64('2012-07-19'),
               np.datetime64('2012-08-02'), np.datetime64('2012-08-17'),
               np.datetime64('2012-08-31'),
               np.datetime64('2012-09-16'), np.datetime64('2012-09-30'),
               np.datetime64('2012-10-15'), np.datetime64('2012-10-29'),
               np.datetime64('2012-11-13'), np.datetime64('2012-11-28'),
               np.datetime64('2012-12-13'), np.datetime64('2012-12-28'),
               np.datetime64('2013-01-11'), np.datetime64('2013-01-27'),
               np.datetime64('2013-02-10'), np.datetime64('2013-02-25'),
               np.datetime64('2013-03-11'), np.datetime64('2013-03-27'),
               np.datetime64('2013-04-10'), np.datetime64('2013-04-25'),
               np.datetime64('2013-05-10'), np.datetime64('2013-05-25'),
               np.datetime64('2013-06-08'), np.datetime64('2013-06-23'),
               np.datetime64('2013-07-08'), np.datetime64('2013-07-22'),
               np.datetime64('2013-08-06'), np.datetime64('2013-08-21'),
               np.datetime64('2013-09-05'), np.datetime64('2013-09-19'),
               np.datetime64('2013-10-05'), np.datetime64('2013-10-18'),
               np.datetime64('2013-11-03'), np.datetime64('2013-11-17'),
               np.datetime64('2013-12-03'), np.datetime64('2013-12-17')]

neaptides =   [np.datetime64('2012-07-11'), np.datetime64('2012-07-26'),
               np.datetime64('2012-08-09'), np.datetime64('2012-08-24'),
               np.datetime64('2012-09-08'), np.datetime64('2012-09-22'),
               np.datetime64('2012-10-08'), np.datetime64('2012-10-22'),
               np.datetime64('2012-11-07'), np.datetime64('2012-11-20'),
               np.datetime64('2012-12-06'), np.datetime64('2012-12-20'),
               np.datetime64('2013-01-05'), np.datetime64('2013-01-18'),
               np.datetime64('2013-02-03'), np.datetime64('2013-02-17'),
               np.datetime64('2013-03-04'), np.datetime64('2013-03-19'),
               np.datetime64('2013-04-03'), np.datetime64('2013-04-18'),
               np.datetime64('2013-05-02'), np.datetime64('2013-05-18'),
               np.datetime64('2013-05-31'), np.datetime64('2013-06-16'),
               np.datetime64('2013-06-30'), np.datetime64('2013-07-16'),
               np.datetime64('2013-07-29'), np.datetime64('2013-08-14'),
               np.datetime64('2013-08-28'), np.datetime64('2013-09-12'),
               np.datetime64('2013-09-27'), np.datetime64('2013-10-11'),
               np.datetime64('2013-10-26'), np.datetime64('2013-11-10'),
               np.datetime64('2013-11-25'), np.datetime64('2013-12-09'),
               np.datetime64('2013-12-25')]
               

#==============================================================================
#%% MAIN
#==============================================================================

# Preparing plot
fig, ax = plt.subplots(3,1, figsize=(7,6), constrained_layout=True,
                       sharex=False)
ax_s=ax[0].twinx() #second y-axis for topmost plot to plot Q_r and m(t,S=S_class)

minx = 0 #lower xlim of plot x-axis
maxx = 0 #upper xlim of plot x-axis

#initialising some stuff
hpmS_s_iim1 = []
hnmS_s_iim1 = []
salt_iim1 = []
time_iim1 = []
salt_time_iim1 = []

#finding all date directories to loop through:
datedirs = find_datedirs(startdate, stopdate)

#finding startdate, stopdate and 1st date after startdate:
dateints = np.array([int(dd) for dd in datedirs])
start = min(dateints)
startp1 = min(dateints[dateints>start])
stop = max(dateints)

for date in datedirs: #Loop through all months

	year = str(date[:4])
	month = str(date[4:6]).zfill(2)
	print('')
	print('==============================\n')
	print('MONTH: ' + month + ' ' + year)

	#Path to GETM output in 'Mean' file:
	mean_base = '../store/' + exp + '/OUT_182/' + date + '/' 
	mean_file_name = 'Mean_all.' + date + '.nc4'

	#Path to GETM output in 'TEF' file:
	mixing_base = '../store/' + exp + '/OUT_182/' + date + '/'
	mixing_file_name = 'Mixing_Mean_all.' + date + '.nc4'


	#==============================================================================
	#  LOAD DATA
	#==============================================================================
	
	# Load mixing stuff from model output. First: Mean_all

	path = mean_base + mean_file_name #full path to file

	print('Loading ' + mean_file_name + ' data...')
	salt_time_ii = xarray.open_mfdataset(path)['time'][:] #only used for min S still inside model domain
	salt_ii = xarray.open_mfdataset(path)['salt_mean'][:] #only used for min S still inside model domain
	dA = xarray.open_mfdataset(path)['areaC'][:]

	print('Converting arrays...')
	dA = np.asarray(dA)
	dA[np.isnan(dA)] = 0


	# Next: load mixing data
	path = mixing_base + mixing_file_name #full path to file
	print('Loading ' + mixing_file_name + ' data...')
	time_ii = xarray.open_mfdataset(path)['time'][:]
	if date==str(start):
		minx = np.nanmin(time_ii) #lower xaxis limit
	hpmS_s_ii = xarray.open_mfdataset(path)['hpmS_s_mean'][:]  #physical mixing
	hnmS_s_ii = xarray.open_mfdataset(path)['hnmS_s_mean'][:]  #numerical mixing


	if date!=str(start):
		print('Concatenating arrays...')

		hpmS_s = xarray.concat([hpmS_s_iim1, hpmS_s_ii], dim='time', data_vars='all')
		hnmS_s = xarray.concat([hnmS_s_iim1, hnmS_s_ii], dim='time', data_vars='all')
		salt = xarray.concat([salt_iim1, salt_ii], dim='time', data_vars='all')
		time = xarray.concat([time_iim1, time_ii], dim='time', data_vars='all')
		salt_time = xarray.concat([salt_time_iim1, salt_time_ii], dim='time', data_vars='all')
                
		#convert from xarray to numpy array
		print('Converting more arrays...')
		hpmS_s = hpmS_s.data
		hnmS_s = hnmS_s.data
		time = time.time
		salt_time = salt_time.time
		time_sse = secsinceepoch(time) #converting to seconds since epoch for interpolation later on

		print('Interpolating river runoff...')
		Q_r_ii = griddata(Q_r_time_sse, Q_r, time_sse, method='linear')


		# =============================================================================
		# Integration along X and Y; Division by ds to get variables per salinity class
		# =============================================================================
		
		#Do integration over full horizontal model domain
		print('Integration along x and y...')
		mms_phy_lp = np.sum(np.sum(hpmS_s*dA,axis=2),axis=2)/ds
		mms_num_lp = np.sum(np.sum(hnmS_s*dA,axis=2),axis=2)/ds

		print('Converting more arrays...')
		mms_num_lp = np.asarray(mms_num_lp)
		mms_phy_lp = np.asarray(mms_phy_lp)

		# Initialising mixing normalised with universal law
		m_num_norm = np.zeros(np.shape(mms_num_lp))
		m_phy_norm = np.zeros(np.shape(mms_phy_lp))

		print('Finding min. S not fully inside model domain...')    
		i = 3 #We have a sponge layer of 3 grid lines
		test_salt = np.copy(salt)
		test_salt[test_salt<0.5] = 0
		s_min = np.nanmin(test_salt[:,:,100:250,i], axis=1) #minimum salinity found in open bdry grid cells
		s_min = np.nanmin(s_min[:,:], axis=1) #minimum salinity found in open bdry grid cells
		#next smallest salinity should then be fully contained inside model domain!

		#remove tidal variation with a low-pass filter:
		print('Low-pass filtering...')
		for ss in range(len(salt_s)-1):
		    order = 6 #order of the butterworth filter
		    Tc = 33 #cutoff period of filter (h) #35
		    cutoff = 1/(3600*Tc) #cutoff frequency of the filter (1/s)
		    fs = 1/(12*3600+25*60) #3600 #sampling frequency of the data (1/s)
		    nyq = fs/2 #Nyquist frequency
		    normal_cutoff = cutoff/nyq #normalised cutoff frequency
		    b, a = signal.butter(order, normal_cutoff,  #33 h low-pass
				          btype='low', analog=False)
		    mms_phy_lp[:,ss] = signal.filtfilt(b, a, mms_phy_lp[:,ss]) #filtfilt
		    mms_num_lp[:,ss] = signal.filtfilt(b, a, mms_num_lp[:,ss]) #filtfilt
            
            #Mixing normalised with universal law:
		    m_phy_norm[:,ss] = mms_phy_lp[:,ss] / (Q_r_ii*salt_s[ss]*2)
		    m_num_norm[:,ss] = mms_num_lp[:,ss] / (Q_r_ii*salt_s[ss]*2)

		    fs = 1/3600 #sampling frequency of the data (1/s)
		    nyq = fs/2 #Nyquist frequency
		    normal_cutoff = cutoff/nyq #normalised cutoff frequency
		    b, a = signal.butter(order, normal_cutoff,  #33 h low-pass
				          btype='low', analog=False)
		    s_min[:] = signal.filtfilt(b, a, s_min) #filtfilt

		mms_tot_lp = mms_phy_lp + mms_num_lp #total mixing
		m_norm_tot = m_phy_norm + m_num_norm #total normalised mixing
        
        # List for plotting:
		mms = [mms_tot_lp, mms_num_lp, mms_phy_lp]

		# =============================================================================
		# Plotting
		# =============================================================================		

		#plt.plot(time, mms_tot_lp[:,50])
		print('Preparing plot additions...')
		if date==str(startp1):
			sta = int(0)
			sto = int(np.round(len(time_ii.time)/2))-1
			ssto = int(np.round(len(salt_time_ii.time)/2))-1
            
			#This here is a line plot of Q_r as well as mixing at certain salinity vs. time:
			ax[0].plot(time[sta:-sto], Q_r_ii[sta:-sto], color='orange')
			for ii in range(2):
				ax_s.plot(time[sta:-sto], mms[ii][sta:-sto,si],
					color=colors[ii])

			#And here we create a heatmap plot of mixing vs. time and salinity
			S, t = np.meshgrid(salt_s, time[sta:-sto])
			ax[1].plot(salt_time[sta:-ssto], s_min[sta:-ssto], linestyle='-',
                    color='white', linewidth=0.8, label='$S_{lim}$')
			cax = ax[1].contourf(t, S, mms_tot_lp[sta:-sto,:],
				levels=np.linspace(0,150000,31), cmap='magma_r', extend='both',
                zorder=-2)
            
            #Here: Heatmap mixing normalised with universal law vs. time, S
			ax[2].plot(salt_time[sta:-ssto], s_min[sta:-ssto], linestyle='-',
                    color='white', linewidth=0.8, label='$S_{lim}$')
			cax2 = ax[2].contourf(t, S, m_norm_tot[sta:-sto,:],
				levels=np.linspace(0,3.5,71), cmap='twilight_shifted', 
                norm=TwoSlopeNorm(1), extend='both', zorder=-2)
    

		elif date==str(stop):
			sta = int(np.round(len(time_iim1.time)/2))-1
			ssta = int(np.round(len(salt_time_iim1.time)/2))-1
            
			#This here is a line plot of Q_r as well as mixing at certain salinity vs. time:
			ax[0].plot(time[sta:], Q_r_ii[sta:], color='orange')
			for ii in range(2):
				ax_s.plot(time[sta:], mms[ii][sta:,si],
					color=colors[ii])

			#And here we create a heatmap plot of mixing vs. time and salinity
			S, t = np.meshgrid(salt_s, time[sta:])
			ax[1].plot(salt_time[ssta:], s_min[ssta:], linestyle='-', color='white',
					linewidth=0.8, label='$S_{lim}$')
			cax = ax[1].contourf(t, S, mms_tot_lp[sta:,:],
				levels=np.linspace(0,150000,31), cmap='magma_r', extend='both',
                zorder=-2)

            #Here: Heatmap mixing normalised with universal law vs. time, S
			ax[2].plot(salt_time[ssta:], s_min[ssta:], linestyle='-', color='white',
					linewidth=0.8, label='$S_{lim}$')
			cax2 = ax[2].contourf(t, S, m_norm_tot[sta:,:],
				levels=np.linspace(0,3.5,71), cmap='twilight_shifted', 
                norm=TwoSlopeNorm(1), extend='both', zorder=-2)


		else:
			sta = int(np.round(len(time_iim1.time)/2))-1
			sto = int(np.round(len(time_ii.time)/2))-1
			ssta = int(np.round(len(salt_time_iim1.time)/2))-1
			ssto = int(np.round(len(salt_time_ii.time)/2))-1
            
			#This here is a line plot of Q_r as well as mixing at certain salinity vs. time:
			ax[0].plot(time[sta:-sto], Q_r_ii[sta:-sto], color='orange')
			for ii in range(2):
				ax_s.plot(time[sta:-sto], mms[ii][sta:-sto,si],
					color=colors[ii])

			#And here we create a heatmap plot of mixing vs. time and salinity
			S, t = np.meshgrid(salt_s, time[sta:-sto])
			ax[1].plot(salt_time[ssta:-ssto], s_min[ssta:-ssto], linestyle='-',
                    color='white', linewidth=0.8, label='$S_{lim}$')
			cax = ax[1].contourf(t, S, mms_tot_lp[sta:-sto,:],
				levels=np.linspace(0,150000,31), cmap='magma_r', extend='both',
                zorder=-2)

            #Here: Heatmap mixing normalised with universal law vs. time, S
			ax[2].plot(salt_time[ssta:-ssto], s_min[ssta:-ssto], linestyle='-',
                    color='white', linewidth=0.8, label='$S_{lim}$')
			cax2 = ax[2].contourf(t, S, m_norm_tot[sta:-sto,:],
				levels=np.linspace(0,3.5,71), cmap='twilight_shifted', 
                norm=TwoSlopeNorm(1), extend='both', zorder=-2)


	print('Preparing next month...')
	hpmS_s_iim1 = hpmS_s_ii
	hnmS_s_iim1 = hnmS_s_ii
	salt_iim1 = salt_ii
	time_iim1 = time_ii
	salt_time_iim1 = salt_time_ii


#==============================================================================
#%% END OF LOOP, Finishing plot
#==============================================================================

print('\n==============================\n')

print('Finishing plot...')
maxx=max(time) #upper xaxis limit

#Marking dates of springtides with vertical lines:
for s in springtides:
    ax_s.vlines(s,-10000,120000, color='magenta', linestyle='--', linewidth=0.5,
                alpha=0.8, zorder=-100)
    ax[1].vlines(s,0,35, color='magenta', linestyle='--', linewidth=0.5)
    ax[2].vlines(s,0,35, color='magenta', linestyle='--', linewidth=0.5)


ax[0].set_ylabel('$Q_{\mathrm{r}}$ (m$^3$s$^{-1}$)')
ax[0].set_xlim([minx, maxx])
ax[0].set_ylim([-10000/(2*S_class), 120000/(2*S_class)])
ax[0].set_xticklabels([])
ax[0].tick_params('both', which='both', direction='in', bottom=True, top=True,
                       left=True, right=False, labelbottom=False)

ax_s.legend(custom_lines, custom_labels, facecolor='white', ncol=1, loc=2,
            borderaxespad=1)
ax_s.set_ylabel('$m(S={s},t)$ (m$^3$(g/kg)s$^{{-1}}$)'.format(s=S_class))
ax_s.tick_params('both', which='both', direction='in', bottom=False, top=False,
                       left=False, right=True, labelbottom=False)
ax_s.set_ylim([-10000,120000])
# Add invisible colorbar to keep axis plot length consistent with the heatmaps:
cb = plt.colorbar(cax, ax=ax[0], location="right", shrink=0.8)
cb.remove()

ax[1].set_ylabel('$S$ (g/kg)')
ax[1].tick_params('both', which='both', direction='in', bottom=True, top=True,
                       left=True, right=True, labelbottom=False)
ax[1].set_ylim([0,35])
ax[1].set_xlim([minx, maxx])
ax[1].legend([Line2D([0], [0], color='white', lw=1)], ['$S_{\mathrm{lim}}$'],
             facecolor='grey', loc=2)
cbar = plt.colorbar(cax, ax=ax[1], location="right", shrink=0.8)#, extend='max')
cbar.set_label('$m(S,t)$ (m$^3$(g/kg)s$^{-1}$)')

ax[2].set_ylabel('$S$ (g/kg)')
ax[2].tick_params('both', which='both', direction='in', bottom=True, top=True,
                       left=True, right=True, labelleft=True, labelbottom=True)
ax[2].set_ylim([0,35])
ax[2].set_xlim([minx, maxx])
ax[2].legend([Line2D([0], [0], color='white', lw=1)], ['$S_{\mathrm{lim}}$'],
             facecolor='grey', loc=2)
cbar = plt.colorbar(cax2, ax=ax[2], location="right", shrink=0.8)
cbar.set_label('$m(S,t) / 2SQ_{\mathrm{r}}$ (-)')


# Adding lettering for each subfigure:
cc = 0
letters = ['(a)', '(b)', '(c)']
lettercolors = ['k', 'k', 'w']
for axs in ax.flatten():
    anchored_text = AnchoredText( letters[cc], loc=1, frameon=False,
                                 prop=dict(color=lettercolors[cc], zorder=1000,
                                           fontweight='bold'))
    axs.add_artist(anchored_text)
    cc += 1

plt.xticks(ha='left')


# Marking detail in first subplot:
ymin = 150
ymax = 1800
sta = dt(2013, 7, 12)
sto = dt(2013, 8, 5)
rect = Rectangle((sta, ymin), width=(sto-sta), height=(ymax-ymin), 
                 edgecolor='k', facecolor='none', alpha=1, zorder=10)
ax[0].add_artist(rect)

ii += 1



if savefig:
    fig.savefig('plots/paper/Temp_TEF_mixing.png', dpi=600)
plt.show()


print('\n' + 'Done!' + '\n' + 'May the Force be with you!')

