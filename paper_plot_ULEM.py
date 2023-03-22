# -*- coding: utf-8 -*-

"""
Created on Fri Dec 9 13:57:26 2022

@author: Nina Reese / nina.reese@io-warnemuende.de

**Data processing / analysis for GETM model output**

Mixing analysis in isohaline framework using code from
Burchard et al. (2021) [1].
Yields plots for temporal average over two manually defined time frames,
testing the Universal Law of Estuarine Mixing [2]

==> Figure 9 used in [3]

[1] Burchard, H., U. Gräwe, K. Klingbeil, N. Koganti, X. Lange, and M. Lorenz, 2021:
    Effective Diahaline Diffusivities in Estuaries. Journal of Advances in Modeling Earth
    Systems, 13 (2), doi:https://doi.org/10.1029/2020MS002307, url: https://agupubs.
    onlinelibrary.wiley.com/doi/abs/10.1029/2020MS002307.
[2] Burchard, H., 2020: A Universal Law of Estuarine Mixing.
    J. Phys. Oceanogr., 50 (1), 81–93. 
    DOI: https://doi.org/10.1175/JPO-D-19-0014.1
[3] N. Reese,U. Graewe, K. Klingbeil, X. Li, M. Lorenz, H. Burchard, 2023:
    Local mixing determines spatial structure of diahaline exchange flow in a
    mesotidal estuary – a study of extreme runoff conditions.
    J. Phys. Oceanogr., submitted.
    
LAST UPDATED:
    NR, 2023-03-10, 16:45h --- clean-up
    NR, 2023-01-30, 11:54h --- added tributaries to total runoff computation
    NR, 2023-01-06, 13:00h --- added m_xy plot 
"""


import datetime
import xarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredText
plt.rcParams.update({       #Use LaTeX for plotting
    "text.usetex": True})


#==============================================================================
# Manual Input

exp = 'exp_2022-08-12'

#timeframes for the plots - defines two averaging periods that will be
#compared in the plots: ([start1, stop1], [start2, stop2])
timeframes = ( ['2012-09-01 00:00:00', '2012-09-30 12:48:00'],
               ['2013-06-01 00:00:00', '2013-06-30 12:48:00'])

s = 13 #salinity class (g/kg) for horizontal map plots

#Path to river runoff file:
river_path = '../rivers.nc'

savefig = True #Figure will only be saved if True


#==============================================================================
#%%  FUNCTION DEFINITIONS
#==============================================================================

def cleanup_data(arr, bathy, lonx, latx, si=0):
    
    """
    Really just extracts given data for a desired salinity class index
    and sets data to NaN where no data should be available (land cells/missing cells)
    
    Parameters
    ----------
    arr : np array (np.float) of dim(salt, y, x) OR dim(y,x)
        local parameter per salinity class
    si : int
        Index of desired salinity class. Default is 0
    bathy : np array (np.float) of dim(y, x)
        Bathymetry. Only used for missing data
    lonx : np array (np.float) of dim(y, x)
        Longitude at grid vertices
    latx : np array (np.float) of dim(y, x)
        Latitude at grid vertices

    Returns
    -------
    arr : np array (np.float) of dim(y, x)
        local parameter per salinity class for salinity class
        at index si

    """
    
    if np.ndim(arr)==3:
        arr = arr[si,:,:] #data for salinity class at index si
    #removing all the cell centers where vertice or center values are missing:
    arr[np.isnan(bathy)] = np.nan
    arr[np.isnan(lonx[:-1,:-1])] = np.nan
    arr[np.isnan(lonx[1:,:-1])] = np.nan
    arr[np.isnan(lonx[:-1,1:])] = np.nan
    arr[np.isnan(lonx[1:,1:])] = np.nan
    
    return arr



#==============================================================================
#%%  START SCRIPT
#==============================================================================

print('')
print('Moin!')

# initialise figure 1
fig1, axs1 = plt.subplots(1,2, figsize=(6,3), tight_layout=True,
                        sharey=True)

#figure labels:
labels = ['(a)', '(b)']

#simple counter:
cc = 0

#==============================================================================
#%% Start time loop
#==============================================================================

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
    
    #Path to GETM output for mixing analysis:
    mixing_base = '../store/' + exp + '/OUT_182/' + date + '/' 
    mixing_file_name = 'Mixing_Mean_all.' + date + '.nc4'
    
    #Path to diahaline GETM output:
    tef_base = '../store/' + exp + '/OUT_182/' + date + '/'  
    tef_file_name = 'Elbe_TEF_mean_all.' + date + '.nc4'
    
    
    #==========================================================================
    #%%  LOAD DATA
    #==========================================================================
    
    #Import observed river runoff data (which was also used as forcing!):
    print('Loading river data...')

    # Read riverinfo to get names and x-indices   
    rivers = pd.read_table('../riverinfo.dat', skiprows=1, comment='!',
                delimiter=' ', header=None, usecols=[1,3], names=['etac', 'name'])
    # remove duplicates (a given runoff might have multiple entry points)            
    rivers = rivers.drop_duplicates()
    
    # Load runoff for each tributary, add to total runoff
    Q_r = 0
    for r_name in rivers['name']:
    	Q_r_ri = xarray.open_mfdataset(river_path)[r_name][:]
    	Q_r_ri = Q_r_ri.loc[start:stop]
    	Q_r_ri = np.nanmean(Q_r_ri) #temporal averaging
    	Q_r += Q_r_ri
    print('Average runoff at Cuxhaven: ' + str(Q_r) + ' m3/s')
    
    
    # =========================================================================
    # Then load data from model output. 
    
    print('Loading ' + mixing_file_name + ' data...')

    path = mixing_base + mixing_file_name #full path to file
    
    #time = xarray.open_mfdataset(path)['time'][:]
    dA = xarray.open_mfdataset(path)['areaC'][:]
    salt_s = xarray.open_mfdataset(path)['salt_s'][:]  #salinity classes
    hpmS_s = xarray.open_mfdataset(path)['hpmS_s_mean'][:]  #physical mixing
    hnmS_s = xarray.open_mfdataset(path)['hnmS_s_mean'][:]  #numerical mixing
    
    print('\tConverting arrays...')
    
    #convert from xarray to numpy array
    #time = np.asarray(time.loc[start:stop])
    dA = np.asarray(dA)
    hpmS_s = np.asarray(hpmS_s.loc[start:stop])
    hnmS_s = np.asarray(hnmS_s.loc[start:stop])
    salt_s = np.asarray(salt_s)
    delta_s = salt_s[1] - salt_s[0]
    
    #find index of salinity class for which mixing will be plotted:
    si = np.where(salt_s==s)[0][0] #index of salinity class
    

    
    #==========================================================================
    #%%  DO MIXING ANALYSIS
    #==========================================================================
    
    print('Starting mixing analysis...')
    
    # =========================================================================
    #%% Temporal averaging of Mixing variables
    # =========================================================================
    
    #Original properties were M2-averaged...
    #Now we average over the entire time span
    
    print('Temporal averaging...')
    hpmS_s_mean = np.nanmean(hpmS_s[:,:,:,:],axis=0) #Physical Mixing
    hnmS_s_mean = np.nanmean(hnmS_s[:,:,:,:],axis=0) #Numerical Mixing
    
    # =========================================================================
    #%% Integration along X and Y; Division by ds to get variables per salinity class
    # =========================================================================
    
    # This is for the Universal Law analysis
    
    print('Integration along X and Y...')
    
    mms_phy = np.zeros(len(salt_s))
    mms_num = np.zeros(len(salt_s))
    
    #sums are first over y (etac), then over x (xic), i.e., y-x-integration
    mms_phy = np.nansum(np.nansum(hpmS_s_mean*dA,axis=1),axis=1)/delta_s
    mms_num = np.nansum(np.nansum(hnmS_s_mean*dA,axis=1),axis=1)/delta_s
    
    mms_total = mms_phy+mms_num  #this is now the total mixing per salinity class
    
    
    # =========================================================================
    #%% Plot results
    # =========================================================================
    
    print('Plotting...')
    
    # Universal Law of Estuarine Mixing:
    
    ax = axs1[cc]
    
    ax.plot(salt_s, mms_total, '-k', label='$m(S)$')
    ax.plot(salt_s, mms_phy, linestyle='-', color='darkgrey', label='$m_{\mathrm{phy}}(S)$')
    ax.plot(salt_s, mms_num, linestyle='-', color='lightgrey', label='$m_{\mathrm{num}}(S)$')
    ax.plot(salt_s, 2*salt_s*Q_r, '--k', alpha=0.5, label='$2SQ_{\mathrm{r}}$')
    ax.set_xlabel('$S$ (g/kg)')
    ax.set_xlim([0, 32])
    ax.set_ylim([-8000, 170000])
    ax.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True)
    ax.set_title(labels[cc] + ' ' + month + ' ' + year)
    
    if cc==0:
        ax.legend(loc=2, borderaxespad=1)
        ax.set_ylabel('$m(S)$ (m$^3$(g/kg)s$^{-1}$)')
    
    
    cc += 1
    
#==============================================================================    
#%% LOOP ENDS
#==============================================================================

# Add grey-shaded area marking isohalines outside model domain:
bdries = [22.5, 18]
txts = ['Isohalines\npartially\noutside\nmodel\ndomain',
        'Isohalines\npartially\noutside\nmodel domain']
ii = 0
for ax in axs1:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    rect = Rectangle((bdries[ii], ymin), width=(xmax-bdries[ii]),
                     height=(ymax-ymin), 
                     edgecolor=None, facecolor='k', alpha=0.1, zorder=-3)
    ax.add_artist(rect)
    
    anchored_text = AnchoredText(txts[ii],
                                 loc=1, frameon=False,
                                 prop=dict(color='k', zorder=1000))
    ax.add_artist(anchored_text)
    
    ii += 1


if savefig:
    fig1.savefig('plots/paper/isohaline_mixing_ULEM.pdf')
    fig1.savefig('plots/paper/isohaline_mixing_ULEM.png', dpi=300)
plt.show()

print('\nDone!')

