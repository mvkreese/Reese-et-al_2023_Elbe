# -*- coding: utf-8 -*-

"""
Created on Tue Feb 21 15:09:26 2023

@author: Nina Reese / nina.reese@io-warnemuende.de

**Data processing / analysis for GETM model output**

Horizontal mixing and diahaline analysis.
Yields plots for temporal average over two manually defined time frames
    (a,b) showing the horizontal distribution of local mixing m_xy
    (c,d) showing the effective vertical diahaline velocity, as computed
          from the S-gradient of m_xy [1]
    (e,f) showing the effective vertical diahaline velocity, as computed
          online with GETM

==> Figure 11 used in [2] 

[1] Klingbeil, K., and E. Henell, 2023: A rigorous derivation of the water
    mass transformation framework, the relation between mixing and dia-surface
    exchange flow, and links to recent theories in estuarine research.
    J. Phys. Oceanogr., submitted.
[2] N. Reese,U. Graewe, K. Klingbeil, X. Li, M. Lorenz, H. Burchard, 2023:
    Local mixing determines spatial structure of diahaline exchange flow in a
    mesotidal estuary – a study of extreme runoff conditions.
    J. Phys. Oceanogr., submitted.

    
LAST UPDATED:
    NR, 2023-03-20, 17:46h --- Documentation, adding open bdry and land
"""


import datetime
import xarray
import netCDF4
import shapefile
import numpy as np
from scipy.stats import pearsonr
from pyproj import Transformer

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from matplotlib import colors as mpl_c
from matplotlib.colors import TwoSlopeNorm #DivergingNorm
from matplotlib.patches import Rectangle, Polygon
from matplotlib.offsetbox import AnchoredText
from matplotlib.collections import PatchCollection
plt.rcParams.update({       #Use LaTeX for plotting
    "text.usetex": True})


#==============================================================================
# Manual Input

exp = 'exp_2022-08-12'

#timeframes for the plots - defines two averaging periods that will be
#compared in the plots: ([start1, stop1], [start2, stop2])
timeframes = ( ['2012-09-01 00:00:00', '2012-09-30 12:48:00'],
               ['2013-06-01 00:00:00', '2013-06-30 12:48:00'])

# # path to the shapefiles containing the coastline and land:
# path = "coastline/"
# coast_name = "Coastline_WaddenSea" #coastline
# land_name  = "shoreline_northsea"  #closed land polygons
# sf_coast = shapefile.Reader(path + coast_name)
# sf_land  = shapefile.Reader(path + land_name)

s = 13 #salinity class (g/kg) for horizontal map plots

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


def add_land(sf, color, edgecolor='None'):
    
    '''
    Creates PatchCollection of land polygons from shapefile. Polygons are
    filled with a given color.
    
    INPUT:
        sf: [scalar, dtype=str]
            Path to shapefile
        color: [scalar] str or other color format compatible with matplotlib
            Facecolor to use for the land polygons
        edgecolor: [scalar] str or other color format compatible with matplotlib
            Edgecolor to use for the land polygons. Default is 'None'
    OUTPUT:
        pc: [matplotlib PatchCollection]
            Collection of all closed land polygons from shapefile
    '''
    
    #Prepare a data coordinate transformer from utm32 to WGS84:
    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    #Initialise patches for land
    patches = []
    
    #Add coastline from shapefile:
    for shape in list(sf.iterShapes()): #pull all shapes out of shapefile
    
        #x- and y-coordinates
        x_lon_sh = np.zeros((len(shape.points)))
        y_lat_sh = np.zeros((len(shape.points)))
        
        #loop through points:
        for ip in range(len(shape.points)):
            # transform coordinates
            xy = transformer.transform(shape.points[ip][0], shape.points[ip][1])
            # add points
            x_lon_sh[ip] = xy[0]
            y_lat_sh[ip] = xy[1]
            
        #Prepare polygon points for patches
        points = np.vstack((x_lon_sh, y_lat_sh)).T
        #Append land polygon to the list of patches
        patches.append( Polygon(points, closed=True, color=color) )
      
    # Add patches to axis
    pc = PatchCollection(patches, match_original=True, edgecolor=edgecolor,
                         linewidths=1., zorder=-2)
    return pc


#==============================================================================
#%%  START SCRIPT
#==============================================================================

print('')
print('Moin!')

# initialise figure
fig = plt.figure(figsize=(6,7.3))

gs0 = gridspec.GridSpec(3, 2, figure=fig, wspace=0.03, hspace=0, top=0.96,
                        bottom=0.03, left=0.1, right=0.99)

ax1 = fig.add_subplot(gs0[0, 0]) #top left
ax2 = fig.add_subplot(gs0[0, 1]) #top right
ax3 = fig.add_subplot(gs0[1, 0]) #center left
ax4 = fig.add_subplot(gs0[1, 1]) #center right
ax5 = fig.add_subplot(gs0[2, 0]) #bottom left
ax6 = fig.add_subplot(gs0[2, 1]) #bottom right

axs = [ax1, ax2, ax3, ax4, ax5, ax6]

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
    
    print('Loading ' + mixing_file_name + ' data...')

    path = mixing_base + mixing_file_name #full path to file
    
    dA = xarray.open_mfdataset(path)['areaC'][:]
    salt_s = xarray.open_mfdataset(path)['salt_s'][:]  #salinity classes
    hpmS_s = xarray.open_mfdataset(path)['hpmS_s_mean'][:]  #physical mixing
    hnmS_s = xarray.open_mfdataset(path)['hnmS_s_mean'][:]  #numerical mixing
    
    print('\tConverting arrays...')
    
    #convert from xarray to numpy array
    dA = np.asarray(dA)
    hpmS_s = np.asarray(hpmS_s.loc[start:stop])
    hnmS_s = np.asarray(hnmS_s.loc[start:stop])
    salt_s = np.asarray(salt_s)
    delta_s = salt_s[1] - salt_s[0]
    
    #find index of salinity class for which mixing will be plotted:
    si = np.where(salt_s==s)[0][0] #index of salinity class
    
    
    # =========================================================================
    
    print('Loading ' + tef_file_name + ' data...')
    path = tef_base + tef_file_name #full path to file
    h_s = xarray.open_mfdataset(path)['h_s_mean'][:,:,:,:] #might need this later
    print('\tConverting arrays...')
    h_s = np.asarray(h_s.loc[start:stop])

    
    # =========================================================================
    # Computing isohaline depths
    # =========================================================================    
        
    # Find average isohaline depth z(S):
    print('Computing average isohaline depth...')
    z_mean = np.cumsum(h_s[:,::-1,:,:], axis=1)[:,::-1,:,:]
    h_s = None; del h_s #clear memory
    h_mean = np.abs(np.nanmin(z_mean[:,:,:,:], axis=1)
                    - np.nanmax(z_mean[:,:,:,:], axis=1)) #total depth
    z_mean = z_mean[:,si,:,:] #depth of given isohaline s
    rel_z_mean = np.nanmean(z_mean/h_mean, axis=0) #normalised, time-averaged isohaline depth
    z_mean = None; del z_mean
    h_mean = None; del h_mean
    
    
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
    
    
    # =============================================================================
    #%% No Integration (variable distribution in [S,y,x])
    # =============================================================================
    
    # And this is for the horizontal mixing distribution maps!
    
    print('Computing local mixing...')
    mms_phy_x_y = hpmS_s_mean*dA/delta_s
    mms_num_x_y = hnmS_s_mean*dA/delta_s
    
    mms_total_x_y = mms_phy_x_y + mms_num_x_y
        
        
    #==============================================================================    
    #%% Diahaline velocity stuff:
    #==============================================================================
    
    #Path to direct diahaline GETM output:
    dia_inst_base = '../store/' + exp + '/OUT_182/' + date + '/'  
    dia_inst_file_name = 'Elbe_dia_getm_all.' + date + '.nc4'
    
    #==============================================================================
    #%%  Find GETM version for w_dia
    #==============================================================================

    print('Loading ' + dia_inst_file_name + ' data...')
    path = dia_inst_base + dia_inst_file_name #full path to file

    print('\t Finding index of salinity class for which mixing will be plotted...')
    salt_s = xarray.open_mfdataset(path)['salt_s'][:]  #salinity classes
    salt_s = np.asarray(salt_s)
    si = np.where(salt_s==s)[0][0] #index of salinity class

    dA = xarray.open_mfdataset(path)['areaC'][:]
    w_dia_getm = xarray.open_mfdataset(path)['wdia_s_mean'][:,si,:,:] #[time, salt_s, etac, xic]

    print('\t Converting arrays...')
    dA = np.asarray(dA)
    w_dia_getm = np.asarray(w_dia_getm.loc[start:stop])

    print('\t Temporal averaging...')
    w_dia_getm = np.nanmean(w_dia_getm, axis=0)
   
    
    #==============================================================================
    #%% Find Klingbeil et al. (2023) version for u_dia:
    #==============================================================================

    dA[dA==0] = np.nan #set area elements that are 0 to NaN to avoid dividing by 0

    print('Computing w_dia from Klingbeil et al. (in prep.)...')

    # Derivative with central differences:
    w_dia_phy = (mms_phy_x_y[2:,:,:] - mms_phy_x_y[:-2,:,:]) / (dA*2*delta_s) / 2
    w_dia_num = (mms_num_x_y[2:,:,:] - mms_num_x_y[:-2,:,:]) / (dA*2*delta_s) / 2
    w_dia_kling = w_dia_phy + w_dia_num

    mms_phy_x_y = None; del mms_phy_x_y
    mms_num_x_y = None; del mms_num_x_y
    w_dia_phy = None; del w_dia_phy
    w_dia_num = None; del w_dia_num
    
    
    #==============================================================================
    #%%  PLOTTING
    #==============================================================================

    print('Loading data for plot...')
    fp='/silod6/reese/tools/getm/setups/elbe_realistic/topo_smoothed_v20.nc4'
    ncfile = netCDF4.Dataset(fp)
    lonx = ncfile['lonx'][:] #longitude of grid vertices
    latx = ncfile['latx'][:] #latitude of grid vertices
    bathy = ncfile['bathymetry'][:] #bathymetry... just used as mask

    #Path to mean GETM output for TEF analysis and for model-km:
    mean_base = '../store/' + exp + '/OUT_182/' + date + '/'  
    mean_file_name = 'Mean_all.' + date + '.nc4'
    fp = mean_base + mean_file_name
    ncfile = netCDF4.Dataset(fp)
    lonc = ncfile['lonc'][:] #longitude of cell centers
    latc = ncfile['latc'][:] #latitude of cell centers

    # prepare output for plot
    print('Preparing plot...')
    mms_total_x_y[mms_total_x_y==0] = np.nan
    mms_total_x_y = cleanup_data(mms_total_x_y, bathy, lonx, latx, si=si)
    
    # w_dia_kling starts at s-class corresponding to s-index 1 instead of 0
    # so we need to use si-1 here
    # (note that it also ends at -2 instead of -1!)
    w_dia_kling = cleanup_data(w_dia_kling, bathy, lonx, latx, si=si-1)
    w_dia_getm = cleanup_data(w_dia_getm, bathy, lonx, latx, si=si)
    bathy = cleanup_data(bathy, bathy, lonx, latx, si=si)
      
    lonx[np.isnan(lonx)] = 12
    latx[np.isnan(latx)] = 53
    lonc[np.isnan(lonc)] = 12
    latc[np.isnan(latc)] = 53
    
    
    # =========================================================================
    # Local mixing m_xy:
    print('Plotting horizontal mixing distribution...')
        
    levels_s=[0.2, 0.5, 0.8]
    colors = pl.cm.viridis(np.linspace(0.2,0.5,3)) #Colours used for line plots

    mix_ax = axs[cc]
    cax = mix_ax.pcolormesh(lonx, latx, mms_total_x_y, cmap='magma_r',
                        vmin=0, vmax=40, zorder=1)
    CS = mix_ax.contour(lonc, latc, rel_z_mean, levels=levels_s, colors='k',
                        zorder=3)
    CC = mix_ax.contour(lonc, latc, bathy, levels=[12], colors=['darkgrey'],
                    linewidths=0.8, zorder=2)
    
    mix_ax.set_xlim([8.35, 9.3])
    mix_ax.set_ylim([53.7, 54.2])
    mix_ax.clabel(CS, inline=True, fontsize=8, fmt='%1.1f')
    mix_ax.set_aspect("1.8")
    mix_ax.set_title(month + ' ' + year)

    if cc==0:
        mix_ax.set_ylabel('lat (°N)')
        mix_ax.tick_params('both', which='both', direction='in', bottom=True,
                           top=True, left=True, right=True, labelbottom=False,
                           labeltop=True)
    if cc==1:
        mix_ax.tick_params('both', which='both', direction='in', bottom=True,
                           top=True, left=True, right=True, labelleft=False,
                           labelbottom=False, labeltop=True)
        cbar = fig.colorbar(cax, ax=axs[:2], orientation='vertical',
                            shrink=0.7, extend='max')
        cbar.set_label('$m_{\mathrm{xy}}(S = $ '
                       + str(s) + '$,x,y)$\n(m(g/kg)s$^{-1}$)')

    # =========================================================================
    # u_dia,z:
    print('Plotting horizontal u_dia,z distribution...')

    lvl = 0.00015
    levels = np.linspace(-lvl, lvl, 100)
    cmap = plt.cm.get_cmap('seismic')
    cmap.set_under("cyan")
    cmap.set_over('magenta')
    norm = TwoSlopeNorm(vmin=-lvl, vcenter=0, vmax=lvl)
    levels_s=[0.2, 0.5, 0.8]
    colors = pl.cm.viridis(np.linspace(0.2,0.5,3)) #Colours used for line plots

    w = [w_dia_kling, w_dia_getm]

    ii = 0 #just a counter
    for dia_ax in [axs[cc+2], axs[cc+4]]:

        cax = dia_ax.pcolormesh(lonx, latx, -w[ii], vmin=-lvl, vmax=lvl,
                                cmap=cmap, norm=norm, zorder=1)
        CS = dia_ax.contour(lonc, latc, rel_z_mean, levels=levels_s,
                            colors='k', zorder=3)
        CC = dia_ax.contour(lonc, latc, bathy, levels=[12],
                            colors=['darkgrey'], linewidths=0.8, zorder=2)
        
        dia_ax.set_xlim([8.35, 9.3])
        dia_ax.set_ylim([53.7, 54.2])
        dia_ax.clabel(CS, inline=True, fontsize=8, fmt='%1.1f')
        dia_ax.set_aspect("1.8")
        
        ii += 1

    axs[cc+4].set_xlabel('lon (°E)')
    
    if cc==0:
        axs[cc+2].set_ylabel('lat (°N)')
        axs[cc+4].set_ylabel('lat (°N)')
        axs[cc+2].tick_params('both', which='both', direction='in',
                              bottom=True, top=True,
                               left=True, right=True,
                               labelbottom=False)
        axs[cc+4].tick_params('both', which='both', direction='in',
                              bottom=True, top=True,
                               left=True, right=True,
                               labelbottom=True)
    if cc==1:
        axs[cc+2].tick_params('both', which='both', direction='in',
                              bottom=True, top=True,
                               left=True, right=True,
                               labelleft=False, labelbottom=False)
        axs[cc+4].tick_params('both', which='both', direction='in',
                              bottom=True, top=True,
                               left=True, right=True,
                               labelleft=False, labelbottom=True)
        
        cbar1 = fig.colorbar(cax, ax=axs[2:4], orientation='vertical',
                             shrink=0.7, extend='both')
        cbar1.set_label('$-1/2 \partial_S m_{\mathrm{xy}}(S = $ '
                        + str(s) + '$,x,y)$\n(ms$^{-1}$)' )
        cbar2 = fig.colorbar(cax, ax=axs[4:], orientation='vertical',
                             shrink=0.7, extend='both')
        cbar2.set_label('$-u_{\mathrm{dia,z}}^{\mathrm{S}}(S = $ '
                        + str(s) + '$,x,y)$\n(ms$^{-1}$)')
        
        # Force scientific notation for colorbar labels:
        for cbar in [cbar1, cbar2]:
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
            cbar.ax.yaxis.set_offset_position('left') 
            
            
    #==========================================================================
    #%% Correlation between online u_dia and Klingbeil et al. (2023) u_dia:
    #==========================================================================

    corr, _ = pearsonr(w_dia_getm[~np.isnan(w_dia_kling) & ~np.isnan(w_dia_getm)],
    				w_dia_kling[~np.isnan(w_dia_kling) & ~np.isnan(w_dia_getm)])
    print('Pearsons correlation: %.3f' % corr)
    print('R^2: %.3f' % corr**2)

    
    cc += 1
    
#==============================================================================    
#%% LOOP ENDS
#==============================================================================

# # Add coastline to each subplot
# for shape in list(sf_coast.iterShapes()): #pull all shapes out of shapefile
#     x_lon = np.zeros((len(shape.points),1))
#     y_lat = np.zeros((len(shape.points),1))
#     for ip in range(len(shape.points)):
#         x_lon[ip] = shape.points[ip][0]
#         y_lat[ip] = shape.points[ip][1]

#     for ax in axs:
#         ax.plot(x_lon, y_lat, color="k", linewidth=1, zorder=-1)


labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
locations = [1, 1, 1, 1, 1, 1, 4, 4]
ii = 0
for ax in axs:
    # Add panel labels
    anchored_text = AnchoredText( labels[ii], loc=locations[ii], frameon=False,
                                 prop=dict(color='k', zorder=1000,
                                           fontweight="bold") )
    ax.add_artist(anchored_text)
    
    #Color grid cell area in plain white
    plain_cmap = mpl_c.ListedColormap(['w'])
    bathy[np.isnan(lonx[:-1,:-1])] = np.nan
    bathy[np.isnan(lonx[1:,:-1])]  = np.nan
    bathy[np.isnan(lonx[:-1,1:])]  = np.nan
    bathy[np.isnan(lonx[1:,1:])]   = np.nan
    lonx_plt = np.copy(lonx.data)
    latx_plt = np.copy(latx.data)
    lonx_plt[np.isnan(lonx_plt)] = 12
    latx_plt[np.isnan(latx_plt)] = 53
    ax.pcolormesh(lonx_plt, latx_plt, bathy, cmap=plain_cmap, zorder=-0.5)
    
    # # Fill land:
    # ax.add_collection(add_land(sf_land, 'lightgrey'))
    # # and now add some patches in the same color because the land polygons
    # # are somewhat faulty... but there is no better dataset :'(
    # rect1 = Rectangle((8.51, 53.7), width=0.25, height=0.17, 
    #                  edgecolor=None, facecolor='lightgrey', zorder=-2)
    # ax.add_artist(rect1)
    # rect2 = Rectangle((9, 53.3), width=1.5, height=0.7, 
    #                  edgecolor=None, facecolor='lightgrey', zorder=-2)
    # ax.add_artist(rect2)
    
    # Add open boundary
    # Set missing values back to NaN to be able to plot the proper line
    lonx_plt[lonx_plt==12] = np.nan
    latx_plt[latx_plt==53] = np.nan
    ax.plot(lonx_plt[:,0], latx_plt[:,0], 'grey', linewidth=2)
    
    ii += 1
    
    
if savefig:
    fig.savefig('plots/paper/horizontal.png', dpi=600)
plt.show()

print('\nDone!')
