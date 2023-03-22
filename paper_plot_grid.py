# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 12:43:01 2022

@author: Nina Reese / nina.reese@io-warnemuende.de

**Data processing / analysis for GETM model output**

Plots the topography and numerical grid used for the tidal Elbe setup presented
in [1], including details of two selected areas:
    (a) Setup topography + observational station locations
    (b) Overview of the German Bight
    (c) Numerical, horizontal grid from the setup
    (d) Grid detail near Cuxhaven
    (e) Grid detail around Hamburg
==> Fig. 2 in [1]

[1] N. Reese,U. Graewe, K. Klingbeil, X. Li, M. Lorenz, H. Burchard, 2023:
    Local mixing determines spatial structure of diahaline exchange flow in a
    mesotidal estuary – a study of extreme runoff conditions.
    J. Phys. Oceanogr., submitted.
"""


import numpy as np
import pandas as pd
import xarray
import shapefile
import cartopy
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from pyproj import Transformer

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_c
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#==============================================================================
#%% Manual Input:
#==============================================================================

# path to nc file containing grid data for the setup:
grid_path = ('/silod6/reese/tools/getm/setups/elbe_realistic/' +
              'topo_smoothed_v20.nc4')

# # path to the shapefiles containing the coastline and land:
# path = "coastline/"
# coast_name = "Coastline_WaddenSea" #coastline
# land_name  = "shoreline_northsea"  #closed land polygons
# sf_coast = shapefile.Reader(path + coast_name)
# sf_land  = shapefile.Reader(path + land_name)


#==============================================================================
#%% FUNCTION DEFINITIONS
#==============================================================================


def plot_connectors(ax, inax, xyA1=1, xyB1=0):
    
    """
    Plots zoom indicating window for inset axis and adds connector lines
    
    INPUT:
        ax: [matplotlib axis object] subplot axis
        inax: [matplotlib axis object] subplot axis
            subplot axis that acts as the zoom window
        xyA1: [scalar, dtype=int] int of value 0 or 1
            if 0, connectors are drawn to the left corners of inax
            if 1, connectors are drawn to the right corners of inax
        xyB1: [scalar, dtype=int] int of value 0 or 1
            if 0, connectors are drawn to the left corners of zoom window
            if 1, connectors are drawn to the right corners of zoom window
    OUTPUT:
        None
    """
    
    import matplotlib.patches as mpatches
    
    xlims = inax.get_xlim()
    ylims = inax.get_ylim()
    
    rect = mpatches.Rectangle((xlims[0], ylims[0]), width=(xlims[1]-xlims[0]), 
                              height=(ylims[1]-ylims[0]), 
                     edgecolor='k', facecolor='None', zorder=3)
    ax.add_artist(rect)

    for j in range(2):
        # inset / zoom axis co-ordinates ax
        xyA = (xyA1, j)
        # data co-ordinates in ax
        xyB = (xlims[xyB1], ylims[j])
        connect = mpatches.ConnectionPatch(xyA, xyB, 'axes fraction', 'data',
                        axesA=inax, axesB=ax, arrowstyle="-", edgecolor='0.6',
                        zorder=10)
        ax.add_patch(connect)

        
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
#%% MAIN: Starting script
#==============================================================================


print('\n============================================')
print('* Moin!')
print('* We are taking the Hobbits to Glueckstadt!')
print('============================================\n')


#%% Computing Model km

print('Preparing data...')

#Path to mean GETM output for TEF analysis and for model-km:
date = '20130101'
mean_base = '../store/exp_2022-08-12/OUT_182/' + date + '/'  
mean_file_name = 'Mean_all.' + date + '.nc4'
print('Loading ' + mean_file_name + ' data...')
fp = mean_base + mean_file_name
lonc   = xarray.open_mfdataset(fp)['lonc'][:]
latc   = xarray.open_mfdataset(fp)['latc'][:]
dx_tot = xarray.open_mfdataset(fp)['dxc'][:]

lonc = np.asarray(lonc)
latc = np.asarray(latc)
dx_tot = np.asarray(dx_tot)

#Along-channel distance in km (for plotting):
dx = dx_tot[157,:]
dx[np.isnan(dx)] = 0
dx[454:509] = dx_tot[160,454:509] #fill Hamburg area with Norderelbe distances
dx[452] = 351.272
dx[453] = 356.421
dx[509] = 256.211
dx[510] = 219.797
x = np.cumsum(dx[::-1])[::-1]/1000 #distance along thalweg j=157 in km, with upstream end at 0km

x_ind = np.arange(len(x)) #indices of x-array
x_interp = np.linspace(0,170,18) #x-indices to interpolate
ind_int = griddata(x, x_ind, x_interp)

x_interp = x_interp[~np.isnan(ind_int)]
ind_int = ind_int[~np.isnan(ind_int)]

lon = lonc[157,:]
lon[454:509] = lonc[161,454:509]
lon[453] = lonc[159,453]
lon[509] = lonc[159,509]
lat = latc[157,:]
lat[454:509] = latc[161,454:509]
lat[453] = latc[159,453]
lat[509] = latc[159,509]
xxc = griddata(x_ind, lon, ind_int)
xyc = griddata(x_ind, lat, ind_int)


#%% Load some more stuff

latx  = xarray.open_mfdataset(grid_path)['latx'][:]
lonx  = xarray.open_mfdataset(grid_path)['lonx'][:]
bathy = xarray.open_mfdataset(grid_path)['bathymetry'][:]

bathy = np.asarray(bathy)


#==============================================================================
#%% PLOTTING
#==============================================================================

print('Preparing plot...')

fig = plt.figure(figsize=(10,10), constrained_layout=False)

gs0 = gridspec.GridSpec(4, 2, figure=fig, hspace=0.12, wspace=0.02, width_ratios=[2.2, 1])
gs1 = gridspec.GridSpec(1, 1, left=0.1, right=0.9, figure=fig)

ax1 = fig.add_subplot(gs0[:2, 1], projection=cartopy.crs.PlateCarree()) #overview German Bight
ax2 = fig.add_subplot(gs0[:2, 0]) #Setup topography Elbe + stations
ax3 = fig.add_subplot(gs0[2:,0])  #Elbe Grid
ax4 = fig.add_subplot(gs0[2,1])   #detail Elbe grid: Cuxhaven
ax5 = fig.add_subplot(gs0[3,1])   #detail Elbe grid: Hamburg

# Add a "global" axis to create shared x- and y-labels
ax6 = fig.add_subplot(gs1[:], frameon=False)
# hide ticks and tick labels of the big axes
ax6.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax6.grid(False)
ax6.set_xlabel('Lon (°E)')
ax6.set_ylabel('Lat (°N)')

#Custom lines and labels for legend for elevation levels
custom_lines = [Line2D([0], [0], color='yellowgreen', lw=1),
                Line2D([0], [0], color='navy', lw=1),
                Line2D([0], [0], color='k', lw=0.2)]
custom_labels = ['Depth = 0 m', 'Depth = 10 m', 'Coastline']


#Load some map features
res = '10m' #resolution of cartopy features. NOT in meters!
bdry = cartopy.feature.NaturalEarthFeature(category='cultural', facecolor='None',
            edgecolor='k', name='admin_0_boundary_lines_land', scale=res)
land = cartopy.feature.NaturalEarthFeature('physical', 'land',
            scale=res, edgecolor='k', facecolor=cfeature.COLORS['land'])
ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', 
            scale=res, edgecolor='none', facecolor=cfeature.COLORS['water'])
lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', 
            scale=res, edgecolor='b', facecolor=cfeature.COLORS['water'])
rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines',
            scale=res, edgecolor='b', facecolor='none')
#land_highres = cfeature.GSHHSFeature()


#%% OVERVIEW GERMAN BIGHT

print('Overview German Bight...')

# Add map features
ax1.add_feature(land)
ax1.add_feature(ocean)
ax1.add_feature(lakes, alpha=1)
ax1.add_feature(rivers, linewidth=0.5)
ax1.add_feature(bdry, linewidth=0.5)
ax1.coastlines(res, linewidth=0.3)

# Formatting
ax1.set_facecolor(cfeature.COLORS['water'])
ax1.set_aspect("1.8", adjustable='datalim')
ax1.set_extent([-6, 14, 48, 66])
ax1.xaxis.tick_top()
ax1.yaxis.tick_right()
ax1.set_xticks([-6, 0, 6, 12], crs=cartopy.crs.PlateCarree())
ax1.set_yticks([51, 54, 57, 60, 63], crs=cartopy.crs.PlateCarree())
ax1.tick_params('both', which='both', direction='in', bottom=True, top=True,
                            left=True, right=True, labelleft=False, 
                            labelright=True, labeltop=True, labelbottom=False)

# Label North Sea
ax1.text(1, 55, 'North\nSea', fontsize=10, fontstyle='italic', color='navy')


#%% SETUP TOPOGRAPHY + STATIONS

print('Setup topography and stations...')

# data preparation for plot
# (contourf cannot handle NaN in x or y-values, so we set z to NaN instead)
lonc[np.where(lonc==0)] = np.nan
latc[np.where(latc==0)] = np.nan
bathy[np.where(np.isnan(lonc))] = np.nan
bathy[np.where(np.isnan(latc))] = np.nan
lonc[np.where(np.isnan(lonc))] = 12
latc[np.where(np.isnan(latc))] = 56

# Plot the topography:
#lvls = [-2,0,2,5,10,15,20,30]
lvls = 60
cax = ax2.contourf(lonc, latc, bathy, levels=lvls, cmap='gist_earth_r')
#ax2.contour(lonc, latc, bathy, levels=[-2,0,2,5,10,15,20,30], linewidths=0.1, colors='k', alpha=0.5) 
cbaxes = inset_axes(ax2, width="50%", height="5%", loc=1)
cbar = fig.colorbar(cax, cax=cbaxes, orientation='horizontal',
                    spacing='proportional') 
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Depth (m)', size=8)


# Plot location of TEF transects T_1 and T_2:
ax2.plot(lonc[127:272,78], latc[127:272,78], '-r', linewidth=3)
ax2.text(np.nanmax(lonc[127:272,78])+0.01, np.nanmax(latc[127:272,78]), 'T$_1$',
        color='r', fontsize=12, fontweight='bold')
ax2.plot(lonc[~np.isnan(bathy[:,254]),254], latc[~np.isnan(bathy[:,254]),254],
        '-r', linewidth=3)
ax2.text(np.nanmin(lonc[~np.isnan(bathy[:,254]),254])-0.08, 
        np.nanmin(latc[~np.isnan(bathy[:,254]),254])-0.015, 'T$_2$', color='r',
        fontsize=12, fontweight='bold') 


# Add stations:
# load dataset containing station info:
stations = pd.read_csv('observations/gauges.txt', comment='#', skip_blank_lines=True,
                        delimiter=';', header=None, skipinitialspace=True)
# Colors and markers for different station types:
colors = ['darkred', 'red', 'salmon']
markers = ['o', '^', '*']
markersizes = [25, 28, 60]

#     [  Cux,   Bru,    Bro,    Glue,   Sta,    Schu,   StPa,   Zol,    LZ4a, Cux AL, LZ2a,   LZ1b,    D4 ]
dx =  [-0.15,  -0.01,  +0.1,   +0.080, -0.25,  -0.065, -0.06,  -0.07,  -0.26, -0.25, -0.03,  -0.02,  -0.16]
dy =  [-0.15,  +0.045, +0.00,  -0.005, -0.02,  +0.075, +0.04,  +0.09,  -0.02, -0.05, +0.1,   -0.07,  -0.03]
dxl = [-0.010, +0.004, +0.09,  +0.070, -0.075, +0.01,  +0.04,  +0.01,  -0.1,  -0.10, +0.01,  +0.02,  -0.06]
dyl = [-0.083, +0.035, +0.01,  +0.001, -0.003, +0.063, +0.025, +0.075, -0.01, -0.02, +0.075, -0.038, -0.01]

# loop through each station: 
for station in stations.T.iteritems():
    ii = station[0]
    station = station[1] #get rid of column that only contains item numbers
    i = station[0] - 1 #index shift by -1 because python starts at 0
    j = station[1] - 1
    handle = 'txt' #initialise station handle, marker, and color
    marker = '.'
    markersize = 25
    c = 'k'
    
    # Assign handles etc. for each station station type
    if station[2] == 'G':
        handle = station[3][:]
        marker = markers[0]
        markersize = markersizes[0]
        c = colors[0]
    elif station[2] == 'ST':
        handle = station[3][:4]
        marker = markers[1]
        markersize = markersizes[1]
        c = colors[1]
    elif station[2] == 'LST':
        handle = station[3][:2]
        marker = markers[2]
        markersize = markersizes[2]
        c = colors[2]
    else:
        raise ValueError( "Caution! There is an unknown station type here!" + 
                " Known types are:"
              + "\n\tG: Gauge" + "\n\tST: Salinity and temperature"
              + "\n\tLST: Surface and bottom salinity and temperature" )
    
    #special cases:
    if station[3] == 'Cuxhaven Steubenhoeft':
        handle = 'Cuxhaven\nSteubenhoeft'
    elif station[3] == 'Stadersand':
        handle = 'Stader-\nsand'
    elif station[3] == 'Zollenspieker':
        handle = 'Zollen-\nspieker'
    elif station[3] == 'Cuxhaven Alte Liebe':
        handle = 'Cux AL'
    elif station[3] == 'St. Pauli':
        handle = 'Hamburg\nSt. Pauli'
        
    ax2.scatter(lonc[j,i], latc[j,i], color='k', facecolor=c, marker=marker,
                s=markersize, linewidth=0.2, zorder=10)
    ax2.text(lonc[j,i]+dx[ii], latc[j,i]+dy[ii], handle, fontsize=10)
    ax2.plot( [lonc[j,i], lonc[j,i]+dxl[ii]], [latc[j,i], latc[j,i]+dyl[ii]],
            linewidth=0.5, color='k')

# Add label for Kiel Canal
ax2.text(9.245, 54.0, 'NOK', fontsize=10, fontstyle='italic', color='navy',
         rotation=84)


# Make a legend
labels = stations[2].unique()
labels[labels=='G'] = 'Gauge'
labels[labels=='ST'] = 'Salt/Temp.'
labels[labels=='LST'] = 'Salt/Temp. \nstratification'
dots = []
for ii in range(len(labels)):
    dots.append( Line2D([0], [0], marker=markers[ii], color=colors[ii],
                        markeredgecolor='k', markersize=6, linestyle='',
                        linewidth=0.2) )
legend = ax2.legend(dots, labels, loc=3, title='Station types') #loc=(0.74,0.62)


# Some formatting
ax2.set_aspect("1.8")
ax2.set_facecolor(cfeature.COLORS['water']) #set background to water color
ax2.grid(zorder=0, linewidth=0.5, color='gainsboro')
ax2.set_xlim([8.36, 10.35])
ax2.set_ylim([53.38, 54.2])
ax2.tick_params('both', which='both', direction='in', bottom=True, top=True,
                        left=True, right=True, labelleft=True, labelright=False,
                        labelbottom=False, labeltop=True)


#%% OVERVIEW ELBE GRID

print('Overview numerical grid...')

ax3.plot(lonx[::3,::3], latx[::3,::3], '-k', linewidth=0.08) #transversal grid lines
ax3.plot(lonx[::3,::3].T, latx[::3,::3].T, '-k', linewidth=0.08) #longitudinal grid lines

# Add open boundary
ax3.plot(lonx[:,0], latx[:,0], '-k', linewidth=2)

# Add topography outlines
CS = ax3.contour(lonc, latc, bathy, levels=[0,10], colors=['yellowgreen', 'navy'], linewidths=0.5)

# Label Medem Branch and Geesthacht
ax3.text(9.22, 53.955, 'Medem Branch', fontsize=10, color='k')
ax3.plot( [9.2, 8.85], [53.96, 53.9], linewidth=0.5, color='k')
ax3.text(10.15, 53.52, 'Geest-\nhacht', fontsize=10, color='k')
ax3.plot( [10.24, 10.35], [53.5, 53.421], linewidth=0.5, color='k')

# Formatting
ax3.set_aspect("1.8")
ax3.set_xlim([8.36, 10.35])
ax3.set_ylim([53.38, 54.2])
ax3.tick_params('both', which='both', direction='in', bottom=True, top=True,
                            left=True, right=True, labelleft=True, labelright=False)
ax3.legend(custom_lines, custom_labels, loc=3)


#%% DETAIL ELBE GRID: Cuxhaven

print('Grid detail Cuxhaven...')

ax4.plot(lonx[::3,::3], latx[::3,::3], '-k', linewidth=0.08) #transversal grid lines
ax4.plot(lonx[::3,::3].T, latx[::3,::3].T, '-k', linewidth=0.08) #longitudinal grid lines

# Add topography outlines
CS = ax4.contour(lonc, latc, bathy, levels=[0,10], colors=['yellowgreen', 'navy'], linewidths=0.5)

# Formatting
ax4.set_aspect("1.8")
ax4.set_xlim([8.6, 8.95])
ax4.set_ylim([53.82, 53.97])
ax4.tick_params('both', which='both', direction='in', bottom=True, top=True,
                        left=True, right=True, labelleft=False, labelright=True,
                        labelbottom=True, labeltop=False)


#%% DETAIL ELBE GRID: Hamburg

print('Grid detail Hamburg...')

ax5.plot(lonx[::1,::1], latx[::1,::1], '-k', linewidth=0.08) #transversal grid lines
ax5.plot(lonx[::1,::1].T, latx[::1,::1].T, '-k', linewidth=0.08) #longitudinal grid lines

# Formatting
ax5.set_aspect("1.8")
ax5.set_xlim([9.9, 10.1])
ax5.set_ylim([53.46, 53.55])
ax5.tick_params('both', which='both', direction='in', bottom=True, top=True,
                       left=True, right=True, labelleft=False, labelright=True)


#%% FINISH PLOT

print('Finishing plot...')


#Color grid cell area with plain white
plain_cmap = mpl_c.ListedColormap(['w'])
bathy[np.isnan(lonx[:-1,:-1])] = np.nan
bathy[np.isnan(lonx[1:,:-1])]  = np.nan
bathy[np.isnan(lonx[:-1,1:])]  = np.nan
bathy[np.isnan(lonx[1:,1:])]   = np.nan
lonx_plt = lonx.data
latx_plt = latx.data
lonx_plt[np.isnan(lonx_plt)] = 12
latx_plt[np.isnan(latx_plt)] = 53
ax3.pcolormesh(lonx_plt, latx_plt, bathy, cmap=plain_cmap)
ax4.pcolormesh(lonx_plt, latx_plt, bathy, cmap=plain_cmap)
ax5.pcolormesh(lonx_plt, latx_plt, bathy, cmap=plain_cmap)


# Add Elbe kms:
for ax in [ax2, ax3, ax4, ax5]:
    ax.scatter(xxc, xyc, s=10, color='yellow', edgecolors='orange',
                marker="o", alpha=1, linewidths=0.1, zorder=3)
    
for ii in range(len(x_interp)):
    if x_interp[ii]==30:
        ax5.text(xxc[ii], xyc[ii]+0.005, str(int(x_interp[ii])),
                size=7, color='k')
    if (x_interp[ii]>=130) & (x_interp[ii]<=150):
        ax4.text(xxc[ii], xyc[ii]+0.01, str(int(x_interp[ii])),
                size=7, color='k')
    
    if ii==3:
        ax2.text(xxc[ii]-0.05, xyc[ii]-0.02, str(int(x_interp[ii])),
                size=7, color='k')
        ax3.text(xxc[ii]-0.05, xyc[ii]-0.02, str(int(x_interp[ii])),
                size=7, color='k')
        ax5.text(xxc[ii]-0.02, xyc[ii]-0.01, str(int(x_interp[ii])),
                size=7, color='k')
    elif ii==9:
        ax2.text(xxc[ii]-0.05, xyc[ii]-0.025, str(int(x_interp[ii])),
                size=7, color='k')
        ax3.text(xxc[ii], xyc[ii]+0.01, str(int(x_interp[ii])),
                size=7, color='k')
    elif ii==11:
        ax2.text(xxc[ii]-0.05, xyc[ii]-0.03, str(int(x_interp[ii])),
                size=7, color='k')
        ax3.text(xxc[ii], xyc[ii]+0.01, str(int(x_interp[ii])),
                size=7, color='k')
    elif ii==14:
        ax2.text(xxc[ii], xyc[ii]+0.01, str(int(x_interp[ii])),
                size=7, color='k')
        ax3.text(xxc[ii]+0.03, xyc[ii]+0.001, str(int(x_interp[ii])),
                size=7, color='k')
    else:
        ax2.text(xxc[ii], xyc[ii]+0.01, str(int(x_interp[ii])),
                size=7, color='k')
        ax3.text(xxc[ii], xyc[ii]+0.01, str(int(x_interp[ii])),
                size=7, color='k')



# #Add coastline from shapefile:
# for shape in list(sf_coast.iterShapes()): #pull all shapes out of shapefile
#     x_lon_sh = np.zeros((len(shape.points)))
#     y_lat_sh = np.zeros((len(shape.points)))
#     for ip in range(len(shape.points)):
#         x_lon_sh[ip] = shape.points[ip][0]
#         y_lat_sh[ip] = shape.points[ip][1]
        
#     ax2.plot(x_lon_sh, y_lat_sh, color="k", linewidth=0.1, zorder=-1)
#     ax3.plot(x_lon_sh, y_lat_sh, color="k", linewidth=0.1, zorder=-1)
#     ax4.plot(x_lon_sh, y_lat_sh, color="k", linewidth=0.1, zorder=-1)
#     ax5.plot(x_lon_sh, y_lat_sh, color="k", linewidth=0.1, zorder=-1)


#Add panel labels:
labels = ['(b)', '(a)', '(c)', '(d)', '(e)']
locations = [2, 2, 2, 3, 3]
colors = ['None', cfeature.COLORS['land'], 'lightgrey', 'lightgrey', 'lightgrey']
cc = 0
for ax in [ax1, ax2, ax3, ax4, ax5]:
    anchored_text = AnchoredText( labels[cc], loc = locations[cc],
                                  prop=dict(fontsize=12), frameon=False)
    ax.add_artist(anchored_text)
    
    # # fill land with color:
    # if cc>0:
    #     ax.add_collection(add_land(sf_land, colors[cc]))
    #     # and now add some patches in the same color because the land polygons
    #     # are somewhat faulty... but there is no better dataset :'(
    #     rect1 = Rectangle((8.51, 53.7), width=0.25, height=0.17, 
    #                      edgecolor=None, facecolor=colors[cc], zorder=-2)
    #     ax.add_artist(rect1)
    #     rect2 = Rectangle((9, 53.3), width=1.5, height=0.7, 
    #                      edgecolor=None, facecolor=colors[cc], zorder=-2)
    #     ax.add_artist(rect2)
    cc += 1
    

# Indicate zoom windows for inset axes:
plot_connectors(ax1, ax2)
plot_connectors(ax3, ax4, xyA1=0, xyB1=1)
plot_connectors(ax3, ax5, xyA1=0, xyB1=1)
ax3.text(9.9, 53.555, 'Hamburg', fontsize=8, color='k') #Label Hamburg inset box

fig.savefig('plots/paper/grid.png', dpi=500)

plt.show()


#==============================================================================
#%% END SCRIPT
#==============================================================================

print('\n============================================')
print('* Done! All Hobbits have been delivered!')
print('============================================\n')

