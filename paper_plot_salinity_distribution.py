# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:39:15 2022

@author: Nina Reese / nina.reese@io-warnemuende.de

**Data processing / analysis for GETM model output**

Plots spring-neap averaged distribution of
(1) Surface salinity
(2) Salinity along thalweg in x-z-space
for two prescribed months from model output from the tidal Elbe setup
presented in [1].
==> Fig. 6 in [1]

[1] N. Reese,U. Graewe, K. Klingbeil, X. Li, M. Lorenz, H. Burchard, 2023:
    Local mixing determines spatial structure of diahaline exchange flow in a
    mesotidal estuary – a study of extreme runoff conditions.
    J. Phys. Oceanogr., submitted.
"""

import datetime
import xarray
import shapefile
import numpy as np
from scipy.interpolate import griddata
from pyproj import Transformer
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
plt.rcParams.update({       #Use LaTeX for plotting
    "text.usetex": True})

# We expect to have mean of empty slices in here, so let's ignore that warning:
import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

#==============================================================================

# Manual Input
exp = 'exp_2022-08-12'

start = ['2012-09-01 00:00:00', '2013-06-01 00:00:00']

s = 15 #salinity class (g/kg) for horizontal map plots
#time span since start for averaging:
timedelta = np.timedelta64(29, 'D') + np.timedelta64(12, 'h') + np.timedelta64(48, 'm')
thalweg = int(157) #y-index of thalweg-following grid line

# Auto-compute startdate and date string from start:
startdate = datetime.datetime.strptime(start[0], '%Y-%m-%d %H:%M:%S')
date = startdate.strftime('%Y%m') + '01'

#Path to GETM output in 'Mean' file:
mean_base = '../store/' + exp + '/OUT_182/' + date + '/' 
mean_file_name = 'Mean_all.' + date + '.nc4'

# # path to the shapefiles containing the coastline and land:
# path = "coastline/"
# coast_name = "Coastline_WaddenSea" #coastline
# land_name  = "shoreline_northsea"  #closed land polygons
# sf_coast = shapefile.Reader(path + coast_name)
# sf_land  = shapefile.Reader(path + land_name)

savefig = True #If True, the resulting figure will be saved




#==============================================================================
#%%  RIVER RUNOFF
#==============================================================================


def prepare_salt_total(path, start, timedelta, x, Xz, Z):
    
    """
    Prepare salinity data in such a way that the average
    salinity distribution along the river thalweg can be plotted in an
    x-z* coordinate system with regular z-coordinate spacing.
    Averaging period is from start to start+timedelta.
    
    INPUT:
        path [scalar, dtype=Str]
            path to file containing salinity and sigma layer height data
        start [scalar, dtype=np.datetime64]
            date from which to start averaging
        timedelta [scalar, dtype=np.timedelta64]
            length of time to consider after given start date for the temporal
            averaging
        x [np.ndarray, dtype=np.float, dim=(time, z, x)]
            x-coordinates along the thalweg of the river
        Xz, Z [np.ndarray, dtype=np.float, dim(x, z*)]
            x- and regularly spaced z-positions along the thalweg line
            
    OUTPUT:
        S_interp [np.ndarray, dtype=np.float, dim=(x, z*)]
            average salinity along the thalweg for a given spring/neap tide,
            interpolated onto a vertically regular grid (x, z*)
    """
    
    print('\t\t Loading data...')
    salt_h = xarray.open_mfdataset(path)['salt_mean'][:,-1,:,:]
    salt = xarray.open_mfdataset(path)['salt_mean'][:,:,thalweg,:]
    h = xarray.open_mfdataset(path)['hn_mean'][:,:,thalweg,:]
    h = h.loc[start:start+timedelta] #first spring/neap tide
    
    print('\t\t Average surface salinity...')
    salt_h = salt_h.loc[start:start+timedelta]
    salt_h = np.nanmean(salt_h, axis=0)
    
    print('\t\t Computing z...')
    z = np.cumsum(h[:,::-1,:], axis=1) #water depth of sigma layer
    h = None; del h #clear memory
    z = (-1)*z
    z = np.asarray(z)
    xt = np.tile( x, (len(z[:,0,0]),len(z[0,:,0]),1) ) #make x same dimension as z
    
    
    print('\t\t Preparing salinity data...')
    salt = salt.loc[start:start+timedelta]
    salt = np.asarray(salt)

    #initialise interpolated S
    S_interp = np.zeros((len(salt[:,0,0]), len(Xz[:,0]), len(Xz[0,:])))


    print('\t\t Interpolating vertically...')
    #Interpolate salinity from sigma coords onto a vertically regular
    #x-z* coordinate grid
    for tt in range(len(salt[:,0,0])):
        S_interp[tt,:,:] = griddata( ((xt[tt,:,:]).flatten(),
                                      (z[tt,::-1,:]).flatten()),
                            (salt[tt,:,:]).flatten(), (Xz, Z), method='linear' )
        
        # find lim index as function of x
        #fill near-surface NaNs with non-NaN value that is closest to surface
        for ii in range(len(S_interp[tt,:,0])):
            lim = 0
            notnan = np.where(~np.isnan(S_interp[tt,ii,:]))
            if np.size(notnan) != 0:
                lim = np.nanmin(notnan)
            S_interp[tt,ii,:lim] = np.tile(S_interp[tt,ii,lim], (lim,1)).T
            
    salt = None; del salt #free some memory
    
    
    # temporally averaged salinity along thalweg:
    print('\t\t Temporal averaging...')
    S_interp = np.nanmean(S_interp[:,:,:], axis=0)
    
    return([salt_h, S_interp])


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
#%%  LOAD DATA
#==============================================================================

print('\n==================================')
print('* Hello there! Starting script...')
print('==================================\n')

print('Salinity plot preparations...')

fig = plt.figure(figsize=(6,7))#, tight_layout=True)

# Create overarching gridspec with two rows:
gs0 = gridspec.GridSpec(2, 1, wspace=0.01, hspace=0.1, left=0.1, right=0.95,
                    top=0.95, bottom=0.17, height_ratios=[1.8, 2])

# Create sub-gridspecs to create the subplots:
gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0], wspace=0.02)
gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1], hspace=0.1)

labels = ['(a)', '(b)', '(c)', '(d)']


print('\t Loading ' + mean_file_name + ' data...')
path = mean_base + mean_file_name #full path to file
bathy = xarray.open_mfdataset(path)['bathymetry'][:] #bathymetry...
thal_bathy = bathy[thalweg,:] #bathymetry along thalweg
dx_tot = xarray.open_mfdataset(path)['dxc'][:]
lonc = xarray.open_mfdataset(path)['lonc'][:] #longitude of cell centers
latc = xarray.open_mfdataset(path)['latc'][:] #latitude of cell centers

print('\t Computing model-km...')
#Along-channel distance in km (for plotting):
dx = np.asarray(dx_tot)[thalweg,:]
dx[np.isnan(dx)] = 0
dx[454:509] = dx_tot[160,454:509] #fill Hamburg area with Norderelbe distances
dx[452] = 351.272
dx[453] = 356.421
dx[509] = 256.211
dx[510] = 219.797
x = np.cumsum(dx[::-1])[::-1]/1000 #distance along thalweg j=157 in km, with upstream end at 0km

dx_tot = None; del dx_tot

#============================================
# Model-km for plotting:
x_ind = np.arange(len(x)) #indices of x-array
x_interp = np.linspace(0,170,18) #x-indices to interpolate
ind_int = griddata(x, x_ind, x_interp)

x_interp = x_interp[~np.isnan(ind_int)]
ind_int = ind_int[~np.isnan(ind_int)]

# Interpolation, with special treatment
# of Hamburg area, where Elbe splits into two channels
lon = np.asarray(lonc[157,:])
lon[454:509] = lonc[161,454:509]
lon[453] = lonc[159,453]
lon[509] = lonc[159,509]
lat = np.asarray(latc[157,:])
lat[454:509] = latc[161,454:509]
lat[453] = latc[159,453]
lat[509] = latc[159,509]
xxc = griddata(x_ind, lon, ind_int)
xyc = griddata(x_ind, lat, ind_int)
#============================================

print('\t Preparing interpolation depths...')
#interpolation depths for regular vertical spacing
depths_interp = (-1)*np.linspace(-1,28,67,dtype=float)
# Coordinates for plotting & interpolation:
Xz, Z = np.meshgrid( x, depths_interp )
Xz = Xz.T
Z = Z.T

print('Loading additional data...')
fp='/silod6/reese/tools/getm/setups/elbe_realistic/topo_smoothed_v20.nc4'
lonx = xarray.open_mfdataset(fp)['lonx'][:] #longitude of grid vertices
latx = xarray.open_mfdataset(fp)['latx'][:] #latitude of grid vertices

lonx = lonx.data
latx = latx.data
lonc = lonc.data
latc = latc.data

# replacing NaN values for plotting
lonx[np.isnan(lonx)] = 12
latx[np.isnan(latx)] = 53
lonc[np.isnan(lonc)] = 12
latc[np.isnan(latc)] = 53


#%%
# =============================================================================
# Spring-neap averaged salinities
# =============================================================================


cc = 0

for s in start:

    #compute average spring and neap tide salinity distribution along thalweg
    print(s + '...')
    start = s
    startdate = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    date = startdate.strftime('%Y%m') + '01'
    month = startdate.strftime('%B')
    year = startdate.strftime('%Y')
    #Path to GETM output in 'Mean' file:
    mean_base = '../store/' + exp + '/OUT_182/' + date + '/' 
    mean_file_name = 'Mean_all.' + date + '.nc4'
    path = mean_base + mean_file_name
    start = np.datetime64(start)
    
    print('\t Computing...')
    [salt_h, salt_v] = prepare_salt_total(path, start, timedelta, x, Xz, Z)
    #set values below topography to NaN:
    salt_v[np.where(Z<-np.tile(thal_bathy+0.3, (len(depths_interp), 1)).T)] = np.nan
    
    print('\t Plot additions...')
    ax_h = fig.add_subplot(gs00[cc]) #horizontal salt distribution
    ax_v = fig.add_subplot(gs01[cc]) #vertical salt distribution
    
    
    print('\t\t Surface salinity...')
    
    cmap = plt.cm.get_cmap('viridis')
    cax = ax_h.pcolormesh(lonx, latx, salt_h, vmin=0, vmax=31, cmap=cmap, zorder=1)
    CC = ax_h.contour(lonc, latc, bathy, levels=[12], colors=['darkgrey'],
                      linewidths=0.8, zorder=5)
    c = ax_h.contour(lonc, latc, salt_h, levels=np.linspace(0, 30, 16),
                              colors='w', linewidths=0.5, zorder=10) #even salinities
    
    # Add TEF transects to plot:
    ax_h.plot(lonc[127:272,78], latc[127:272,78], '-r', linewidth=3, zorder=1000)
    ax_h.text(np.nanmin(lonc[127:272,78])-0.05,
              np.nanmin(latc[127:272,78])-0.05, 'T$_1$',
              color='r', fontsize=12, fontweight='bold', zorder=1000)
    
    ax_h.set_xlim([8.35, 9.3])
    ax_h.set_ylim([53.7, 54.2])
    ax_h.set_aspect("1.8")
    ax_h.set_xlabel('lon (°E)')
    ax_h.tick_params('both', which='both', direction='in', bottom=True, top=True,
                           left=True, right=True,
                           labelbottom=False,labeltop=True)
    ax_h.xaxis.set_label_position('top')
    
    #-------------------------------------------------------
    # #Add Coastline to subplot:
    # lon = []
    # lat = []

    # for shape in list(sf_coast.iterShapes()): #pull all shapes out of shapefile
    #     x_lon = np.zeros((len(shape.points),1))
    #     y_lat = np.zeros((len(shape.points),1))
    #     for ip in range(len(shape.points)):
    #         x_lon[ip] = shape.points[ip][0]
    #         y_lat[ip] = shape.points[ip][1]
               
    #     ax_h.plot(x_lon, y_lat, color="k", linewidth=0.8, zorder=-1)
    #-------------------------------------------------------
    
    props = {'backgroundcolor': 'w',
            'color':  'k',
            'weight': 'normal'
            }
    anchored_text = AnchoredText( labels[cc] + ' ' + month +' ' + year, loc=4,
                                 frameon=False, prop=props )
    ax_h.add_artist(anchored_text)
    
    # # Fill land:
    # ax_h.add_collection(add_land(sf_land, 'lightgrey'))
    # # and now add some patches in the same color because the land polygons
    # # are somewhat faulty... but there is no better dataset :'(
    # rect1 = Rectangle((8.51, 53.7), width=0.25, height=0.17, 
    #                  edgecolor=None, facecolor='lightgrey', zorder=-2)
    # ax_h.add_artist(rect1)
    # rect2 = Rectangle((9, 53.3), width=1.5, height=0.7, 
    #                  edgecolor=None, facecolor='lightgrey', zorder=-2)
    # ax_h.add_artist(rect2)
    
    # Add Elbe kms in horizontal plot:
    ax_h.scatter(xxc, xyc, s=10, color='yellow', edgecolors='orange',
                marker="o", alpha=1, linewidths=0.1, zorder=100)
    for ii in range(9, len(x_interp)):
        if ii==9:
            t = ax_h.text(xxc[ii]-0.06, xyc[ii]-0.03, str(int(x_interp[ii])),
                    size=7, color='k', weight='bold', zorder=100)
        elif ii==10:
            t = ax_h.text(xxc[ii]-0.03, xyc[ii]-0.03, str(int(x_interp[ii])),
                    size=7, color='k', weight='bold', zorder=100)
        else:
            t = ax_h.text(xxc[ii], xyc[ii]+0.01, str(int(x_interp[ii])),
                    size=7, color='k', weight='bold', zorder=100)


    
    
    print('\t\t Thalweg salinity...')
    
    ax_v.tick_params('both', which='both', direction='in', bottom=True, top=True,
                            left=True, right=True, zorder=1000)
    
    # Topography shape:
    ax_v.set_xlim([x[0], x[270]])
    ax_v.fill_between(x, -30, -thal_bathy, facecolor='grey', zorder=-2)
    ax_v.plot(x, -thal_bathy, color='k', linewidth=2, zorder=5)
    
    # Salt:
    cax = ax_v.contourf(Xz, Z, salt_v, levels=np.linspace(0,31,63),
                                cmap='viridis', zorder=-3)
    c = ax_v.contour(Xz, Z, salt_v, levels=np.linspace(0, 30, 16),
                              colors='w', zorder=-2) #even salinities
    c2 = ax_v.contour(Xz, Z, salt_v, levels=np.linspace(1, 31, 16),
                              colors='w', linestyles='--', linewidths=0.8,
                              alpha=0.5, zorder=-2) #odd salinities

    ax_v.set_ylim([-26,0])
    miny, maxy = ax_v.get_ylim()
    cl = ax_v.clabel(c, levels=np.linspace(0, 30, 16), inline = 1, fmt ='% 2d',
                  fontsize = 10)

    anchored_text = AnchoredText( labels[cc+2] + ' ' + month + ' ' + year, loc=4,
                                  frameon=False, prop=dict(color='w', zorder=1000),
                                  pad=0.02)
    ax_v.add_artist(anchored_text)
    ax_v.set_ylabel("z (m)")
    
    # add vlines marking the TEF transects:
    ax_v.vlines(x[78], -thal_bathy[78], maxy, colors='r', linewidths=3, zorder=3)
    t1 = ax_v.text(x[78]-2, -4.0, 'T$_1$', color='r', fontsize=12, fontweight='bold')
    t1.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor=None))
    ax_v.vlines(x[254], -thal_bathy[254], maxy, colors='r', linewidths=3, zorder=3)
    t2 = ax_v.text(x[254]+5, -4.0, 'T$_2$', color='r', fontsize=12, fontweight='bold')
    t2.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor=None))
    
    
    if cc==0:
        ax_h.set_ylabel('lat (°N)')
        ax_v.set_xticklabels('')
    else:
        ax_h.set_yticklabels('')
        ax_v.set_xlabel('$x$ (Elbe model-km)')
        
    cc += 1
  


#%%
# =============================================================================
# Plot results
# =============================================================================

print('Finishing plot...')

grid = plt.GridSpec(1,1, wspace=0.2, hspace=0.2, left=0.1, right=0.95,
                    top=0.95, bottom=-0.06)
# add a big axis, hide frame
k = fig.add_subplot(grid[0], frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
cbar = fig.colorbar(cax, ax=k, orientation='horizontal', shrink=0.6)#, anchor=(0.5,0))
cbar.set_label('$S$ (g/kg)')

if savefig:
    fig.savefig('plots/paper/isohalines.png', dpi=600)
plt.show()

print('\n==================================')
print('* Done! Another happy landing!')
print('==================================\n')