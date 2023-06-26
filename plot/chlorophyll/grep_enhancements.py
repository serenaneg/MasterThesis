#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 08:18:30 2023

@author: serena
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm, ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
import cmocean

#%%FUNCTIONS
def set_plot_theta(ncols, date, vmin, vmax):
    fig, axes = plt.subplots(ncols = ncols, nrows = 1, figsize = [18*ncols,10])
    levels = np.linspace(vmin, vmax, 20)
    cmap = 'cmo.thermal'
    
    for ax in axes.flat:
        ax.set_xlabel('Latitude [°N]', fontsize = 35)
        ax.set_ylabel('Depth [m]', fontsize = 35)
        
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        ax.xaxis.set_tick_params(labelsize = 30)
        ax.yaxis.set_tick_params(labelsize = 30)
        
        if ax != axes.flat[0] :
            ax.yaxis.set_tick_params(labelsize = 0)
            ax.set_ylabel('')
            
    for i, ax in enumerate(axes.flat):
        scalar = ax.contourf(lats, depth, theta.sel(time = date[i]), levels = levels,
                             extend = 'both', cmap=cmap, zorder = 1)
        ax.set_title('Temperature ' + str(date[i]), fontsize = 40, y=1.02)
        
        norm= Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
        sm.set_array([])
    
    fig.tight_layout(pad = 2.3)  
    
    return (fig, axes, sm)

def set_plot_sal(ncols, date, vmin, vmax):
    fig, axes = plt.subplots(ncols = ncols, nrows = 1, figsize = [18*ncols,10])
    levels = np.linspace(vmin, vmax, 20)
    cmap = 'cmo.haline'
    
    for ax in axes.flat:
        ax.set_xlabel('Latitude [°N]', fontsize = 35)
        ax.set_ylabel('Depth [m]', fontsize = 35)
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        ax.xaxis.set_tick_params(labelsize = 30)
        ax.yaxis.set_tick_params(labelsize = 30)
        
        if ax != axes.flat[0] :
            ax.yaxis.set_tick_params(labelsize = 0)
            ax.set_ylabel('')
            
    for i, ax in enumerate(axes.flat):
        scalar = ax.contourf(lats, depth, sal.sel(time = date[i]), levels = levels,
                             extend = 'both', cmap=cmap, zorder = 1)
        lines = ax.contour(lats, depth, sal.sel(time = date[i]), levels = [34.5], colors='black', linewidths = 2)
        ax.clabel(lines, fontsize=35, colors = 'black')
              
        ax.set_title('Salinity at ' + str(date[i]), fontsize = 40, y=1.02)
        norm= Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
        sm.set_array([])
        
        
        
    fig.tight_layout(pad = 2.3)  
       
    return (fig, axes, sm)

def set_plot_currents(ncols, date, cmap, vmin, vmax):
    fig, axes = plt.subplots(ncols = ncols, nrows = 1, figsize = [18*ncols,10])
    levels = np.linspace(vmin, vmax, 20)
    cmap = cmap
    
    for ax in axes.flat:
        ax.set_xlabel('Latitude [°N]', fontsize = 35)
        ax.set_ylabel('Depth [m]', fontsize = 35)
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        ax.xaxis.set_tick_params(labelsize = 30)
        ax.yaxis.set_tick_params(labelsize = 30)
        
        if ax != axes.flat[0] :
            ax.yaxis.set_tick_params(labelsize = 0)
            ax.set_ylabel('')
            
    for i, ax in enumerate(axes.flat):
        scalar = ax.contourf(lats, depth, speed.sel(time = date[i]), levels = levels,
                             extend = 'both', cmap=cmap, zorder = 1)
              
        ax.set_title('Speed currents at ' + str(date[i]), fontsize = 35, y=1.02)
        norm= Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
        sm.set_array([])
   
    fig.tight_layout(pad = 2.3)  
       
    return (fig, axes, sm)

def set_cbar(fig, c, title):    
    cbar = plt.colorbar(c, format='%.1f', spacing='proportional', ax=axes, shrink=0.9, pad = 0.01,
                        orientation = 'vertical', location = "right")
    cbar.set_label(label = title, fontsize = 35, y = 0.5, labelpad = 30)
    cbar.ax.tick_params(which='minor', size=15, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=25, width=1, color='k', direction='in', labelsize = 30)
    # fig.subplots_adjust(bottom=0.25)
    # cbar.ax.set_position([0.2, 0.08, 0.6, 0.08])
    return cbar

def set_plot_currents_2d(ncols, date, cmap, vmin, vmax, z):
    
    lon2d, lat2d = np.meshgrid(lons, lats)
    #sel depth z for velocity componets
    fig, axes = plt.subplots(nrows=1, ncols=ncols, subplot_kw = dict(projection=ccrs.PlateCarree()),  figsize = [19*ncols,12])

    for i, ax in enumerate(axes.flat):
    # Downsampling a 2d array and select date and depth
        step=2
        lon2d_sub = lon2d[1:len(lats)-1:step,1:len(lons)-1:step]
        lat2d_sub = lat2d[1:len(lats)-1:step,1:len(lons)-1:step]
        u_sub = u_2d.sel(time = date[i], depth = z, method='nearest')[1:len(lats)-1:step,1:len(lons)-1:step]
        v_sub = v_2d.sel(time = date[i], depth = z, method='nearest')[1:len(lats)-1:step,1:len(lons)-1:step]
        speed_sub = speed_2d.sel(time = date[i], depth = z, method='nearest')[1:len(lats)-1:step,1:len(lons)-1:step]
   
    # Define the starting point
        u_sub = u_sub.to_numpy()
        v_sub = v_sub.to_numpy()
        speed_sub = speed_sub.to_numpy()
        speed_flat = speed_sub.flatten()
        start_points = np.array([lon2d_sub.flatten(),lat2d_sub.flatten()]).T

    
       # plot a quiver with axes.quiver() from that point. Make the quiver small enough so that only the arrow is visible
        levels = np.linspace(vmin, vmax, 20)
        scalar = ax.contourf(lon2d, lat2d, speed_2d.sel(time = date[i], depth = z, method='nearest'), 
                             levels = levels, transform = ccrs.PlateCarree(), cmap = cmap, zorder=-1)
        ax.set_title('Speed currents at ' + str(date[i]) + '\n depth = ' + str(z) + ' [m]', fontsize = 35, y=1.02)
        norm= Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
        sm.set_array([])
    
        # make streamplot with axes.streamplot() with no arrows that is traced backward from a given point
        scale = 0.1/np.max(speed_sub)
        
        uu = u_2d.sel(time = date[i], depth = z, method='nearest').to_numpy()
        vv = v_2d.sel(time = date[i], depth = z, method='nearest').to_numpy()
        
        for j in range(start_points.shape[0]):
            ax.streamplot(lon2d,lat2d,uu,vv,
                color='black',
                start_points=np.array([start_points[j,:]]),
                minlength=0.2*speed_flat[j],
                maxlength=0.22*speed_flat[j],
                integration_direction='backward',
                density=20,
                arrowsize=0.0,
                linewidth=3.)
    
    # plot a quiver with axes.quiver() from that point. Make the quiver small enough so that only the arrow is visible
        ax.quiver(lon2d_sub,lat2d_sub,u_sub/speed_sub, v_sub/speed_sub,scale=30) 
    
    for ax in axes.flat:
        depths = (50, 200, 1000)

        ax.coastlines(resolution="10m", linewidths=0.7)
        ax.add_feature(cfeature.LAND.with_scale("10m"),
                   edgecolor='lightgray',facecolor='lightgray',
                   zorder=0)
        
        lines = ax.contour(lons, lats, -bathy, levels = depths, colors='black', linewidths = .75)
        ax.clabel(lines, inline=2, fontsize=25, colors = 'black')
    
        ax.vlines(-70.8, lats.min(), lats.max(), linestyle = (5, (10, 3)), linewidth=3, color = 'red')
        
        ax.tick_params(axis = "both", labelsize = 30)
        ax.set_xticks(lons, crs=ccrs.PlateCarree())
        ax.set_xticks(lats, crs=ccrs.PlateCarree())
        ax.axes.axis("tight")
    
        gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5,
                      linestyle='--',draw_labels=True, zorder = 1)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'fontsize': 30}
        gl.ylabel_style = {'fontsize': 30}
        
        if ax != axes.flat[0] :
            gl.ylabels_left = False
        
        fig.tight_layout(pad = 5)
            
    
    return (fig, axes, sm, z)

#%% LOAD MONTHLY CMEMS DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/CMEMS/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/chlor/enhancements/"

#monthly data
ds = xr.open_dataset(path + 'cmems_grep.nc')
#%%ARRAY
# #choose ARRAY coordinate
lat_range = slice(38, 41)
lon_range = slice(-72.6, -68)

ds = ds.sel(longitude=lon_range)
ds = ds.sel(latitude=lat_range)
#%%
#lat, lon, bathy
lons, lats = ds['longitude'], ds['latitude']
bathy = xr.open_dataset(path + "bathymetry_interpolated_cmems.nc")

bathy = bathy['elevation']
bathy= bathy.sel(latitude = lats, longitude = lons)

#%%
colors = [tuple(np.array((235,245,255))/255), # light blue
          tuple(np.array((157,218,247))/255), # blue
          tuple(np.array((72,142,202))/255),  # dark blue
          tuple(np.array((73,181,70))/255),   # green
          tuple(np.array((250,232,92))/255),  # yellow
          tuple(np.array((245,106,41))/255),  # orange
          tuple(np.array((211,31,40))/255),   # red
          tuple(np.array((146,21,25))/255)]   # dark red
mycmap = LinearSegmentedColormap.from_list('WhiteBlueGreenYellowRed', colors, N=100)



#%%
lon_array = -70.8
ds_array = ds.sel(longitude = lon_array, method='nearest') #-70.75

theta = ds_array['thetao_mean']
mld = ds_array['mlotst_mean']
sal = ds_array['so_mean']
depth = ds_array['depth']
u = ds_array['uo_mean']
v = ds_array['vo_mean']
speed = np.sqrt(u*u + v*v)

u_2d = ds['uo_mean']
v_2d = ds['vo_mean']
speed_2d = np.sqrt(u_2d*u_2d + v_2d*v_2d)
#%%
date_18 = np.array(('2018-10-31', '2018-11-01', '2018-11-04', '2018-11-07'))
date_19_j = np.array(('2019-06-22', '2019-06-23'))
date_19_m = np.array(('2019-05-19', '2019-05-21', '2019-05-22', '2019-05-24'))
date_19_a = np.array(('2019-04-24', '2019-04-27'))
date_12_a =  np.array(('2012-04-08', '2012-04-13'))

#%%2018
fig, axes, sm = set_plot_theta(4, date_18, vmin = 10, vmax = 24)
set_cbar(fig, sm, '[$^{\circ}C$]')
fig.savefig(plot_path + 'tetha_2018_nov.png', bbox_inches='tight')
#%%
fig, axes, sm = set_plot_sal(4, date_18, vmin = 32.6, vmax = 36.6)
    
set_cbar(fig, sm, '[PSU]')
fig.savefig(plot_path + 'sal_2018_nov.png', bbox_inches='tight')

#%%CURRENTS
fig, axes, sm = set_plot_currents(4, date_18, mycmap, vmin = 0.01, vmax = 1.3)
set_cbar(fig, sm, '[m/s]')
fig.savefig(plot_path + 'currents_2018_nov.png', bbox_inches='tight')

#%%JUNE 2019
fig, axes, sm = set_plot_theta(2, date_19_j, vmin = 6, vmax = 24)
set_cbar(fig, sm, '[$^{\circ}C$]')
fig.savefig(plot_path + 'tetha_2019_june.png', bbox_inches='tight')

fig, axes, sm = set_plot_sal(2,  date_19_j, vmin = 32.6, vmax = 36.6)
set_cbar(fig, sm, '[PSU]')
fig.savefig(plot_path + 'sal_2019_jun.png', bbox_inches='tight')

fig, axes, sm = set_plot_currents(2,  date_19_j, mycmap, vmin = 0.01, vmax = 1.3)
set_cbar(fig, sm, '[m/s]')
fig.savefig(plot_path + 'currents_2019_jun.png', bbox_inches='tight')

#%%may 2019
fig, axes, sm = set_plot_theta(4, date_19_m, vmin = 6, vmax = 24)
set_cbar(fig, sm, '[$^{\circ}C$]')
fig.savefig(plot_path + 'tetha_2019_may.png', bbox_inches='tight')

fig, axes, sm = set_plot_sal(4,  date_19_m, vmin = 32.6, vmax = 36.6)
set_cbar(fig, sm, '[PSU]')
fig.savefig(plot_path + 'sal_2019_may.png', bbox_inches='tight')

fig, axes, sm = set_plot_currents(4,  date_19_m, mycmap, vmin = 0.01, vmax = 1.3)
set_cbar(fig, sm, '[m/s]')
fig.savefig(plot_path + 'currents_2019_may.png', bbox_inches='tight')

#%%APRIL 2019
fig, axes, sm = set_plot_theta(2, date_19_a, vmin = 6, vmax = 24)
set_cbar(fig, sm, '[$^{\circ}C$]')
fig.savefig(plot_path + 'tetha_2019_apr.png', bbox_inches='tight')

fig, axes, sm = set_plot_sal(2,  date_19_a, vmin = 32.6, vmax = 36.6)
set_cbar(fig, sm, '[PSU]')
fig.savefig(plot_path + 'sal_2019_apr.png', bbox_inches='tight')

fig, axes, sm = set_plot_currents(2,  date_19_a, mycmap, vmin = 0.01, vmax = 1.3)
set_cbar(fig, sm, '[m/s]')
fig.savefig(plot_path + 'currents_2019_apr.png', bbox_inches='tight')

#%%APRIL 2012
fig, axes, sm = set_plot_theta(2, date_12_a, vmin = 6, vmax = 24)
set_cbar(fig, sm, '[$^{\circ}C$]')
fig.savefig(plot_path + 'tetha_2012_apr.png', bbox_inches='tight')

fig, axes, sm = set_plot_sal(2,  date_12_a, vmin = 32.6, vmax = 36.6)
set_cbar(fig, sm, '[PSU]')
fig.savefig(plot_path + 'sal_2012_apr.png', bbox_inches='tight')

fig, axes, sm = set_plot_currents(2,  date_12_a, mycmap, vmin = 0.01, vmax = 1.3)
set_cbar(fig, sm, '[m/s]')
fig.savefig(plot_path + 'currents_2012_apr.png', bbox_inches='tight')

#%%2019 2D CURRENTS MAP
fig, axes, sm, z = set_plot_currents_2d(2, date_19_a, mycmap, vmin = 0.01, vmax = 1., z=20)
set_cbar(fig, sm, '[m/s]')
fig.savefig(plot_path + 'currents_2019_apr_z=' + str(z) + '.png', bbox_inches='tight')

fig, axes, sm, z = set_plot_currents_2d(4, date_19_m, mycmap, vmin = 0.01, vmax = 1., z=20)
set_cbar(fig, sm, '[m/s]')
fig.savefig(plot_path + 'currents_2019_may_z=' + str(z) + '.png', bbox_inches='tight')

fig, axes, sm, z = set_plot_currents_2d(2, date_19_j, mycmap, vmin = 0.01, vmax = 1., z=20)
set_cbar(fig, sm, '[m/s]')
fig.savefig(plot_path + 'currents_2019_jun_z=' + str(z) + '.png', bbox_inches='tight')

#%%
fig, axes, sm, z = set_plot_currents_2d(4, date_18, mycmap, vmin = 0.01, vmax = 1., z=20)
set_cbar(fig, sm, '[m/s]')
fig.savefig(plot_path + 'currents_2018_nov_z=' + str(z) + '.png', bbox_inches='tight')
