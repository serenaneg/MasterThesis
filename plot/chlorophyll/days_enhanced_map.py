#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 19:28:06 2023

@author: serena
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm, ListedColormap
import matplotlib.colors as colors
import cmocean

#%%FUNCTIONS
def set_plot(lats, lons, depth, ncols):
    depths = (50, 75, 100, 200, 500, 1000, 2500, 3000)
    
    fig, axes = plt.subplots(ncols = ncols, nrows = 1, subplot_kw = dict(projection = ccrs.PlateCarree()), 
                             figsize = [19*ncols,12])
    
    font_size = 30
    for ax in axes.flat:
        ax.coastlines(resolution="10m", linewidths=0.7)
        ax.add_feature(cfeature.LAND.with_scale("10m"),
                   edgecolor='lightgray',facecolor='lightgray',
                   zorder=0)
        
        lines = ax.contour(lons, lats, -depth, levels = depths, colors='black', linewidths = .75)
        ax.clabel(lines, inline=2, fontsize=25, colors = 'black')
    
        ax.vlines(-70.8, lats.min(), lats.max(), linestyle = (5, (10, 3)), linewidth=3, color = 'red')
        
        ax.tick_params(axis = "both", labelsize = 25)
        ax.set_xticks(lons, crs=ccrs.PlateCarree())
        ax.set_xticks(lats, crs=ccrs.PlateCarree())
        ax.axes.axis("tight")
    
        gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5,
                      linestyle='--',draw_labels=True, zorder = 1)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'fontsize': font_size}
        gl.ylabel_style = {'fontsize': font_size}
        
        if ax != axes.flat[0] :
            gl.ylabels_left = False
        
        fig.tight_layout(pad = 5)
           
    return (fig, axes)

def set_cbar(fig, c, title):    
    cbar = plt.colorbar(c, format='%.1f', spacing='proportional', ax=axes, shrink=0.9, pad = 0.03,
                        orientation = 'vertical', location = "right")
    cbar.set_label( label = title, fontsize = 35, y = 0.5)
    cbar.ax.tick_params(which='minor', size=25, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=35, width=1, color='k', direction='in', labelsize = 30)
    # fig.subplots_adjust(bottom=0.25)
    # cbar.ax.set_position([0.2, 0.08, 0.6, 0.08])
    return cbar

#%% LOAD MONTHLY CMEMS DATA
path = "/home/serena/Scrivania/Magistrale/thesis/data/MODIS_INTERPOLATED_DATA/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/chlor/enhancements/"

#monthly data
ds = xr.open_dataset(path + 'MODIS_chl_1D_nomissing.nc')

#%%ARRAY
# #choose ARRAY coordinate
lat_range = slice(41, 38)
lon_range = slice(-72.6, -68)

ds = ds.sel(lon=lon_range)
ds = ds.sel(lat=lat_range)

#%%
#lat, lon, bathy
lons, lats = ds['lon'], ds['lat']
bathy = xr.open_dataset(path + "bathymetry_interpolated.nc")

depth = bathy['elevation']
depth= depth.sel(lat = lats, lon = lons)

#%% CHLOROPHYLL
chl = ds['chlor_a']

#%%cmap
top = cm.get_cmap('YlGnBu_r', 128) # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)# combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')

#%% PLOT PARAMS
vmin = 0.1
vmax = 20
levels = np.linspace(vmin, vmax, 100)
cmap = orange_blue

#%%
date = np.array(('2019-04-24', '2019-04-27'))

fig, axes = set_plot(lats, lons, depth, 2)

for i, ax in enumerate(axes.flat):
    levels = np.linspace(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max(), 100)
    scalar = ax.contourf(lons, lats, chl.sel(time = date[i]),  norm = 'log', levels = levels,
                         extend = 'both', cmap=cmap, zorder = 1)

    ax.set_title('Chlor-a at ' + str(date[i]), fontsize = 35, y=1.02)
    norm= LogNorm(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])

set_cbar(fig, sm, 'Chlorophyll-a concentration')
fig.savefig(plot_path + 'chl_2019_apr.png', bbox_inches='tight')

#%%
date = np.array(('2012-04-08', '2012-04-13'))

fig, axes = set_plot(lats, lons, depth, 2)

for i, ax in enumerate(axes.flat):
    levels = np.linspace(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max(), 100)
    scalar = ax.contourf(lons, lats, chl.sel(time = date[i]),  norm = 'log', levels = levels,
                         extend = 'both', cmap=cmap, zorder = 1)

    ax.set_title('Chlor-a at ' + str(date[i]), fontsize = 35, y=1.02)
    norm= LogNorm(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])

set_cbar(fig, sm, 'Chlorophyll-a concentration')
fig.savefig(plot_path + 'chl_2012_apr.png', bbox_inches='tight')

#%%
date = np.array(('2019-05-19', '2019-05-21', '2019-05-22', '2019-05-24'))

fig, axes = set_plot(lats, lons, depth, 4)

for i, ax in enumerate(axes.flat):
    levels = np.linspace(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max(), 100)
    scalar = ax.contourf(lons, lats, chl.sel(time = date[i]),  norm = 'log', levels = levels,
                         extend = 'both', cmap=cmap, zorder = 1)

    ax.set_title('Chlor-a at ' + str(date[i]), fontsize = 35, y=1.02)
    norm= LogNorm(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])

set_cbar(fig, sm, 'Chlorophyll-a concentration')
fig.savefig(plot_path + 'chl_2019_may.png', bbox_inches='tight')

#%%
date = np.array(('2019-06-22', '2019-06-23'))

fig, axes = set_plot(lats, lons, depth, 2)

for i, ax in enumerate(axes.flat):
    levels = np.linspace(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max(), 100)
    scalar = ax.contourf(lons, lats, chl.sel(time = date[i]),  norm = 'log', levels = levels,
                         extend = 'both', cmap=cmap, zorder = 1)

    ax.set_title('Chlor-a at ' + str(date[i]), fontsize = 35, y=1.02)
    norm= LogNorm(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])

set_cbar(fig, sm, 'Chlorophyll-a concentration')
fig.savefig(plot_path + 'chl_2019_june.png', bbox_inches='tight')
#%%
date = np.array(('2019-07-13', '2019-07-15'))

fig, axes = set_plot(lats, lons, depth, 2)

for i, ax in enumerate(axes.flat):
    levels = np.linspace(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max(), 100)
    scalar = ax.contourf(lons, lats, chl.sel(time = date[i]),  norm = 'log', levels = levels,
                         extend = 'both', cmap=cmap, zorder = 1)

    ax.set_title('Chlor-a at ' + str(date[i]), fontsize = 35, y=1.02)
    norm= LogNorm(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])

set_cbar(fig, sm, 'Chlorophyll-a concentration')
fig.savefig(plot_path + 'chl_2019_july.png', bbox_inches='tight')
#%%
date = np.array(('2018-10-31', '2018-11-01', '2018-11-04', '2018-11-07'))

fig, axes = set_plot(lats, lons, depth, 4)

for i, ax in enumerate(axes.flat):
    levels = np.linspace(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max(), 100)
    scalar = ax.contourf(lons, lats, chl.sel(time = date[i]),  norm = 'log', levels = levels,
                         extend = 'both', cmap=cmap, zorder = 1)

    ax.set_title('Chlor-a at ' + str(date[i]), fontsize = 35, y=1.02)
    norm= LogNorm(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])

set_cbar(fig, sm, 'Chlorophyll-a concentration')
fig.savefig(plot_path + 'chl_2018_nov.png', bbox_inches='tight')

#%%
date = np.array(('2016-05-11', '2016-05-12', '2016-05-15'))

fig, axes = set_plot(lats, lons, depth, 3)

for i, ax in enumerate(axes.flat):
    levels = np.linspace(chl.sel(time = date[i]).min(), chl.sel(time = date[i]).max(), 100)
    scalar = ax.contourf(lons, lats, chl.sel(time = date[i]),  norm = 'log', levels = levels,
                         extend = 'both', cmap=cmap, zorder = 1)

    ax.set_title('Chlor-a at ' + str(date[i]), fontsize = 35, y=1.02)
    norm= LogNorm(vmin, vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap = scalar.cmap)
    sm.set_array([])

set_cbar(fig, sm, 'Chlorophyll-a concentration')
fig.savefig(plot_path + 'chl_2016_may.png', bbox_inches='tight')