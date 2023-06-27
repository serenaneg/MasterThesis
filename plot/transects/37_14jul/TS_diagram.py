#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:26:40 2023

@author: serena
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean
import gsw
import scipy.stats as ss
import datetime
import gsw as gsw               # Python seawater package
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm


def set_cbar(fig, c, title, ax):   
    # skip = (slice(None, None, 2))
    cbar = plt.colorbar(c, format='%.0f', spacing='proportional', ax=ax, shrink=0.9, pad = 0.01,
                        orientation = 'vertical', location = "right")
    cbar.set_label(label = title, fontsize = 25, y = 0.5, labelpad = 20)
    cbar.ax.tick_params(which='minor', size=5, width=1, color='k', direction='in')
    cbar.ax.tick_params(which='major', size=10, width=1, color='k', direction='in', labelsize = 20)
    # fig.subplots_adjust(bottom=0.25)
    # cbar.ax.set_position([0.2, 0.08, 0.6, 0.08])
    return cbar

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


path = "/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/"
bathy_path = "/home/serena/Scrivania/Magistrale/thesis/data/"

plot_path = "/home/serena/Scrivania/Magistrale/thesis/plot/transects/2019/"

file = path + 'ctd_tn_withlocation.csv'
ctd = pd.read_csv(file)

#%%CHOOSE DAY

ctd['day'] = pd.to_datetime(ctd['day'])
# Filter rows where day is 14
tn37 = ctd[(ctd['day'].dt.day == 14)]
#%%
# Load data from first cast
temperature = tn37.temperature # Load temperature data
salinity    = tn37.salinity   # Load salinity data
depth       = tn37.depth         # Load depth information

# Define the min / max values for plotting isopycnals
t_min = temperature.min() - 1
t_max = temperature.max() + 1
s_min = salinity.min() - 1
s_max = salinity.max() + 1

# Calculate how many gridcells we need in the x and y dimensions
xdim = np.array(np.ceil(s_max - s_min)/0.1, dtype = 'int64') #return the smallest integer
ydim = np.array(np.ceil(t_max-t_min), dtype = 'int64') 
dens = np.zeros((int(ydim),int(xdim)))

# Create temp and salt vectors of appropiate dimensions
ti = np.linspace(0,ydim,ydim)+t_min
si = np.linspace(1,xdim,xdim)*0.1+s_min

# Loop to fill in grid with densities
for j in range(0,int(ydim)):
    for i in range(0, int(xdim)):
        dens[j,i]=gsw.rho(si[i],ti[j],0)

# Subtract 1000 to convert to sigma-t
dens = dens - 1000

spicy = np.zeros((int(ydim),int(xdim)))
#spiciness
for j in range(0,int(ydim)):
    for i in range(0, int(xdim)):
        spicy[j,i]=gsw.spiciness0(si[i],ti[j])

#%%
top = cm.get_cmap('YlGnBu_r', 128)  # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)

top = truncate_colormap(top, 0.12, 1) 
bottom = truncate_colormap(bottom, 0.18, 1)  # combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))  # create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')
#%%
# plot the data
mycmap = orange_blue
# plt.get_cmap('RdYlBu_r')


# mycmap = truncate_colormap(cmap, 0, 0.7)

norm = colors.BoundaryNorm(np.linspace(0, 280, 15), mycmap.N)

fig, ax = plt.subplots(1,1, dpi = 200, figsize = ([12,10]))

lines = ax.contour(si, ti, dens, colors='lightslategrey')
ax.clabel(lines, fontsize=15, inline=1, fmt='%.2f')

lines2 = ax.contour(si, ti, spicy, linestyles = 'dotted', colors='grey')
ax.clabel(lines2, fontsize=15, inline=1, fmt='%.2f')

scalar = ax.scatter(salinity, temperature, c=depth, cmap = mycmap, norm = norm, zorder = -1)

ax.xaxis.set_tick_params(labelsize = 20)
ax.yaxis.set_tick_params(labelsize = 20) 

ax.set_xlim(31.5, 36.5)
ax.set_xlabel('Salinity [PSU]', fontsize = 25)
ax.set_ylabel('Temperature [$^\circ$C]', fontsize = 25)
ax.set_title('T-S diagram Transect 37, July 14 2019', fontsize = 25)
ax.set_facecolor("whitesmoke")
set_cbar(fig, scalar, 'Depth [m]', ax)

fig.tight_layout() 
fig.savefig(plot_path + 'ts_diagram_14july19.png')





