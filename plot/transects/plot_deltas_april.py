#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:17:02 2024

@author: serena
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import scipy.stats
import matplotlib as mpl 
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

#colorbar
top = cm.get_cmap('YlGnBu_r', 128)  # r means reversed version
bottom = cm.get_cmap('YlOrBr', 128)  # combine it all
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))  # create a new colormaps with a name of OrangeBlue
orange_blue = ListedColormap(newcolors, name='OrangeBlue')

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

path_deltas = '/home/serena/Scrivania/Magistrale/thesis/deltas/'
days = [17, 19, 21,  23, 25, 27] #april
#%%APRIL
# data_delta1 = pd.read_csv(path_deltas + 'april_266.csv', sep = ' ')
# data_delta2 = pd.read_csv(path_deltas + 'april_265.csv', sep = ' ')
# data_delta3 = pd.read_csv(path_deltas + 'april_264.csv', sep = ' ')

data_delta1 = pd.read_csv(path_deltas + 'april_266_final.csv', sep = ' ')
data_delta2 = pd.read_csv(path_deltas + 'april_265_final.csv', sep = ' ')
data_delta3 = pd.read_csv(path_deltas + 'april_264_final.csv', sep = ' ')

data_delta = pd.concat((data_delta1, data_delta2, data_delta3), axis = 0)

# data_delta1_pon = pd.read_csv(path_deltas + 'april_266_pon.csv', sep = ' ')
# data_delta2_pon = pd.read_csv(path_deltas + 'april_265_pon.csv', sep = ' ')
# data_delta3_pon = pd.read_csv(path_deltas + 'april_264_pon.csv', sep = ' ')

# data_delta_pon = pd.concat((data_delta1_pon, data_delta2_pon, data_delta3_pon), axis = 0)


#%%
normalize_no3 = (data_delta['no3'] - np.min(data_delta['no3'])) / (np.max(data_delta['no3']) - np.min(data_delta['no3']))
normalize_chl = (data_delta['chl'] - np.min(data_delta['chl'])) / (np.max(data_delta['chl']) - np.min(data_delta['chl']))
sns.set_style("darkgrid")

fig, (ax,ax2, ax3) = plt.subplots(ncols=3, figsize=(52, 16))
fig.subplots_adjust(wspace=0.05)

new_colors = truncate_colormap(plt.get_cmap('gist_rainbow'), 0.86, 0)

gfg = sns.scatterplot(data_delta, x=data_delta['lat'], y=data_delta['depth'], hue = data_delta['no3'], palette=new_colors,
                ax=ax, s = 1000, alpha = .7)
ax.invert_xaxis()
ax.invert_yaxis()
norm = plt.Normalize(np.min(data_delta['no3']), np.max(data_delta['no3']))
ax.yaxis.set_tick_params(labelsize = 40)
ax.xaxis.set_tick_params(labelsize = 40)
ax.tick_params(rotation=0)
ax.set_ylabel('Depth', fontsize = 40)
ax.set_xlabel('Latitude $^\circ$N', fontsize = 40)
ax.vlines(40.178, *ax.get_ylim(), colors = 'k', linestyle = '--',linewidth = 3, zorder = -1)
ax.set_title('Nitrate (NO$^{-}_{3}$)', fontsize = 40, y = 1.02)
sm = plt.cm.ScalarMappable(cmap=new_colors, norm=norm)
sm.set_array([])
ax.get_legend().remove()
cbar = ax.figure.colorbar(sm, ax=ax)
cbar.ax.tick_params(labelsize = 40)
cbar.set_label(label = '[$\mu$mol/L]', size = 40)

norm2 = plt.Normalize(0, np.max(data_delta['chl']))
sns.scatterplot(data_delta, x=data_delta['lat'], y=data_delta['depth'], hue = data_delta['chl'], 
                palette=new_colors, ax=ax2, s = 1000, alpha = .7, hue_norm = norm2)
ax2.yaxis.tick_right()
#ax2.set_xticks(ax.get_xticks()[0::2])
ax2.tick_params(rotation=0)
ax2.invert_xaxis()
ax2.invert_yaxis()
ax2.yaxis.set_tick_params(labelsize = 0)
ax2.xaxis.set_tick_params(labelsize = 40)
ax2.set_xlabel('Latitude $^\circ$N', fontsize = 40)
ax2.set_ylabel('', fontsize = 0)
ax2.vlines(40.178, *ax2.get_ylim(), colors = 'k', linestyle = '--', linewidth = 3, zorder = -1)
ax2.set_title('Chlorophyll', fontsize = 40, y = 1.02)
sm2 = plt.cm.ScalarMappable(cmap=new_colors, norm=norm2)
sm2.set_array([])
ax2.get_legend().remove()
cbar2 = ax2.figure.colorbar(sm2, ax=ax2)
cbar2.ax.tick_params(labelsize = 40)
cbar2.set_label(label = '[mg/m$^{3}$]', size = 40)

norm3 = plt.Normalize(np.min(data_delta['pon']), np.max(data_delta['pon']))
sns.scatterplot(data_delta, x=data_delta['lat'], y=data_delta['depth'], hue = data_delta['pon'], 
                palette=new_colors, ax=ax3, s = 1000, alpha = .7, hue_norm = norm3)
ax3.yaxis.tick_right()
ax3.tick_params(rotation=0)
ax3.invert_xaxis()
ax3.invert_yaxis()
ax3.yaxis.set_tick_params(labelsize = 0)
ax3.xaxis.set_tick_params(labelsize = 40)
ax3.set_xlabel('Latitude $^\circ$N', fontsize = 40)
ax3.set_ylabel('', fontsize = 0)
ax3.vlines(40.178, *ax2.get_ylim(), colors = 'k', linestyle = '--',linewidth = 3, zorder = -1)
ax3.set_title('PON', fontsize = 40, y = 1.02)
sm3 = plt.cm.ScalarMappable(cmap=new_colors, norm=norm3)
sm3.set_array([])
ax3.get_legend().remove()
cbar3 = ax3.figure.colorbar(sm3, ax=ax3)
cbar3.ax.tick_params(labelsize = 40)
cbar3.set_label(label = '[$\mu$mol/L]', size = 40)

fig.tight_layout(pad = 3)
fig.suptitle("Along isopycnals properties change: [26.3, 26.65]  $\sigma$. April 2018", fontsize = 40, y = 1.05)
fig.savefig(path_deltas + 'along_isopycnals_linear_April_shallower.png', bbox_inches='tight')
#%%PROFILES
sns.set_style("whitegrid")

fig, ax = plt.subplots(3, 6, dpi=50, figsize=([60, 45]))

for i in range(len(days)):
    sel_263 = data_delta1[data_delta1['day'] == days[i]]
    sel_26 = data_delta3[data_delta3['day'] == days[i]]
    sel_2615 = data_delta2[data_delta2['day'] == days[i]]
    
    ax[0, i].plot(sel_263['no3'], sel_263['depth'], c = 'b', markersize = 20, linestyle='-', marker='o')
    ax[1, i].plot(sel_2615['no3'], sel_2615['depth'], c = 'b', markersize = 20, linestyle='-', marker='o')
    ax[2, i].plot(sel_26['no3'], sel_26['depth'], c = 'b', markersize = 20, linestyle='-', marker='o')
    
    for j in range(0, 3):
        #ax[j, i].invert_yaxis() 
        ax[j, i].xaxis.set_tick_params(labelcolor = 'b', labelsize = 30)
        ax[j, i].yaxis.set_tick_params(labelsize = 30) 
        
        if i == 0:
            ax[j, i].set_ylabel('Depth [m]', fontsize = 35)
            
        if j == 2:
            ax[j, i].set_xlabel('NO$^{-}_{3}$ [$\mu$mol L$^{-1}$]', c = 'b', fontsize = 35)

        
    ax22 = ax[0, i].twiny()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax22.set_xlabel('Chlorophyll [mg m$^{-3}$]', color=color, fontsize = 35)
    ax22.invert_yaxis() 
    ax22.semilogx(sel_263['chl'], sel_263['depth'], c = color, markersize = 20, linestyle='-', marker='o')
    ax22.xaxis.set_tick_params(labelcolor = color, labelsize = 30)
    ax22.yaxis.set_tick_params(labelsize = 0) 
    
    ax33 = ax[1, i].twiny()  # instantiate a second axes that shares the same x-axis
    ax33.invert_yaxis() 
    ax33.semilogx(sel_2615['chl'], sel_2615['depth'], c = color, markersize = 20, linestyle='-', marker='o')
    ax33.xaxis.set_tick_params(labelcolor = color, labelsize = 30)
    ax33.yaxis.set_tick_params(labelsize = 0) 
    
    ax44 = ax[2, i].twiny()  # instantiate a second axes that shares the same x-axis
    ax44.invert_yaxis() 
    ax44.semilogx(sel_26['chl'], sel_26['depth'], c = color, markersize = 20, linestyle='-', marker='o')
    ax44.xaxis.set_tick_params(labelcolor = color, labelsize = 30)
    ax44.yaxis.set_tick_params(labelsize = 0) 

    fig.tight_layout(pad = 3)
    #fig.suptitle('Along isopycnlas profiles July 2019, [26.3, 26.15, 26 $\sigma$]', fontsize = 50, y = 1.01)
    fig.suptitle('Along isopycnlas profiles April 2018, [26.6, 26.4, 26.5 $\sigma$]', fontsize = 50, y = 1.01)
    #fig.savefig(path_deltas + "profiles_linear_july.png", bbox_inches='tight', dpi = 50)
    fig.savefig(path_deltas + "profiles_linear_april.png", bbox_inches='tight', dpi = 50)


#%%CALCULATE DELTA
bottom_no3 = []
surface_no3 = []
bottom_chl = []
surface_chl = []
bottom_pon = []
surface_pon = []

for i in range(len(days)):
    pick_day = data_delta['day'] == days[i]
    sel = data_delta[pick_day]
    
    for k in range(len(sel)):
        lat = sel.iloc[k]['lat']

        
        if i == 5:#!!ONLY FOR APRIL
            if 40.7 >= lat >= 40.21:
                bottom_no3.append(sel.iloc[k][['no3', 'day', 'lat', 'depth']])
                bottom_chl.append(sel.iloc[k][['chl', 'day', 'lat']])
                bottom_pon.append(sel.iloc[k][['pon', 'day', 'lat']])

            else:
                surface_no3.append(sel.iloc[k][['no3', 'day', 'lat', 'depth']])
                surface_chl.append(sel.iloc[k][['chl', 'day', 'lat']])
                surface_pon.append(sel.iloc[k][['pon', 'day', 'lat']])
        
        elif 40.7 >= lat >= 40.17:
            #APRIL elif 40.7 >= lat >= 40.17: #JULY if 40.35 >= lat >= 40.05:
            bottom_no3.append(sel.iloc[k][['no3', 'day', 'lat', 'depth']])
            bottom_chl.append(sel.iloc[k][['chl', 'day', 'lat']])
            bottom_pon.append(sel.iloc[k][['pon', 'day', 'lat']])

        else:
            surface_no3.append(sel.iloc[k][['no3', 'day', 'lat', 'depth']])
            surface_chl.append(sel.iloc[k][['chl', 'day', 'lat']]) 
            surface_pon.append(sel.iloc[k][['pon', 'day', 'lat']])


bottom_no3 = np.array(bottom_no3)
surface_no3 = np.array(surface_no3)
bottom_chl = np.array(bottom_chl)
surface_chl = np.array(surface_chl)
bottom_pon = np.array(bottom_pon)
surface_pon = np.array(surface_pon)
#%%
#dday = [19, 27]
results_bottom = []
resuts_surface = []
for i, t in enumerate(days):
    pick_day = bottom_no3[:, 1] == t
    sel_bottom = bottom_no3[pick_day]
    sel_surface = surface_no3[pick_day]
    
    df_bottom = pd.DataFrame({
        'Lat max': np.max(sel_bottom[:, 2]),
        'lat min': np.min(sel_bottom[:, 2]),
        'depth max': np.max(sel_bottom[:, 3]),
        'depth min' : np.min(sel_bottom[:, 3]),
        'day' : t
    }, index=[0])

    # print('BOTTOM BOX DAY' + str(t))
    # print('lat max ' + str(np.max(sel_bottom[:, 2])))
    # print('lat min ' + str(np.min(sel_bottom[:, 2])))
    # print('depth max ' + str(np.max(sel_bottom[:, 3])))
    # print('depth min ' + str(np.min(sel_bottom[:, 3])))
    
    df_surface = pd.DataFrame({
        'Lat max': np.max(sel_surface[:, 2]),
        'lat min': np.min(sel_surface[:, 2]),
        'depth max': np.max(sel_surface[:, 3]),
        'depth min' : np.min(sel_surface[:, 3]),
        'day' : t
    }, index=[0])

    # print('SURFACE BOX')
    # print('lat max ' + str(np.max(sel_surface[:, 2])))
    # print('lat min ' + str(np.min(sel_surface[:, 2])))
    # print('depth max ' + str(np.max(sel_surface[:, 3])))
    # print('depth min ' + str(np.min(sel_surface[:, 3])) + '\n')
    
    results_bottom.append(df_bottom)
    resuts_surface.append(df_surface)
    
bottom = pd.concat(results_bottom)
surface = pd.concat(resuts_surface)

with pd.ExcelWriter(path_deltas + 'bottom_box_april.xlsx') as writer:
    bottom.to_excel(writer)
    
with pd.ExcelWriter(path_deltas + 'surface_box_april.xlsx') as writer:
    surface.to_excel(writer)
#%%
delta_no3 = surface_no3 - bottom_no3
delta_pon = surface_pon - bottom_pon
delta_chl = surface_chl - bottom_chl

#if surface_chl[i] < 0 take abs

delta_no3 = np.vstack((delta_no3[:, 0], bottom_no3[:, 1]))
delta_no3 = delta_no3.T

delta_chl = np.vstack((delta_chl[:, 0], bottom_chl[:, 1]))
delta_chl = delta_chl.T

delta_pon = np.vstack((delta_pon[:, 0], bottom_pon[:, 1]))
delta_pon = delta_pon.T
#%%CORRELATION COEFFICIENT REMOVING POSITIVE DELTA NITRATE
positive_indices = np.where(delta_no3 < 0)[0]
#POSTITVE DELTA MAILY 11 AND 14, POSSIBLY SOME EFFECTS RELATED TO MIXING OF THE STREAMER

delta_no3_nopos = delta_no3[positive_indices]
delta_chl_nopos = delta_chl[positive_indices]
delta_pon_nopos = delta_pon[positive_indices]

positive_indices_chl = np.where(delta_chl_nopos[:, 0] > 0)[0]
delta_no3_corr = delta_no3_nopos[positive_indices_chl]
delta_chl_corr = delta_chl_nopos[positive_indices_chl]

positive_indices_pon = np.where(delta_pon_nopos[:, 0] > 0)[0]
delta_pon_corr = delta_pon_nopos[positive_indices_pon]
delta_no3_corr2 = delta_no3_nopos[positive_indices_pon]
delta_chl_corr2 = delta_chl_nopos[positive_indices_pon]

carbon = np.array([val * (106/16) for val in delta_no3_corr[:, 0]])

stats_263 = scipy.stats.linregress(delta_no3_corr[:, 0], delta_chl_corr[:, 0])
print(f'Slope26.3 = {stats_263.slope}')
print(f'Intercept = {stats_263.intercept}')
print(f'R_squared = {stats_263.rvalue}')
print(f'p-value = {stats_263.pvalue} \n')

stats_pon = scipy.stats.linregress(delta_no3_corr2[:, 0], delta_pon_corr[:, 0])
print(f'Slope26.3 = {stats_pon.slope}')
print(f'Intercept = {stats_pon.intercept}')
print(f'R_squared = {stats_pon.rvalue}')
print(f'p-value = {stats_pon.pvalue} \n')

expect_chlor = np.abs(carbon / 50)

#underly perturbated day 
noPertubation = np.where(delta_chl_corr[:, 1] != 27)[0]

delta_no3_NoPert =delta_no3_corr[noPertubation]
delta_chl_NoPert =delta_chl_corr[noPertubation]
delta_pon_NoPert =delta_pon_corr[np.where(delta_pon_corr[:, 1] != 27)[0]]
delta_no3_NoPert_pon =delta_no3_corr2[np.where(delta_no3_corr2[:, 1] != 27)[0]]

stats_263_Pert = scipy.stats.linregress(delta_no3_NoPert[:, 0], delta_chl_NoPert[:, 0])
print('Regression without 27th april')
print(f'Slope26.3 = {stats_263_Pert.slope}')
print(f'Intercept = {stats_263_Pert.intercept}')
print(f'R_squared = {stats_263_Pert.rvalue}')
print(f'p-value = {stats_263_Pert.pvalue} \n')

stats_pon_Pert = scipy.stats.linregress(delta_no3_NoPert_pon[:, 0], delta_pon_NoPert[:, 0])
print(f'Slope26.3 = {stats_pon_Pert.slope}')
print(f'Intercept = {stats_pon_Pert.intercept}')
print(f'R_squared = {stats_pon_Pert.rvalue}')
print(f'p-value = {stats_pon_Pert.pvalue} \n')
#%%
def correlation_coeff(x, y):
    # Calculate mean of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate standard deviation of x and y
    std_x = np.std(x)
    std_y = np.std(y)
    
    # Calculate the sum of (x_i - mean(x)) * (y_i - mean(y))
    sum_xy = np.sum((x - mean_x) * (y - mean_y))
    
    # Calculate the Pearson correlation coefficient
    r = (1 / (len(x) - 1)) * (sum_xy / (std_x * std_y))
    
    return r

r = correlation_coeff(delta_no3_corr[:,0], delta_chl_corr[:,0])
r_pon = correlation_coeff(delta_no3_corr2[:,0], delta_pon_corr[:,0])

print(r)  #0.28499301277594324
print(r_pon)

r_pert = correlation_coeff(delta_no3_NoPert[:,0], delta_chl_NoPert[:,0])
r_pert_pon = correlation_coeff(delta_no3_NoPert_pon[:,0], delta_pon_NoPert[:,0])

#%%
sns.set_style("whitegrid")
markerstyle = ['o', 'x', '^',  's', 'D', "*"]
fig, ax = plt.subplots(figsize=(11,7))

ax.plot(delta_no3_corr[:,0], stats_263.slope*delta_no3_corr[:,0] + stats_263.intercept, color='cornflowerblue', linewidth = 1, alpha = .7,label = 'Observed linear regression')   
ax.plot(delta_no3_corr[:,0], expect_chlor, color='green', linestyle='dotted', lw = 1, alpha = .6,label='Expected Chl (C:Chl = 50:1)') 
   
for i in range(len(days)):
    sel_delta = delta_no3[:, 1] == days[i]
    ax.scatter(delta_no3[:, 0][sel_delta], delta_chl[:, 0][sel_delta], c = 'orangered', marker= markerstyle[i], s = 38)
    sel_pos = delta_no3_nopos[:, 1] == days[i]
    ax.scatter(delta_no3_nopos[:,0][sel_pos], delta_chl_nopos[:,0][sel_pos], marker= markerstyle[i], c = 'orange', s = 38)
    sel_neg = delta_no3_corr[:, 1] == days[i]
    ax.scatter(delta_no3_corr[:,0][sel_neg], delta_chl_corr[:, 0][sel_neg], marker= markerstyle[i], c = 'b', s = 48, label = str(days[i]) + ' April')
    sel_day = delta_no3_corr[:, 1] == 27
    ax.scatter(delta_no3_corr[:,0][sel_day], delta_chl_corr[:, 0][sel_day], marker= '*', c = 'deepskyblue', s = 48)

text = 'Estimated quatities:\n'
text += f'r: {stats_263.rvalue:.2f}\n'
text += f'r$^2$: {stats_263.rvalue**2:.2f}\n'
text += f'p_value: {stats_263.pvalue:.4f}'
props = dict(boxstyle='round', facecolor= 'whitesmoke')
ax.text(1.03, 0.85, text, transform = ax.transAxes, fontsize = 18, verticalalignment = 'top', bbox = props)
ax.text(1.03, 0.95, 'N. data points: ' + str(len(delta_no3_corr)), transform = ax.transAxes, fontsize = 18, verticalalignment = 'top', bbox = props)
plt.xlabel('$\Delta(NO_3^-)$', fontsize = 20)
plt.ylabel('$\Delta$(Chlor)', fontsize = 20)
plt.tight_layout()
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.grid(linewidth = .7, linestyle='--')
plt.legend(loc = 'lower right', labelcolor = ['cornflowerblue','green', 'b', 'b', 'b', 'b', 'b', 'b'], facecolor = 'whitesmoke', fontsize = 18, bbox_to_anchor = (0.5, 0, 1.21, 0.4))
fig.suptitle("Along ispopycnlas $\Delta$s, linearly interpolated. April 2018", fontsize = 25, y = 1.05)
plt.savefig(path_deltas + 'scatterplot_deltaNO3_linearInterp_April_shallower.png', bbox_inches = 'tight', dpi = 300)
#%%
sns.set_style("whitegrid")
plt.figure(figsize=(8,6))
tick = [17, 19, 21,  23, 25, 27, 29] #april
cmap = (mpl.colors.ListedColormap(['magenta', 'royalblue', 'cyan', 'lawngreen', 'gold', 'orangered']))
#cmap = mpl.cm.jet
norm =  mpl.colors.BoundaryNorm(tick, cmap.N) 
sm = plt.scatter(delta_no3[:,0], delta_chl[:,0], c = delta_chl[:,1], cmap = cmap, norm = norm, s = 50)
cbar = plt.colorbar(sm, ticks=days)
cbar.ax.tick_params(labelsize = 15, size = 0)
cbar.set_label(label = 'Days', size = 20)
plt.xlabel('$\Delta(NO_3^-)$', fontsize = 20)
plt.ylabel('$\Delta$(Chlor)', fontsize = 20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.grid(linewidth = .7, linestyle='--')
plt.title('Along ispopycnlas [26.3 - 26.65 $\sigma$] $\Delta$s, \n Linearly interpolated, April 2018',  fontsize = 20)
plt.tight_layout()
plt.savefig(path_deltas + 'scatterplot_data_linear_APRIL_shallower.png', dpi = 300)
#%%
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(11,7))

ax.plot(delta_no3_corr2[:,0], stats_pon.slope*delta_no3_corr2[:,0] + stats_pon.intercept, color='cornflowerblue', linewidth = 1, label = 'Observed linear regression', alpha = .7)   

for i in range(len(days)):
    sel_delta = delta_no3[:, 1] == days[i]
    ax.scatter(delta_no3[:, 0][sel_delta], delta_pon[:, 0][sel_delta], c = 'orangered', marker= markerstyle[i], s = 38)
    sel_pos = delta_no3_nopos[:, 1] == days[i]
    ax.scatter(delta_no3_nopos[:,0][sel_pos], delta_pon_nopos[:,0][sel_pos], marker= markerstyle[i], c = 'orange', s = 38)
    sel_neg = delta_no3_corr2[:, 1] == days[i]
    points3 = ax.scatter(delta_no3_corr2[:,0][sel_neg], delta_pon_corr[:, 0][sel_neg], marker= markerstyle[i], c = 'b', s = 48, label = str(days[i]) + ' April')
    sel_day = delta_no3_corr2[:, 1] == 27
    ax.scatter(delta_no3_corr2[:,0][sel_day], delta_pon_corr[:, 0][sel_day], marker= '*', c = 'deepskyblue', s = 48)

text = 'Estimated quatities:\n'
text += f'r: {stats_pon.rvalue:.2f}\n'
text += f'r$^2$: {stats_pon.rvalue**2:.2f}\n'
text += f'p_value: {stats_pon.pvalue:.3f}'
props = dict(boxstyle='round', facecolor= 'whitesmoke')
ax.text(1.03, 0.85, text, transform = ax.transAxes, fontsize = 18, verticalalignment = 'top', bbox = props)
ax.text(1.03, 0.95, 'N. data points: ' + str(len(delta_no3_corr2)), transform = ax.transAxes, fontsize = 18, verticalalignment = 'top', bbox = props)
plt.xlabel('$\Delta(NO_3^-)$', fontsize = 20)
plt.ylabel('$\Delta$(PON)', fontsize = 20)
plt.tight_layout()
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.grid(linewidth = .7, linestyle='--')
plt.legend(loc = 'upper right', labelcolor = ['cornflowerblue', 'b', 'b', 'b', 'b', 'b', 'b'], facecolor = 'whitesmoke', fontsize = 18, bbox_to_anchor = (0.5, 0, 1.18, 0.47))
fig.suptitle("Along ispopycnlas $\Delta$s, linearly interpolated. April 2018", fontsize = 25, y = 1.05)
plt.savefig(path_deltas + 'scatterplot_deltaPON_linearInterp_April_shallower.png', bbox_inches = 'tight', dpi = 300)
#%%CORRELATION CHL VS PON
stat = correlation_coeff(delta_chl_corr, delta_pon_corr)
sns.set_style("whitegrid")
plt.figure(figsize=(8,6))
tick = [17, 19, 21,  23, 25, 27, 29] #april
cmap = (mpl.colors.ListedColormap(['magenta', 'royalblue', 'cyan', 'lawngreen', 'gold', 'orangered']))
norm =  mpl.colors.BoundaryNorm(tick, cmap.N) 
sm = plt.scatter(delta_no3[:,0], delta_pon[:,0], c = delta_pon[:,1], cmap = cmap, norm = norm, s = 50)
cbar = plt.colorbar(sm, ticks=days)
cbar.ax.tick_params(labelsize = 15, size = 0)
cbar.set_label(label = 'Days', size = 20)
plt.xlabel('$\Delta(NO_3^-)$', fontsize = 20)
plt.ylabel('$\Delta$(PON)', fontsize = 20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.grid(linewidth = .7, linestyle='--')
plt.title('Along ispopycnlas [26.3 - 26.65 $\sigma$] $\Delta$s, \n Linearly interpolated, April 2018',  fontsize = 20)
plt.tight_layout()
plt.savefig(path_deltas + 'PON_scatterplot_data_linear_APRIL_shallower.png', dpi = 300)
#%%
#CALCULATE CORRELATINO CHL AND PON, SHOULDN'T
#SAME TREND ALSO PO4?
idx = np.where(delta_chl[:, 0] > 0)[0]
delta_chl_pos = delta_chl[idx]
delta_pon_pos = delta_pon[idx]

delta_pon_corr = delta_pon_pos[np.where(delta_pon_pos[:, 0] > 0)[0]]
delta_chl_corr2 = delta_chl_pos[np.where(delta_pon_pos[:, 0] > 0)[0]]

stats_chlPon = scipy.stats.linregress(delta_chl_corr2[:, 0], delta_pon_corr[:, 0])
print(f'Slope = {stats_chlPon.slope}')
print(f'Intercept = {stats_chlPon.intercept}')
print(f'R_squared = {stats_chlPon.rvalue}')
print(f'p-value = {stats_chlPon.pvalue} \n')

plt.scatter(delta_chl, delta_pon)
plt.scatter(delta_chl_corr2, delta_pon_corr, alpha = .5, color = 'yellow')

