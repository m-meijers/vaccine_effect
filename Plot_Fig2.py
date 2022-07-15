import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import scipy.stats as ss
import scipy
from collections import defaultdict
import time
from scipy.optimize import minimize
import scipy.integrate as si
import copy
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib as mpl

color_list = [(76,109,166),(215,139,45),(125,165,38),(228,75,41),(116,97,164),(182,90,36),(80,141,188),(246,181,56),(125,64,119),(158,248,72)]
color_list2 = []
for i in range(len(color_list)):
	color_list2.append(np.array(color_list[i])/256.)
color_list = color_list2

def sigmoid_func(t,mean=np.log10(0.2 * 94),s=3.0):
	val = 1 / (1 + np.exp(-s * (t - mean)))
	return val

def sigmoid_func_inv(c, mean= np.log10(0.2*94),s=3.0):
	val = 1/s * np.log(c/(1-c)) + mean
	return val

def C_variant(dT, t, T0=223):
	return sigmoid_func(np.log10(T0) - np.log10(np.exp(t/tau_decay)) - np.log10(dT))

k=3.0 #2.2-- 4.2 -> sigma = 1/1.96
n50 = np.log10(0.2 * 94) #0.14 -- 0.28 -> sigma 0.06/1.96

tau_decay = 91 #2021 Nov;232:108871.doi: 10.1016/j.clim.2021.108871. -> tau = 28 A LOT OF FREEDOM HERE

dT_alpha_alpha = 1.
dT_delta_alpha = 2.8
dT_alpha_delta = 3.5
dT_delta_delta = 1.

dS_0_da = 0.045
dS_0_od = 0.037


dT_list = np.arange(0,6,0.1)
s_vac = [[],[]]

for dT in dT_list:
	C_var_0 = C_variant(1.0,0)
	s_vac[0].append((C_variant(1.,0.) - C_variant(2**dT,0.)))
	s_vac[1].append((C_variant(1.,180) - C_variant(2**dT,180.)))


T_list = np.linspace(0,3)

tau = 91

ratio = 1
fig = plt.figure(figsize=[9,7])
fs=10
tw=0.8
lp=1
spec = GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[0.5,0.5],wspace=0.001,hspace=0.15)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[1, 0])
ax4 = fig.add_subplot(spec[1, 1])


ratio=1
ax1.plot(T_list, sigmoid_func(T_list),'k-')
ax1.plot(np.log10(94) - np.log10(dT_alpha_alpha),sigmoid_func(np.log10(94) - np.log10(dT_alpha_alpha)), 'o',color=color_list[1])
ax1.plot(np.log10(94) - np.log10(dT_delta_alpha),sigmoid_func(np.log10(94) - np.log10(dT_delta_alpha)), 'o',color=color_list[2])
ax1.set_xticks([],[])
ax1.set_yticks([0.0,1.0],['0.0','1.0'])
ax1.tick_params(direction="in",width=tw)
ax1.set_ylim([-0.02,1.02])
ax1.set_ylabel("Cross-immunity, $C$",fontsize=fs,labelpad=lp)
ax1.set_xlabel("Titer, $T$",fontsize=fs)
x_left, x_right = ax1.get_xlim()
y_low, y_high = ax1.get_ylim()
ax1.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

ax3.plot(T_list, sigmoid_func(T_list),'k-')
ax3.plot(np.log10(94) - np.log10(dT_alpha_delta),sigmoid_func(np.log10(94) - np.log10(dT_alpha_delta)), 'o',color=color_list[1])
ax3.plot(np.log10(94) - np.log10(dT_delta_delta),sigmoid_func(np.log10(94) - np.log10(dT_delta_delta)), 'o',color=color_list[2])
ax3.set_xticks([],[])
ax3.set_yticks([0.0,1.0],['0.0','1.0'])
ax3.tick_params(direction="in",width=tw)
ax3.set_ylim([-0.02,1.02])
ax3.set_ylabel("Cross-immunity, $C$",fontsize=fs,labelpad=lp)
ax3.set_xlabel("Titer, $T$",fontsize=fs)
x_left, x_right = ax3.get_xlim()
y_low, y_high = ax3.get_ylim()
ax3.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

t_range = np.linspace(0,np.log(1000)*tau)
ax2.plot(t_range, sigmoid_func(3 - np.log10(np.exp(t_range/tau))),'k-')
t_alpha_start = tau * np.log(10**(3 - np.log10(94) + np.log10(dT_alpha_alpha)))
t_alpha = np.linspace(t_alpha_start,np.log(1000)*tau)
ax2.plot(t_alpha_start, sigmoid_func(3 - np.log10(np.exp(t_alpha_start/tau))),'o',color=color_list[1])
ax2.plot(t_alpha, sigmoid_func(3 - np.log10(np.exp(t_alpha/tau))),'-',color=color_list[1])
ax2.plot(np.ones(50) * t_alpha_start, np.linspace(0,sigmoid_func(3 - np.log10(np.exp(t_alpha_start/tau)))),'-',color=color_list[1])
t_delta_start = tau * np.log(10**(3 - np.log10(94) + np.log10(dT_delta_alpha)))
t_delta = np.linspace(t_delta_start, np.log(2800)*tau)
ax2.plot(t_alpha_start, sigmoid_func(3- np.log10(np.exp(t_delta_start/tau))),'o',color=color_list[2])
ax2.plot(t_delta - (t_delta_start - t_alpha_start), sigmoid_func(3-np.log10(np.exp(t_delta/tau))),'-',color=color_list[2])
ax2.plot(np.ones(50) * t_alpha_start, np.linspace(0,sigmoid_func(3 - np.log10(np.exp(t_delta_start/tau)))),'-',color=color_list[2])
ax2.fill_between(t_alpha,  sigmoid_func(3-np.log10(np.exp((t_alpha + (t_delta_start - t_alpha_start))/tau))),sigmoid_func(3 - np.log10(np.exp(t_alpha/tau))),color=color_list[2],alpha=0.3)
ax2.set_xticks([],[])
ax2.set_yticks([],[])
ax2.set_ylim([-0.02,1.02])
ax2.set_xlabel("Scaled time after immunisation, $\\Delta t / \\tau$",fontsize=fs)
ax2.set_yticks([0.0,1.0],['',''])
ax2.tick_params(direction="in",width=tw)

x_left, x_right = ax2.get_xlim()
y_low, y_high = ax2.get_ylim()
ax2.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)


t_range = np.linspace(0,np.log(1000)*tau)
ax4.plot(t_range, sigmoid_func(3 - np.log10(np.exp(t_range/tau))),'k-')
t_delta_start = tau * np.log(10**(3 - np.log10(94) + np.log10(dT_delta_delta)))
t_delta = np.linspace(t_delta_start, np.log(1000)*tau)
ax4.plot(t_delta_start, sigmoid_func(3- np.log10(np.exp(t_delta_start/tau))),'o', color=color_list[2])
ax4.plot(t_delta, sigmoid_func(3-np.log10(np.exp(t_delta/tau))),'-', color=color_list[2])
ax4.plot(np.ones(50) * t_alpha_start, np.linspace(0,sigmoid_func(3 - np.log10(np.exp(t_delta_start/tau)))),'-', color=color_list[2])
t_alpha_start = tau * np.log(10**(3 - np.log10(94) + np.log10(dT_alpha_delta)))
t_alpha = np.linspace(t_alpha_start,np.log(3500)*tau)
t_alpha0 = np.linspace(t_alpha_start-150,t_alpha_start)
ax4.plot(t_delta_start, sigmoid_func(3 - np.log10(np.exp(t_alpha_start/tau))),'o',color=color_list[1])
ax4.plot(t_alpha - (t_alpha_start - t_delta_start), sigmoid_func(3 - np.log10(np.exp(t_alpha/tau))),'-',color=color_list[1])
# ax4.plot(t_alpha0 - (t_alpha_start - t_delta_start), sigmoid_func(3 - np.log10(np.exp(t_alpha/tau))),'-',color=color_list[1])
# ax4.plot(t_alpha0 - (t_alpha_start - t_delta_start), sigmoid_func(3 - np.log10(np.exp(t_alpha/tau))),'-',color='k')
ax4.plot(np.ones(50) * t_delta_start, np.linspace(0,sigmoid_func(3 - np.log10(np.exp(t_alpha_start/tau)))),'-',color=color_list[1])

ax4.fill_between(t_delta,sigmoid_func(3 - np.log10(np.exp(t_delta/tau))),sigmoid_func(3-np.log10(np.exp((t_delta + (t_alpha_start - t_delta_start))/tau))),color=color_list[1],alpha=0.3)

ax4.set_xticks([],[])
ax4.set_yticks([],[])
ax4.set_ylim([-0.02,1.02])
ax4.set_xlabel("Scaled time after immunisation, $\\Delta t / \\tau$",fontsize=fs)
ax4.set_yticks([0.0,1.0],['',''])
ax4.tick_params(direction="in",width=tw)

x_left, x_right = ax2.get_xlim()
y_low, y_high = ax2.get_ylim()
ax2.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

x_left, x_right = ax4.get_xlim()
y_low, y_high = ax4.get_ylim()
ax4.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

plt.subplots_adjust(wspace=0.0001)
plt.savefig("figures/Fig2.pdf")
plt.close()



