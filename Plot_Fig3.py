import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from matplotlib.lines import Line2D
import scipy.integrate as si
import scipy.stats as ss
import scipy.optimize as so
import json
from flai.util.Time import Time
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import matplotlib as mpl


states_short2state = {'NJ':'New Jersey','NY':'New York State','CA':'California','TX':'Texas','GA':'Georgia','OH':'Ohio'}
df_da = pd.read_csv("output/df_da_result.txt",'\t',index_col=False)
df_s_da = pd.read_csv("output/s_hat_delta_alpha.txt",sep='\t',index_col=False)

df_od = pd.read_csv("output/df_od_result.txt",'\t',index_col=False)
df_s_od = pd.read_csv("output/s_hat_omi_delta.txt",sep='\t',index_col=False)

countries_da = sorted(list(set(df_da.Country)))
countries_od = sorted(list(set(df_od.Country)))

all_countries = sorted(list(set(countries_da).union(set(countries_od))))
countries_ad = countries_da
file = open("output/result_da.txt",'r')
coefs_da = file.readline().split()
file.close()
coefs_da = [float(c) for c in coefs_da]

file = open("output/result_od.txt",'r')
coefs_od = file.readline().split()
file.close()
coefs_od = [float(c) for c in coefs_od]

country2color = {}
country2style = {}
country2marker = {}
cm = plt.get_cmap("hsv")
for i in range(int(len(all_countries)/2)):
    country2color[all_countries[i]] = cm((i+1) / (int(len(all_countries)/2)+1))
    country2style[all_countries[i]] = '-'
    country2marker[all_countries[i]] = 'o'
add = int(len(all_countries)/2)
for i in range(int(len(all_countries)/2), len(all_countries)):
    country2color[all_countries[i]] = cm((i-add+1) / (int(len(all_countries)/2)+1))
    country2style[all_countries[i]] = '--'
    country2marker[all_countries[i]] = 'v'



inch2cm = 2.54

ratio = 1/1.62

mpl.rcParams['axes.linewidth'] = 0.3 #set the value globally

fig = plt.figure(figsize=(18/inch2cm,22/inch2cm))
gs0 = gridspec.GridSpec(1,2,figure=fig,wspace=0.3)
gs00 = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs0[0],hspace=0.25,wspace=0.1)
gs01 = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs0[1],hspace=0.25,wspace=0.1)

fs = 8
ls=7
ms=3
ms2=20
lw=0.75
elw=0.75
mlw=1.5
lp=0.7
lp1=0.2
rot=15

ratio=1/1.5
#======================================================================
#s - shat plot Delta - Alpha
#======================================================================
ax = fig.add_subplot(gs00[0,0])
for c in countries_da:
	df_c = df_da.loc[list(df_da.Country == c)]
	ax.plot(np.array(df_c.s_model) - np.mean(df_c.s_model), np.array(df_c.s_hat) - np.mean(df_c.s_hat),country2style[c],marker=country2marker[c],color=country2color[c],alpha=0.7,linewidth=lw,markersize=ms)
x_range = np.linspace(-0.075,0.075)
plt.plot(x_range, x_range,'k-',linewidth=lw,alpha=0.3)
plt.xlim([-0.06,0.06])
plt.ylim([-0.06,0.06])
plt.xlabel("Model variation, $\\Delta s$",fontsize=fs,labelpad=lp)
plt.ylabel("Measured variation, $\\Delta \\hat{s}$",fontsize=fs)
RMSQE = np.sqrt(np.mean((np.array(df_da['s_model']) - np.array(df_da['s_hat']))**2))
R = ss.linregress(np.array(df_da.s_model),np.array(df_da.s_hat))
x0, xmax = plt.xlim()
y0, ymax = plt.ylim()
data_width = xmax - x0
data_height = ymax - y0
plt.text(x0 + data_width * 0.6, y0 + data_height * 0.1, 'R2$=' + str(np.round(R.rvalue**2,2)) + '$ \nP  $= ' + str(format(R.pvalue,'.1E')) + '$',fontsize=ls)
plt.tick_params(direction="in",width=0.3,pad=lp,labelsize=ls)

#======================================================================
#s - shat plot Omicron Delta
#======================================================================
ax = fig.add_subplot(gs01[0,0])
for c in countries_od:
	df_c = df_od.loc[list(df_od.Country == c)]
	ax.plot(np.array(df_c.s_model) - np.mean(df_c.s_model), np.array(df_c.s_hat) - np.mean(df_c.s_hat),country2style[c],marker=country2marker[c],color=country2color[c],alpha=0.7,linewidth=lw,markersize=ms)
x_range = np.linspace(-0.1,0.1)
plt.plot(x_range, x_range,'k-',linewidth=lw,alpha=0.3)
plt.xlim([-0.06,0.06])
plt.ylim([-0.06,0.06])
plt.xlabel("Model variation, $\\Delta s$",fontsize=fs,labelpad=lp)
plt.ylabel("Measured variation, $\\Delta \\hat{s}$",fontsize=fs)
RMSQE = np.sqrt(np.mean((np.array(df_od['s_model']) - np.array(df_od['s_hat']))**2))
R = ss.linregress(np.array(df_od.s_model),np.array(df_od.s_hat))
x0, xmax = plt.xlim()
y0, ymax = plt.ylim()
data_width = xmax - x0
data_height = ymax - y0
plt.text(x0 + data_width * 0.6, y0 + data_height * 0.1, '$R2=' + str(np.round(R.rvalue**2,2)) + '$ \nP  $= ' + str(format(R.pvalue,'.2E')) + '$',fontsize=ls)
plt.tick_params(direction="in",width=0.3,pad=lp,labelsize=ls)


# plt.plot(np.linspace(-0.001,0.0011), np.linspace(-0.001,0.0011),'k--',linewidth=lw)
#======================================================================
#Selection ranking Delta - alpha
#======================================================================
ax = fig.add_subplot(gs00[1,0])
selection_list = defaultdict(lambda: defaultdict(lambda: []))
for c in countries_ad:
	df_c = df_da.loc[list(df_da.Country == c)]
	s0 = df_c.iloc[0].s_0
	s_vac = np.array(df_c.x_eff_vac) * coefs_da[0]
	s_alpha = np.array(df_c.x_eff_alpha) * coefs_da[1]
	s_delta = np.array(df_c.x_eff_delta) * coefs_da[2]

	selection_list['s0']['means'].append(s0)
	selection_list['s_vac']['means'].append(np.mean(s_vac))
	selection_list['s_alpha']['means'].append(np.mean(s_alpha))
	selection_list['s_delta']['means'].append(np.mean(s_delta))
	selection_list['s_vac']['var'].append(np.sqrt(np.var(s_vac)))
	selection_list['s_alpha']['var'].append(np.sqrt(np.var(s_alpha)))
	selection_list['s_delta']['var'].append(np.sqrt(np.var(s_delta)))
	
	
a = [np.mean(selection_list['s0']['means']),np.mean(selection_list['s_vac']['means']),np.mean(selection_list['s_alpha']['means']),np.mean(selection_list['s_delta']['means'])]
b = [np.mean(selection_list['s0']['var']),np.mean(selection_list['s_vac']['var']),np.mean(selection_list['s_alpha']['var']),np.mean(selection_list['s_delta']['var'])]
plt.bar([1,2,3,4],a,width=0.6,color='grey',linewidth=0.6)
plt.errorbar([1,2,3,4],a,fmt='none',yerr=b,capsize=4,marker='',color='k',linewidth=lw)
plt.ylabel("Selection, $s_{\\delta \\alpha}$",fontsize=fs)
plt.ylim([-0.015,0.09])
plt.yticks([0.0,0.02,0.04,0.06],['0.0','0.02','0.04','0.06'],fontsize=ls)
plt.tick_params(direction="in",width=0.3,pad=lp,labelsize=ls)
plt.xticks([1,2,3,4],['$s_0$','$s_{\\rm vac}$','$s_{\\alpha}$','$s_{\\delta}$'],fontsize=fs+3)

#======================================================================
#Selection ranking Omicron - Delta
#======================================================================
ax = fig.add_subplot(gs01[1,0])
for c in countries_od:
	df_c = df_od.loc[list(df_od.Country == c)]
	s0 = df_c.iloc[0].s_0
	s_vac0 = np.array(df_c.x_eff_vac0) * coefs_od[0]
	s_vac = np.array(df_c.x_eff_vac) * coefs_od[0]
	s_boost = np.array(df_c.x_eff_boost) * coefs_od[1]
	s_boost_eff = s_vac - s_vac0 + s_boost
	s_delta = np.array(df_c.x_eff_delta) * coefs_od[2]
	s_omi = np.array(df_c.x_eff_omi) * coefs_od[3]
	
	selection_list['OD_s0']['means'].append(s0)
	selection_list['OD_s_vac']['means'].append(np.mean(s_vac0))
	selection_list['OD_s_boost']['means'].append(np.mean(s_boost_eff))
	selection_list['OD_s_delta']['means'].append(np.mean(s_delta))
	selection_list['OD_s_omi']['means'].append(np.mean(s_omi))
	selection_list['OD_s_vac']['var'].append(np.sqrt(np.var(s_vac0)))
	selection_list['OD_s_boost']['var'].append(np.sqrt(np.var(s_boost_eff)))
	selection_list['OD_s_delta']['var'].append(np.sqrt(np.var(s_delta)))
	selection_list['OD_s_omi']['var'].append(np.sqrt(np.var(s_omi)))

a = [np.mean(selection_list['OD_s0']['means']),np.mean(selection_list['OD_s_vac']['means']),np.mean(selection_list['OD_s_boost']['means']),np.mean(selection_list['OD_s_delta']['means']),np.mean(selection_list['OD_s_omi']['means'])]
b = [np.mean(selection_list['OD_s0']['var']),np.mean(selection_list['OD_s_vac']['var']),np.mean(selection_list['OD_s_boost']['var']),np.mean(selection_list['OD_s_delta']['var']),np.mean(selection_list['OD_s_omi']['var'])]
plt.bar([1,2,3,4,5],a,width=0.6,color='grey')
plt.errorbar([1,2,3,4,5],a,fmt='none',yerr=b,capsize=4,marker='',color='k',linewidth=lw)
plt.ylabel("Selection, $s_{o \\delta}$",fontsize=fs)
plt.ylim([-0.015,0.09])
# plt.yticks([-0.02,0.0,0.05,0.1],['-0.02','0.0','0.05','0.1'])
plt.tick_params(direction="in",width=0.3,pad=lp,labelsize=ls)
plt.yticks([0.0,0.02,0.04,0.06],['0.0','0.02','0.04','0.06'],fontsize=ls)
plt.xticks([1,2,3,4,5],['$s_0$','$s_{\\rm vac}$','$s_{\\rm boost}$','$s_{\\delta}$','$s_{o}$'],fontsize=fs+3)

legend_elements = []
for c in sorted(all_countries):
	label = c
	if len(label) == 2:
		label = states_short2state[c]
	else:
		label = c[0] + c[1:].lower()
	legend_elements.append(Line2D([],[],marker=country2marker[c],color=country2color[c],linestyle=country2style[c],markersize=3,label=label,linewidth=1.))
plt.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1.15,2.3),prop={'size':6})


plt.savefig("figures/Fig3.pdf")
plt.close()

