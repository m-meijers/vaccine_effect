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

color_list = [(76,109,166),(215,139,45),(125,165,38),(228,75,41),(116,97,164),(182,90,36),(80,141,188),(246,181,56),(125,64,119),(158,248,72)]
color_list2 = []
for i in range(len(color_list)):
	color_list2.append(np.array(color_list[i])/256.)
color_list = color_list2

# x_range = np.linspace(0,10)
# for i in range(10):
# 	plt.plot(x_range, i * x_range,color=color_list[i])
# plt.show()

states_short2state = {'NJ':'New Jersey','NY':'New York State','CA':'California','TX':'Texas','GA':'Georgia','OH':'Ohio'}

df_s_awt = pd.read_csv("output/s_hat_alpha_wt.txt",'\t',index_col=False)

df_da = pd.read_csv("output/df_da_result.txt",'\t',index_col=False)
df_s_da = pd.read_csv("output/s_hat_delta_alpha.txt",sep='\t',index_col=False)
C_da = pd.read_csv("output/Pop_C_DA.txt",'\t',index_col=False)

df_od = pd.read_csv("output/df_od_result.txt",'\t',index_col=False)
df_s_od = pd.read_csv("output/s_hat_omi_delta.txt",sep='\t',index_col=False)
C_od = pd.read_csv("output/Pop_C_OD.txt",'\t',index_col=False)

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
    country2color[all_countries[i]] = color_list[i]
    country2style[all_countries[i]] = '-'
    country2marker[all_countries[i]] = 'o'
add = int(len(all_countries)/2)
for i in range(int(len(all_countries)/2), len(all_countries)):
    country2color[all_countries[i]] = color_list[i - int(len(all_countries)/2)]
    country2style[all_countries[i]] = '--'
    country2marker[all_countries[i]] = 'v'



inch2cm = 2.54

ratio = 1/1.62

mpl.rcParams['axes.linewidth'] = 0.3 #set the value globally

fig = plt.figure(figsize=(18/inch2cm,22/inch2cm))
gs0 = gridspec.GridSpec(1,2,figure=fig,wspace=0.3)
gs00 = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=gs0[0],hspace=0.25,wspace=0.1)
gs01 = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=gs0[1],hspace=0.25,wspace=0.1)
gs001 = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs01[2,0],hspace=0.05,wspace=0.1)

fs = 8
ls=6
ms=5
ms2=20
lw=0.75
elw=0.75
mlw=1.
lp=1.0
lp1=0.2
rot=0

ratio=1/1.5
#======================================================================
#Freq trajectory ITALY Delta - ALPHA
#======================================================================
ax = fig.add_subplot(gs00[0,0])
country='ITALY'
C_da_c = C_da.loc[list(C_da.country == country)]
t_range = list(C_da_c.time)
plt.plot(t_range, C_da_c.freq_alpha, color=color_list[1], linewidth=lw)
plt.plot(t_range, C_da_c.freq_delta, color=color_list[2], linewidth=lw)
plt.plot(t_range, C_da_c.freq_other, color=color_list[0], linewidth=lw)
plt.fill_between(t_range, np.array(C_da_c.freq_alpha) - np.sqrt(np.array(C_da_c.var_a)),np.array(C_da_c.freq_alpha) + np.sqrt(np.array(C_da_c.var_a)),color=color_list[1],alpha=0.4,linewidth=0.0)
plt.fill_between(t_range, np.array(C_da_c.freq_delta) - np.sqrt(np.array(C_da_c.var_d)),np.array(C_da_c.freq_delta) + np.sqrt(np.array(C_da_c.var_d)),color=color_list[2],alpha=0.4,linewidth=0.0)
plt.fill_between(t_range, np.array(C_da_c.freq_other) - np.sqrt(np.array(C_da_c.var_other)),np.array(C_da_c.freq_other) + np.sqrt(np.array(C_da_c.var_other)),color=color_list[0],alpha=0.4,linewidth=0.0)


plt.ylabel("Frequency, $x(t)$",fontsize=fs,labelpad=lp)
ax.set_ylim([-0.02,1.02])
ax.set_yticks([0,0.5,1.0],['0.0','0.5','1.0'],fontsize=ls)
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
xtick_labels = ['2021-05-01','2021-06-01','2021-07-01','2021-08-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['May $\'$21','Jun. $\'$21','Jul. $\'$21','Aug. $\'$21']
ax.set_xticks(xtick_pos,xtick_labels,rotation=rot,ha='center',fontsize=fs)
# ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
ax.tick_params(direction="in",width=0.5,pad=lp)


#======================================================================
#Freq trajectory ITALY Omicron - Delta
#======================================================================
ax = fig.add_subplot(gs01[0,0])
C_od_c = C_od.loc[list(C_od.country == country)]
t_range = list(C_od_c.time)
plt.plot(t_range,C_od_c.freq_omi,color=color_list[4],linewidth=lw)
plt.plot(t_range,C_od_c.freq_delta,color=color_list[2],linewidth=lw)
plt.fill_between(t_range, np.array(C_od_c.freq_omi) - np.sqrt(np.array(C_od_c.var_o)),np.array(C_od_c.freq_omi) + np.sqrt(np.array(C_od_c.var_o)),color=color_list[4],alpha=0.4,linewidth=0.0)
plt.fill_between(t_range, np.array(C_od_c.freq_delta) - np.sqrt(np.array(C_od_c.var_d)),np.array(C_od_c.freq_delta) + np.sqrt(np.array(C_od_c.var_d)),color=color_list[2],alpha=0.4,linewidth=0.0)

# xtick_pos = list(np.arange(t_range[0],t_range[-1],30)) + [t_range[-1]]
# xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
plt.ylabel("Frequency, $x(t)$",fontsize=fs,labelpad=lp)
ax.set_ylim([-0.02,1.02])
ax.set_yticks([0,0.5,1.0],['0.0','0.5','1.0'],fontsize=ls)
ax.tick_params(direction="in",width=0.5,pad=lp)
xtick_labels = ['2021-12-01','2022-01-01','2022-02-01','2022-03-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Dec. $\'$21','Jan. $\'$22','Feb. $\'$22','Mar. $\'$22']

ax.set_xticks(xtick_pos,xtick_labels,rotation=rot,ha='center',fontsize=fs)
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
# ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

#======================================================================
#C trajecoties ITALY Delta - ALPHA
#======================================================================
ax = fig.add_subplot(gs00[2,0])
country='ITALY'
C_da_c = C_da.loc[list(C_da.country == country)]
t_range = list(C_da_c.time)
# plt.plot(t_range, C_da_c.R_vac,   'k-', linewidth=lw,label='Vaccination')
# plt.plot(t_range, C_da_c.R_alpha,'k--', linewidth=lw,label='Recovery Ancestral')
# plt.plot(t_range, C_da_c.R_delta,'k:', linewidth=lw,label='Recovery invading')
plt.plot(t_range, C_da_c.Cav, '-',color=color_list[1], linewidth=lw)
plt.plot(t_range, C_da_c.Cdv, '-',color=color_list[2], linewidth=lw)
plt.plot(t_range, C_da_c.Caa,'--',color=color_list[1], linewidth=lw)
plt.plot(t_range, C_da_c.Cda,'--',color=color_list[2], linewidth=lw)
# plt.plot(t_range,  C_da_c.Cad,':',color=color_list[0], linewidth=lw)
# plt.plot(t_range,  C_da_c.Cdd,':',color=color_list[1], linewidth=lw)

plt.fill_between(t_range, C_da_c.Cdv,C_da_c.Cav,color=color_list[2],alpha=0.3,linewidth=0.0)
plt.fill_between(t_range, C_da_c.Cda,C_da_c.Caa,color=color_list[2],alpha=0.3,linewidth=0.0)
plt.ylabel("Population immunity, $\\bar{C}_i^k$",fontsize=fs,labelpad=lp)
ax.set_ylim([-0.02,0.6])
ax.set_yticks([0,0.2,0.4,0.6],['0.0','0.2','0.4','0.6'],fontsize=ls)
ax.tick_params(direction="in",width=0.5,pad=lp)
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
xtick_labels = ['2021-05-01','2021-06-01','2021-07-01','2021-08-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['May $\'$21','Jun. $\'$21','Jul. $\'$21','Aug. $\'$21']
ax.set_xticks(xtick_pos,xtick_labels,rotation=rot,ha='center',fontsize=fs)
legend_elements = []
legend_elements.append(Line2D([],[],marker='',color='k',linestyle='-',linewidth=1.0,label='Vaccination'))
legend_elements.append(Line2D([],[],marker='',color='k',linestyle='--',linewidth=1.0,label='Alpha recovery'))
plt.legend(handles = legend_elements,loc='upper left', prop={'size':ls-1})
# ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

#======================================================================
#C trajectories ITALY Omicron - Delta
#======================================================================
# ax = fig.add_subplot(gs01[2,0])
ax = fig.add_subplot(gs001[0,0])
C_od_c = C_od.loc[list(C_od.country == country)]
t_range = list(C_od_c.time)
# plt.plot(t_range, C_od_c.R_vac,   'k-',linewidth=lw,label='Vaccination')
# plt.plot(t_range, C_od_c.R_boost,   'k-.',linewidth=lw,label='Booster')
# plt.plot(t_range, C_od_c.R_delta,'k--',linewidth=lw,label='Recovery Ancestral')
# plt.plot(t_range, C_od_c.R_omi,':k',linewidth=lw,label='Recovery Invading')

plt.plot(t_range, C_od_c.Cdv,  '-',color = color_list[2],linewidth=lw)
plt.plot(t_range, C_od_c.Cov,   '-',color = color_list[4],linewidth=lw)
plt.plot(t_range, C_od_c.Cdb, '-.',color = color_list[2],linewidth=lw)
plt.plot(t_range, C_od_c.Cob,  '-.',color = color_list[4],linewidth=lw)


plt.fill_between(t_range, C_od_c.Cov,C_od_c.Cdv,color=color_list[4],alpha=0.3,linewidth=0.0)
plt.fill_between(t_range, C_od_c.Cob,C_od_c.Cdb,color=color_list[4],alpha=0.3,linewidth=0.0)

plt.ylabel("Population immunity, $\\bar{C}_i^k$",fontsize=fs,labelpad=lp)
ax.set_ylim([0.0,0.6])
ax.set_yticks([0,0.2,0.4,0.6],['0.0','0.2','0.4','0.6'],fontsize=ls)
ax.tick_params(direction="in",width=0.5,pad=lp)
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
xtick_labels = ['2021-12-01','2022-01-01','2022-02-01','2022-03-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['','','','']
ax.set_xticks(xtick_pos,xtick_labels,rotation=rot,ha='center',fontsize=fs)

legend_elements = []
legend_elements.append(Line2D([],[],marker='',color='k',linestyle='-.',linewidth=1.0,label='Booster'))
legend_elements.append(Line2D([],[],marker='',color='k',linestyle='-',linewidth=1.0,label='Vaccination'))
legend_elements.append(Line2D([],[],marker='',color='k',linestyle='--',linewidth=1.0,label='Delta recovery'))
legend_elements.append(Line2D([],[],marker='',color='k',linestyle=':',linewidth=1.0,label='Omicron recovery'))
plt.legend(handles = legend_elements,loc='upper left', prop={'size':ls-1})

ax = fig.add_subplot(gs001[1,0])
plt.plot(t_range, C_od_c.Cdd, '--',color = color_list[2],linewidth=lw)
plt.plot(t_range, C_od_c.Cod,  '--',color = color_list[4],linewidth=lw)
plt.plot(t_range, C_od_c.Cdo,  ':',color = color_list[2],linewidth=lw)
plt.plot(t_range, C_od_c.Coo,   ':',color = color_list[4],linewidth=lw)
plt.fill_between(t_range, C_od_c.Cod,C_od_c.Cdd,color=color_list[4],alpha=0.3,linewidth=0.0)
plt.fill_between(t_range, C_od_c.Cdo,C_od_c.Coo,color=color_list[2],alpha=0.3,linewidth=0.0)
xtick_labels = ['2021-12-01','2022-01-01','2022-02-01','2022-03-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Dec. $\'$21','Jan. $\'$22','Feb. $\'$22','Mar. $\'$22']
ax.set_xticks(xtick_pos,xtick_labels,rotation=rot,ha='center',fontsize=fs)
ax.tick_params(direction="in",width=0.5,pad=lp)
ax.set_ylim([-0.02,0.2])
ax.set_yticks([0,0.05,0.1,0.15],['0.0','0.05','0.1','0.15'],fontsize=ls)

# ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)



#======================================================================
#Observed selection coefficients
#======================================================================
ax = fig.add_subplot(gs00[1,0])
means = []
slopes = []
times = []
points = []
for c in countries_da:
	df_c = df_da.loc[list(df_da.Country == c)]
	t_range = df_c.time
	plt.plot(t_range - np.mean(t_range), df_c.s_hat,linestyle=country2style[c],marker='',color=country2color[c],alpha=0.6,markersize=ms,linewidth=lw)
	R = ss.linregress(t_range - np.mean(t_range), np.array(df_c.s_hat))
	means.append(np.mean(df_c.s_hat))
	slopes.append(R.slope)
	times += list(t_range - np.mean(t_range))
	points += list(np.array(df_c.s_hat))
# plt.plot(0,np.mean(means),'k_',markersize=ms+6,linewidth=lw+2)
plt.errorbar([0],[np.mean(means)],fmt='none',yerr=[np.sqrt(np.var(means))],capsize=0,marker='',color='k',linewidth=lw,elinewidth=lw)
x_range= np.linspace(-30,30)
plt.plot(x_range, np.mean(slopes) * x_range + np.mean(means),'k-',linewidth=lw)
R = ss.linregress(times, points)
print(R)
plt.xlabel("Time from midpoint, $t$ [days]",fontsize=fs)
plt.ylabel("Selection coefficient, $\\hat{s}(t)$",fontsize=fs)
ax.tick_params(direction="in",width=0.5,pad=lp,labelsize=ls)

#======================================================================
#Observed selection coefficients
#======================================================================
ax = fig.add_subplot(gs01[1,0])
means1 = []
slopes1 = []
times = []
points = []
for c in countries_od:
	df_c = df_od.loc[list(df_od.Country == c)]
	t_range = df_c.time
	plt.plot(t_range - np.mean(t_range), df_c.s_hat,linestyle=country2style[c],marker='',color=country2color[c],alpha=0.6,markersize=ms,linewidth=lw)
	R = ss.linregress(t_range - np.mean(t_range), np.array(df_c.s_hat))
	means1.append(np.mean(df_c.s_hat))
	slopes1.append(R.slope)
	times += list(t_range - np.mean(t_range))
	points += list(-np.array(df_c.s_hat))
# plt.plot(0,np.mean(means),'k_',markersize=ms+6,linewidth=lw+2)
plt.errorbar([0],[np.mean(means1)],fmt='none',yerr=[np.sqrt(np.var(means1))],capsize=0,marker='',color='k',linewidth=lw,elinewidth=lw)
x_range= np.linspace(-25,25)
R = ss.linregress(times, points)
print(R)
plt.plot(x_range, np.mean(slopes1) * x_range + np.mean(means1),'k-',linewidth=lw)
plt.xlabel("Time from midpoint, $t$ [days]",fontsize=fs)
plt.ylabel("Selection coefficient, $\\hat{s}(t)$",fontsize=fs)
ax.tick_params(direction="in",width=0.5,pad=lp,labelsize=ls)

legend_elements = []
for c in sorted(all_countries):
	label = c
	if len(label) == 2:
		label = states_short2state[c]
	else:
		label = c[0] + c[1:].lower()
	legend_elements.append(Line2D([],[],marker='',color=country2color[c],linestyle=country2style[c],label=label,markersize=4, linewidth=lw))
legend_elements.append(Line2D([],[],marker='',color=color_list[0],linestyle='-',linewidth=1.0,label='wt / 1'))
legend_elements.append(Line2D([],[],marker='',color=color_list[1],linestyle='-',linewidth=1.0,label='Alpha'))
legend_elements.append(Line2D([],[],marker='',color=color_list[2],linestyle='-',linewidth=1.0,label='Delta'))
legend_elements.append(Line2D([],[],marker='',color=color_list[4],linestyle='-',linewidth=1.0,label='Omicron'))
plt.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1.15,1.3),prop={'size':6})


plt.savefig("figures/Fig1.pdf")
plt.close()


