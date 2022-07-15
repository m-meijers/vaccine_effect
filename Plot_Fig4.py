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
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import scipy.integrate as integrate
from flai.util.Time import Time


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

df_da = pd.read_csv("output/df_da_result.txt",'\t',index_col=False)
df_s_da = pd.read_csv("output/s_hat_delta_alpha.txt",sep='\t',index_col=False)
C_da = pd.read_csv("output/Pop_C_DA.txt",'\t',index_col=False)

df_od = pd.read_csv("output/df_od_result.txt",'\t',index_col=False)
df_s_od = pd.read_csv("output/s_hat_omi_delta.txt",sep='\t',index_col=False)
C_od = pd.read_csv("output/Pop_C_OD.txt",'\t',index_col=False)
df_s_awt = pd.read_csv("output/s_hat_alpha_wt.txt",sep='\t',index_col=False)
df_s_ba2ba1 = pd.read_csv("output/s_hat_ba2_ba1.txt",sep='\t',index_col=False)
df_s_ba45ba2 = pd.read_csv("output/s_hat_ba45_ba2.txt",sep='\t',index_col=False)

dT_alpha_vac = 1.8
dT_delta_vac = 3.2
dT_omi_vac   = 47
dT_wt_vac = 1.

dT_alpha_booster = 1.8 ##no value -> use the same as vaccinatino
dT_delta_booster = 2.8
dT_omi_booster   = 6.4
dT_ba2_booster = 5.8
dT_ba45_booster = 15
dT_wt_booster = 1.

dT_alpha_alpha = 1.
dT_delta_alpha = 2.8
dT_omi_alpha   = 33
dT_wt_alpha = 1.8 #symmetry

dT_alpha_delta = 3.5
dT_delta_delta = 1.
dT_omi_delta   = 27.
dT_wt_delta = 3.2 #symmetry

dT_alpha_omi = 33.
dT_delta_omi = 27.
dT_omi_omi   = 1.
dT_ba2_ba1 = 2.6
dT_ba45_ba1 = 4.7
dT_wt_omi = 47 #symmetry

dT_ba1_ba2 = 4.4 
dT_ba45_ba2 = 2.3

gamma_vac_ad = 1.15
gamma_inf_ad = 2.3
gamma_vac_od = 0.30
gamma_inf_od = 0.6

dS_0_od = 0.058
dS_0_da = 0.048
T_decay = np.log10(np.exp(1))

c_alpha_vac = integrate.quad(sigmoid_func,np.log10(223) - np.log10(dT_alpha_vac) - T_decay, np.log10(223) - np.log10(dT_alpha_vac))[0]/T_decay
c_delta_vac = integrate.quad(sigmoid_func,np.log10(223) - np.log10(dT_delta_vac) - T_decay, np.log10(223) - np.log10(dT_delta_vac))[0]/T_decay
c_omi_vac = integrate.quad(sigmoid_func,np.log10(223) - np.log10(dT_omi_vac) - T_decay, np.log10(223) - np.log10(dT_omi_vac))[0]/T_decay
c_wt_vac = integrate.quad(sigmoid_func,np.log10(223) - np.log10(dT_wt_vac) - T_decay, np.log10(223) - np.log10(dT_wt_vac))[0]/T_decay

c_alpha_boost = integrate.quad(sigmoid_func,np.log10(223*4) - np.log10(dT_alpha_booster) - T_decay, np.log10(223*4) - np.log10(dT_alpha_booster))[0]/T_decay
c_delta_boost = integrate.quad(sigmoid_func,np.log10(223*4) - np.log10(dT_delta_booster) - T_decay, np.log10(223*4) - np.log10(dT_delta_booster))[0]/T_decay
c_omi_boost = integrate.quad(sigmoid_func,np.log10(223*4) - np.log10(dT_omi_booster) - T_decay, np.log10(223*4) - np.log10(dT_omi_booster))[0]/T_decay
c_wt_boost = integrate.quad(sigmoid_func,np.log10(223*4) - np.log10(dT_wt_booster) - T_decay, np.log10(223*4) - np.log10(dT_wt_booster))[0]/T_decay

c_alpha_alpha = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_alpha_alpha) - T_decay, np.log10(94) - np.log10(dT_alpha_alpha))[0]/T_decay
c_delta_alpha = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_delta_alpha) - T_decay, np.log10(94) - np.log10(dT_delta_alpha))[0]/T_decay
c_omi_alpha = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_omi_alpha) - T_decay, np.log10(94) - np.log10(dT_omi_alpha))[0]/T_decay
c_wt_alpha = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_wt_alpha) - T_decay, np.log10(94) - np.log10(dT_wt_alpha))[0]/T_decay

c_alpha_delta = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_alpha_delta) - T_decay, np.log10(94) - np.log10(dT_alpha_delta))[0]/T_decay
c_delta_delta = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_delta_delta) - T_decay, np.log10(94) - np.log10(dT_delta_delta))[0]/T_decay
c_omi_delta = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_omi_delta) - T_decay, np.log10(94) - np.log10(dT_omi_delta))[0]/T_decay
c_wt_delta = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_wt_delta) - T_decay, np.log10(94) - np.log10(dT_wt_delta))[0]/T_decay

c_alpha_omi = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_alpha_omi) - T_decay, np.log10(94) - np.log10(dT_alpha_omi))[0]/T_decay
c_delta_omi = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_delta_omi) - T_decay, np.log10(94) - np.log10(dT_delta_omi))[0]/T_decay
c_omi_omi = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_omi_omi) - T_decay, np.log10(94) - np.log10(dT_omi_omi))[0]/T_decay
c_wt_omi = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_wt_omi) - T_decay, np.log10(94) - np.log10(dT_wt_omi))[0]/T_decay

c_ba2_boost = integrate.quad(sigmoid_func,np.log10(223*4) - np.log10(dT_ba2_booster) - T_decay, np.log10(223*4) - np.log10(dT_ba2_booster))[0] / T_decay
c_ba45_boost = integrate.quad(sigmoid_func,np.log10(223*4) - np.log10(dT_ba45_booster) - T_decay, np.log10(223*4) - np.log10(dT_ba45_booster))[0] / T_decay

c_ba2_ba1 = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_ba2_ba1) - T_decay, np.log10(94) - np.log10(dT_ba2_ba1))[0]/ T_decay
c_ba45_ba1 = integrate.quad(sigmoid_func,np.log10(94) - np.log10(dT_ba45_ba1) - T_decay, np.log10(94) - np.log10(dT_ba45_ba1))[0]/ T_decay


df = pd.read_csv("output/Fig4_data.txt",'\t',index_col=False)
countries = sorted(list(set(df.country)))
f_hat = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: []))) #f_hat[country][variant][time]
for c in countries:
	df_c = df_s_da.loc[list(df_s_da.country == c)]
	for line in df_c.iterrows():
		line = line[1]
		f_hat_delta = line.s_hat * (1 - line.x_delta)
		f_hat_alpha = - line.s_hat * (line.x_delta)
		f_var_delta = line.s_var * (1 - line.x_delta)**2
		f_var_alpha = line.s_var * (line.x_delta)**2

		f_hat[c]['DELTA'][line.FLAI_time] = [f_hat_delta, f_var_delta]
		f_hat[c]['ALPHA'][line.FLAI_time] = [f_hat_alpha, f_var_alpha]
for c in countries:
	df_c = df_s_od.loc[list(df_s_od.country == c)]
	for line in df_c.iterrows():
		line = line[1]
		f_hat_omi = line.s_hat * (1 - line.x_omi)
		f_hat_delta = - line.s_hat * line.x_omi
		f_var_omi = line.s_var * (1 - line.x_omi)**2
		f_var_delta = line.s_var * line.x_omi**2
		
		f_hat[c]['DELTA'][line.FLAI_time] = [f_hat_delta, f_var_delta]
		f_hat[c]['BA1'][line.FLAI_time] = [f_hat_omi, f_var_omi]
for c in countries:
	df_c = df_s_awt.loc[list(df_s_awt.country == c)]
	for line in df_c.iterrows():
		line = line[1]
		f_hat_alpha = line.s_hat * (line.x_wt)
		f_hat_wt =  -line.s_hat * (1 - line.x_wt)
		f_var_alpha = line.s_var * (1 - line.x_alpha )**2
		f_var_wt = line.s_var * (line.x_alpha)**2
		
		f_hat[c]['ALPHA'][line.FLAI_time] = [f_hat_alpha, f_var_alpha]
		f_hat[c]['WT'][line.FLAI_time] = [f_hat_wt, f_var_wt]
for c in countries:
	df_c = df_s_ba2ba1.loc[list(df_s_ba2ba1.country == c)]
	for line in df_c.iterrows():
		line = line[1]
		f_hat_ba2 = line.s_hat * (1 - line.x_ba2)
		f_hat_ba1 = - line.s_hat * line.x_ba2
		f_var_ba2 = line.s_var * (1 - line.x_ba2)**2
		f_var_ba1 = line.s_var * line.x_ba2**2
		
		f_hat[c]['BA1'][line.FLAI_time] = [f_hat_ba1, f_var_ba1]
		f_hat[c]['BA2'][line.FLAI_time] = [f_hat_ba2, f_var_ba2]

for c in countries:
	df_c = df_s_ba45ba2.loc[list(df_s_ba45ba2.country == c)]
	for line in df_c.iterrows():
		line = line[1]
		f_hat_ba45 = line.s_hat * (1 - line.x_ba45)
		f_hat_ba2 = - line.s_hat * line.x_ba45
		f_var_ba45 = line.s_var * (1 - line.x_ba45)**2
		f_var_ba2 = line.s_var * line.x_ba45**2
		
		f_hat[c]['BA45'][line.FLAI_time] = [f_hat_ba45, f_var_ba45]
		f_hat[c]['BA22'][line.FLAI_time] = [f_hat_ba2, f_var_ba2]


f_av_dict = defaultdict(lambda: defaultdict(lambda: []))
for voc in ['WT','ALPHA','DELTA','BA1','BA2','BA45','BA22']:
	f_av = []
	for c in countries:
		for t in f_hat[c][voc].keys():
			f_av.append([int(t), f_hat[c][voc][t][0], f_hat[c][voc][t][1]])

	f_av = pd.DataFrame(f_av,columns=['time','f_hat','f_var'])
	f_av = f_av.sort_values(by='time',ascending=True)
	tmin = f_av.iloc[0].time
	tmax = f_av.iloc[-1].time

	while tmin + 7 <= tmax:
		f_av_c = f_av.loc[[t >= tmin and t < tmin+7 for t in list(f_av.time)]]
		if len(f_av_c) == 0:
			tmin += 7
			continue
		f_av_dict[voc][int(tmin)+3].append(np.mean(f_av_c.f_hat))
		f_av_dict[voc][int(tmin)+3].append(np.sum(f_av_c.f_var) / len(f_av_c)**2)
		tmin += 7

inch2cm = 2.54

ratio = 1/1.62

mpl.rcParams['axes.linewidth'] = 0.3 #set the value globally


ms2=20
lw=0.75
elw=0.75
mlw=1.5
lp=0.7
lp1=0.2
rot=0
fs = 8
ls = 6

ratio=1/1.5

integrate.quad(sigmoid_func,np.log10(223) - 2, np.log10(223))[0]/2
#==========================================Vaccination

def C_variant(dT, t, T0=223):
	return sigmoid_func(np.log10(T0) - np.log10(np.exp(t/tau_decay)) - np.log10(dT))

T_decay = np.log10(np.exp(1))
R = [0.1,0.2,0.3,0.4,0.5]
dT_list = np.arange(0,6.5,0.1)
f_vac = []
for T in dT_list:
	f_vac.append(sigmoid_func(np.log10(223) - np.log10(2**T)))
f_recov = []
for T in dT_list:
	f_recov.append(sigmoid_func(np.log10(94) - np.log10(2**T)))
f_bst = []
for T in dT_list:
	f_bst.append(sigmoid_func(np.log10(223*4) - np.log10(2**T)))
f_vac=  np.array(f_vac)
f_recov = np.array(f_recov)
f_bst = np.array(f_bst)


T_list = np.linspace(0.0,np.log2(223*6))


fig = plt.figure(figsize=(18/inch2cm,18/inch2cm))
gs0 = gridspec.GridSpec(4,1,figure=fig,hspace=0.35)
gs00 = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs0[0],hspace=0.1,wspace=0.4)
gs01 = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=gs0[1],hspace=0.2,wspace=0.1)
gs02 = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=gs0[2],hspace=0.2,wspace=0.1)
gs03 = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=gs0[3],hspace=0.2,wspace=0.1)

ms=4
ax = fig.add_subplot(gs00[0,0])
plt.plot(T_list, sigmoid_func(np.log10(2**T_list)),'k-')
plt.plot(np.log2(223) - np.log2(dT_alpha_vac), sigmoid_func(np.log10(223) - np.log10(dT_alpha_vac)),'o',color=color_list[1],markersize=ms)
plt.plot(np.log2(223) - np.log2(dT_delta_vac), sigmoid_func(np.log10(223) - np.log10(dT_delta_vac)),'o',color=color_list[2],markersize=ms)
plt.plot(np.log2(223) - np.log2(dT_omi_vac)-0.6, sigmoid_func(np.log10(223) - np.log10(dT_omi_vac))-0.055,'o',color=color_list[6],markersize=ms)
plt.plot(np.log2(223) - np.log2(dT_omi_vac)-0.3, sigmoid_func(np.log10(223) - np.log10(dT_omi_vac))-0.03,'o',color=color_list[5],markersize=ms)
plt.plot(np.log2(223) - np.log2(dT_omi_vac), sigmoid_func(np.log10(223) - np.log10(dT_omi_vac)),'o',color=color_list[4],markersize=ms)
plt.plot(np.log2(223) - np.log2(dT_wt_vac), sigmoid_func(np.log10(223) - np.log10(dT_wt_vac)),'o',color=color_list[0],markersize=ms)


plt.xticks([0,2,4,6,8,10],['0','2','4','6','8','10'])
# plt.xlabel("Antigenic distance, $\\Delta T_i^{\\rm vac}$",fontsize=fs)
plt.ylabel("Cross-immunity , $c^{\\rm vac}$",fontsize=fs)
plt.tick_params(direction='in',labelsize=ls)
plt.xlabel("Titer, $T$",fontsize=fs)

ax = fig.add_subplot(gs00[0,1])
plt.plot(T_list, sigmoid_func(np.log10(2**T_list)),'k-')
plt.plot(np.log2(223*4) - np.log2(dT_alpha_booster), sigmoid_func(np.log10(223*4) - np.log10(dT_alpha_booster)),'o',color=color_list[1],markersize=ms)
plt.plot(np.log2(223*4) - np.log2(dT_delta_booster), sigmoid_func(np.log10(223*4) - np.log10(dT_delta_booster)),'o',color=color_list[2],markersize=ms)
plt.plot(np.log2(223*4) - np.log2(dT_ba45_booster), sigmoid_func(np.log10(223*4) - np.log10(dT_ba45_booster)),'o',color=color_list[6],markersize=ms)
plt.plot(np.log2(223*4) - np.log2(dT_ba2_booster)+0.2, sigmoid_func(np.log10(223*4) - np.log10(dT_ba2_booster))+0.012,'o',color=color_list[5],markersize=ms)
plt.plot(np.log2(223*4) - np.log2(dT_omi_booster), sigmoid_func(np.log10(223*4) - np.log10(dT_omi_booster)),'o',color=color_list[4],markersize=ms)
plt.plot(np.log2(223*4) - np.log2(dT_wt_booster), sigmoid_func(np.log10(223*4) - np.log10(dT_wt_booster)),'o',color=color_list[0],markersize=ms)

plt.xticks([0,2,4,6,8,10],['0','2','4','6','8','10'])
plt.ylabel("Cross-immunity, $c^{\\rm bst}$",fontsize=fs)
plt.tick_params(direction='in',labelsize=ls)
plt.xlabel("Titer, $T$",fontsize=fs)

ax = fig.add_subplot(gs00[0,2])
plt.plot(T_list, sigmoid_func(np.log10(2**T_list)),'k-')
plt.plot(np.log2(94) - np.log2(dT_alpha_omi), sigmoid_func(np.log10(94) - np.log10(dT_alpha_omi)),'o',color=color_list[1],markersize=ms)
plt.plot(np.log2(94) - np.log2(dT_delta_omi), sigmoid_func(np.log10(94) - np.log10(dT_delta_omi)),'o',color=color_list[2],markersize=ms)
plt.plot(np.log2(94) - np.log2(dT_ba45_ba1), sigmoid_func(np.log10(94) - np.log10(dT_ba45_ba1)),'o',color=color_list[6],markersize=ms)
plt.plot(np.log2(94) - np.log2(dT_ba2_ba1), sigmoid_func(np.log10(94) - np.log10(dT_ba2_ba1)),'o',color=color_list[5],markersize=ms)
plt.plot(np.log2(94) - np.log2(dT_omi_omi), sigmoid_func(np.log10(94) - np.log10(dT_omi_omi)),'o',color=color_list[4],markersize=ms)
plt.plot(np.log2(94) - np.log2(dT_wt_omi), sigmoid_func(np.log10(94) - np.log10(dT_wt_omi)),'o',color=color_list[0],markersize=ms)
plt.xlabel("Titer, $T$",fontsize=fs)
plt.xticks([0,2,4,6,8,10],['0','2','4','6','8','10'])
plt.ylabel("Cross-immunity, $c^{o}$",fontsize=fs)
plt.tick_params(direction='in',labelsize=ls)

ax = fig.add_subplot(gs01[0,0])
df_R = pd.read_csv("output/R_average.txt",'\t',index_col=False)
df_c = df_R.loc[[int(t) < Time.dateToCoordinate("2022-05-15") and int(t) > Time.dateToCoordinate("2020-12-31") for t in list(df_R.time)]]

t_range = list(df_c.time)
plt.plot(t_range, df_c.R_vac,'k-', linewidth=1.0)
plt.plot(t_range, df_c.R_boost,'k--', linewidth=1.0)
plt.plot(t_range, df_c.R_delta,'-',color=color_list[2], linewidth=1.0)	
plt.plot(t_range, df_c.R_ba1,'-',color=color_list[4], linewidth=1.0)
plt.plot(t_range, df_c.R_ba2,'-',color=color_list[5], linewidth=1.0)

plt.ylabel("Immune weight, $Q_k(t)$",fontsize=fs)
xtick_labels = ['2021-01-01','2021-05-01','2021-09-01','2022-01-01','2022-05-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Jan. $\'$21','May $\'$ 21','Sep. $\'$21','Jan. $\'$22','May $\'$22']
ax.set_xticks(xtick_pos,['','','','',''],rotation=rot,ha='right',fontsize=fs)
plt.tick_params(direction='in')
plt.tick_params(direction='in',labelsize=ls)


ax = fig.add_subplot(gs02[0,0])
df_R = pd.read_csv("output/R_average.txt",'\t',index_col=False)

df_c = df_R.loc[[int(t) < Time.dateToCoordinate("2022-05-15") and int(t) > Time.dateToCoordinate("2020-12-31") for t in list(df_R.time)]]
# plt.plot(df_c.time, np.zeros(len(df_c.time)),'k-',linewidth=0.5)
df_cc = df_c.loc[list(df_c.x_alpha > 0.01)]
df_cc = df_cc.loc[list(df_cc.x_delta > 0.01)]
df_cc_ant = df_c.loc[list(df_c.time <= df_cc.iloc[0].time)]

gamma_vac = 1.15
gamma_inf = 2.3
dC_vac = gamma_vac * (np.array(df_cc.C_av) - np.array(df_cc.C_dv))
dC_alpha = gamma_inf * (np.array(df_cc.C_aa) - np.array(df_cc.C_da))
dC_delta = gamma_inf * (np.array(df_cc.C_ad) - np.array(df_cc.C_dd))
plt.fill_between(df_cc.time,  np.zeros(len(df_cc.time)), dC_vac  + dC_alpha + dC_delta , linewidth=0.0, color=color_list[2],alpha=0.3)
plt.plot(df_cc.time, dC_vac + dC_alpha + dC_delta, color = color_list[2], linewidth=1.0)
dC_vac = gamma_vac * (np.array(df_cc_ant.C_av) - np.array(df_cc_ant.C_dv))
dC_alpha = gamma_inf * (np.array(df_cc_ant.C_aa) - np.array(df_cc_ant.C_da))
dC_delta = gamma_inf * (np.array(df_cc_ant.C_ad) - np.array(df_cc_ant.C_dd))
# plt.fill_between(df_cc_ant.time,  np.zeros(len(df_cc_ant.time)), dC_vac  + dC_alpha + dC_delta , linewidth=0.0, color=color_list[2],alpha=0.2)
plt.plot(df_cc_ant.time, dC_vac + dC_alpha + dC_delta,'-', color = color_list[2], linewidth=1.0)



df_c = df_R.loc[[int(t) < Time.dateToCoordinate("2022-05-15") and int(t) > Time.dateToCoordinate("2020-12-31") for t in list(df_R.time)]]
df_cc = df_c.loc[list(df_c.x_delta > 0.01)]
df_cc = df_cc.loc[list(df_cc.x_ba1 > 0.01)]
df_cc_ant = df_c.loc[list(df_c.time <= df_cc.iloc[0].time)]
df_cc_ant = df_cc_ant.loc[list(df_cc_ant.time > Time.dateToCoordinate("2021-06-15"))]

gamma_inf = 0.58
gamma_vac = 0.29
dC_vac = gamma_vac * (np.array(df_cc.C_dv0) - np.array(df_cc.C_ov0))
dC_bst = gamma_vac * (np.array(df_cc.C_db) - np.array(df_cc.C_ob) + np.array(df_cc.C_dv) - np.array(df_cc.C_ov)  - (np.array(df_cc.C_dv0) - np.array(df_cc.C_ov0)))
dC_omi = gamma_inf * (np.array(df_cc.C_do) - np.array(df_cc.C_oo))
dC_delta = gamma_inf * (np.array(df_cc.C_dd) - np.array(df_cc.C_od))
plt.fill_between(df_cc.time, np.zeros(len(df_cc.time)), dC_vac + dC_bst + dC_omi + dC_delta , linewidth=0.0, color=color_list[4],alpha=0.3)
plt.plot(df_cc.time, dC_vac + dC_bst + dC_omi + dC_delta, color=color_list[4], linewidth=1.0)
dC_vac = gamma_vac * (np.array(df_cc_ant.C_dv0) - np.array(df_cc_ant.C_ov0))
dC_bst = gamma_vac * (np.array(df_cc_ant.C_db) - np.array(df_cc_ant.C_ob) + np.array(df_cc_ant.C_dv) - np.array(df_cc_ant.C_ov)  - (np.array(df_cc_ant.C_dv0) - np.array(df_cc_ant.C_ov0)))
dC_omi = gamma_inf * (np.array(df_cc_ant.C_do) - np.array(df_cc_ant.C_oo))
dC_delta = gamma_inf * (np.array(df_cc_ant.C_dd) - np.array(df_cc_ant.C_od))
# plt.fill_between(df_cc_ant.time, np.zeros(len(df_cc_ant.time)), dC_vac + dC_bst + dC_omi + dC_delta , linewidth=0.0, color=color_list[4],alpha=0.2)
plt.plot(df_cc_ant.time, dC_vac + dC_bst + dC_omi + dC_delta, '-',color = color_list[4], linewidth=1.0)


df_c = df_R.loc[[int(t) < Time.dateToCoordinate("2022-05-15") and int(t) > Time.dateToCoordinate("2020-12-31") for t in list(df_R.time)]]
df_cc = df_c.loc[list(df_c.x_ba1 > 0.01)]
df_cc = df_cc.loc[list(df_cc.x_ba2 > 0.01)]
df_cc_ant = df_c.loc[list(df_c.time <= df_cc.iloc[0].time)]
df_cc_ant = df_cc_ant.loc[list(df_cc_ant.time > Time.dateToCoordinate("2022-01-01"))]

dC_bst = gamma_vac * (np.array(df_cc.C_a2b) - np.array(df_cc.C_ob))
dC_o1 = gamma_inf * (np.array(df_cc.C_a1a1) - np.array(df_cc.C_a2a1))
dC_o2 = gamma_inf * (np.array(df_cc.C_a1a2) - np.array(df_cc.C_a2a2))
plt.fill_between(df_cc.time, np.zeros(len(df_cc.time)),dC_bst + dC_o1 + dC_o2,linewidth=0.0,color=color_list[5], alpha=0.3)
plt.plot(df_cc.time, dC_bst + dC_o1 + dC_o2, color=color_list[5], linewidth=1.0)

dC_bst = gamma_vac * (np.array(df_cc_ant.C_a2b) - np.array(df_cc_ant.C_ob))
dC_o1 = gamma_inf * (np.array(df_cc_ant.C_a1a1) - np.array(df_cc_ant.C_a2a1))
dC_o2 = gamma_inf * (np.array(df_cc_ant.C_a1a2) - np.array(df_cc_ant.C_a2a2))
# plt.fill_between(df_cc_ant.time, np.zeros(len(df_cc_ant.time)),dC_bst + dC_o1 + dC_o2,linewidth=0.0,color=color_list[5], alpha=0.2)
plt.plot(df_cc_ant.time, dC_bst + dC_o1 + dC_o2, '-', color=color_list[5], linewidth=1.0)



df_c = df_R.loc[[int(t) < Time.dateToCoordinate("2022-05-15") and int(t) > Time.dateToCoordinate("2020-12-31") for t in list(df_R.time)]]
df_cc = df_c.loc[list(df_c.x_ba2 > 0.01)]
df_cc = df_cc.loc[list(df_cc.x_ba45 > 0.01)]
df_cc_ant = df_c.loc[list(df_c.time <= df_cc.iloc[0].time)]
df_cc_ant = df_cc_ant.loc[list(df_cc_ant.time > Time.dateToCoordinate("2022-01-28"))]

dC_bst = gamma_vac * (np.array(df_cc.C_a2b) - np.array(df_cc.C_a45b))
dC_o1 = gamma_inf * (np.array(df_cc.C_a2a1) - np.array(df_cc.C_a45a1))
dC_o2 = gamma_inf * (np.array(df_cc.C_a2a2) - np.array(df_cc.C_a45a2))
plt.fill_between(df_cc.time, np.zeros(len(df_cc.time)),dC_bst + dC_o1 + dC_o2,linewidth=0.0,color=color_list[6], alpha=0.3)
plt.plot(df_cc.time, dC_bst + dC_o1 + dC_o2, color=color_list[6], linewidth=1.0)

dC_bst = gamma_vac * (np.array(df_cc_ant.C_a2b) - np.array(df_cc_ant.C_a45b))
dC_o1 = gamma_inf * (np.array(df_cc_ant.C_a2a1) - np.array(df_cc_ant.C_a45a1))
dC_o2 = gamma_inf * (np.array(df_cc_ant.C_a2a2) - np.array(df_cc_ant.C_a45a2))
# plt.fill_between(df_cc_ant.time, np.zeros(len(df_cc_ant.time)),dC_bst + dC_o1 + dC_o2,linewidth=0.0,color=color_list[6], alpha=0.2)
plt.plot(df_cc_ant.time, dC_bst + dC_o1 + dC_o2,'-', color=color_list[6], linewidth=1.0)


plt.ylabel("Antigenic selection, $s_{\\rm ag}(t)$",fontsize=fs)
xtick_labels = ['2021-01-01','2021-05-01','2021-09-01','2022-01-01','2022-05-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Jan. $\'$21','May $\'$ 21','Sep. $\'$21','Jan. $\'$22','May $\'$22']
ax.set_xticks(xtick_pos,['','','','',''],rotation=rot,ha='right',fontsize=fs)
plt.tick_params(direction='in',labelsize=ls)
plt.ylim([0.0,0.12])
plt.xlim([Time.dateToCoordinate("2020-12-31"), Time.dateToCoordinate("2022-05-15")-1])


ax = fig.add_subplot(gs03[0,0])
df = pd.read_csv("output/Fig4_data.txt",'\t',index_col=False)
plt.plot(df_c.time, np.zeros(len(df_c.time)),'k-',linewidth=0.5)
ms=2
fitness_plot = []
gamma_vac = 1.15
gamma_inf = 2.3
dS_0_od = 0.059
dS_0_da = 0.048
dS_0_awt = 0.081
dS_0_ba2 = 0.075
dS_0_ba45 = 0.057
for line in df_c.iterrows():
	line = line[1]
	t = line.time
	if t > Time.dateToCoordinate("2021-09-01"):
		gamma_vac = 0.29
		gamma_inf = 0.58

	F_wt = - gamma_vac * line.C_wtv - gamma_vac * line.C_wtb -  gamma_inf * line.C_wta  -  gamma_inf * line.C_wtd - gamma_inf * line.C_wto
	F_alpha = - gamma_vac * line.C_av - gamma_vac * line.C_ab -  gamma_inf * line.C_aa  -  gamma_inf * line.C_ad - gamma_inf * line.C_ao
	F_delta = - gamma_vac * line.C_dv - gamma_vac * line.C_db -  gamma_inf * line.C_da   - gamma_inf * line.C_dd - gamma_inf * line.C_do
	F_ba1   = - gamma_vac * line.C_ov - gamma_vac * line.C_ob -  gamma_inf * line.C_oa   - gamma_inf * line.C_od - gamma_inf * line.C_a1a1 - gamma_inf * line.C_a1a2
	F_ba2   = - gamma_vac * line.C_ov - gamma_vac * line.C_a2b - gamma_inf * line.C_oa  -  gamma_inf * line.C_od - gamma_inf * line.C_a2a1 - gamma_inf * line.C_a2a2
	F_ba45   =- gamma_vac * line.C_ov - gamma_vac * line.C_a45b -gamma_inf * line.C_oa -   gamma_inf * line.C_od - gamma_inf * line.C_a45a1 - gamma_inf * line.C_a45a2

	F_alpha  += dS_0_awt
	F_delta  += dS_0_awt + dS_0_da
	F_ba1    +=	dS_0_awt + dS_0_da + dS_0_od
	F_ba2    +=	dS_0_awt + dS_0_da + dS_0_od + dS_0_ba2
	F_ba45   +=	dS_0_awt + dS_0_da + dS_0_od + dS_0_ba2 + dS_0_ba45

	x = [line.x_alpha + line.x_voc, line.x_delta, line.x_ba1, line.x_ba2, line.x_ba45]
	x.append(1-np.sum(x))
	
	F_av = np.dot(np.array(x),np.array([F_alpha, F_delta, F_ba1, F_ba2, F_ba45, F_wt]))

	fitness_plot.append([t, line.x_wt, line.x_alpha, line.x_delta, line.x_ba1, line.x_ba2, line.x_ba45, F_wt - F_av, F_alpha - F_av, F_delta - F_av, F_ba1 - F_av, F_ba2 - F_av, F_ba45 - F_av])
	# fitness_plot.append([c,t, line.x_alpha, line.x_delta, line.x_omi, line.x_ba1, line.x_ba2, line.x_ba45, F0_alpha, F0_delta, F0_omi, F_alpha, F_delta, F_ba1, F_ba2, F_ba45])
fitness_plot = pd.DataFrame(fitness_plot, columns=['t','x_wt','x_alpha','x_delta','x_ba1','x_ba2','x_ba45','f_wt','f_alpha','f_delta','f_ba1','f_ba2','f_ba45'])

f_wt = fitness_plot.loc[list(fitness_plot.x_wt > 0.05)]
f_alpha = fitness_plot.loc[list(fitness_plot.x_alpha > 0.01)]
f_delta = fitness_plot.loc[list(fitness_plot.x_delta > 0.01)]
f_ba1 = fitness_plot.loc[list(fitness_plot.x_ba1 > 0.01)]
f_ba2 = fitness_plot.loc[list(fitness_plot.x_ba2 > 0.01)]
f_ba45 = fitness_plot.loc[list(fitness_plot.x_ba45 > 0.01)]

plt.plot(f_wt.t, f_wt.f_wt, '-',color=color_list[0], linewidth=1.0)
plt.plot(f_alpha.t, f_alpha.f_alpha, color=color_list[1], linewidth=1.0)
plt.plot(f_delta.t, f_delta.f_delta,color=color_list[2], linewidth=1.0)
plt.plot(f_ba1.t, f_ba1.f_ba1,color=color_list[4], linewidth=1.0)
plt.plot(f_ba2.t, f_ba2.f_ba2,color=color_list[5], linewidth=1.0)
plt.plot(f_ba45.t, f_ba45.f_ba45,color=color_list[6], linewidth=1.0)

df_c = df_R.loc[[int(t) < Time.dateToCoordinate("2022-05-15") and int(t) > Time.dateToCoordinate("2020-12-31") for t in list(df_R.time)]]
df_cc = df_c.loc[list(df_c.x_alpha > 0.01)]
df_cc = df_cc.loc[list(df_cc.x_delta > 0.01)]
plt.fill_between(df_cc.time, np.ones(len(df_cc)) * -0.15, np.ones(len(df_cc)) * 0.15,color = color_list[2],alpha=0.3, linewidth=0.0)

df_cc = df_c.loc[list(df_c.x_delta > 0.01)]
df_cc = df_cc.loc[list(df_cc.x_ba1 > 0.01)]
plt.fill_between(df_cc.time, np.ones(len(df_cc)) * -0.15, np.ones(len(df_cc)) * 0.15,color = color_list[4],alpha=0.3, linewidth=0.0)

df_cc = df_c.loc[list(df_c.x_ba1 > 0.01)]
df_cc = df_cc.loc[list(df_cc.x_ba2 > 0.01)]
plt.fill_between(df_cc.time, np.ones(len(df_cc)) * -0.15, np.ones(len(df_cc)) * 0.15,color = color_list[5],alpha=0.3, linewidth=0.0)

df_cc = df_c.loc[list(df_c.x_ba2 > 0.01)]
df_cc = df_cc.loc[list(df_cc.x_ba45 > 0.01)]
plt.fill_between(df_cc.time, np.ones(len(df_cc)) * -0.15, np.ones(len(df_cc)) * 0.15,color = color_list[6],alpha=0.3, linewidth=0.0)

voc2color={'ALPHA':color_list[1],'WT':color_list[0],'DELTA':color_list[2],'BA1':color_list[4],'BA2':color_list[5],'BA22':color_list[5],'BA45':color_list[6]}
for voc in f_av_dict:
	if voc == 'ALPHA' or voc == 'DELTA':
		times = sorted(list(f_av_dict[voc].keys()))
		for bb in range(len(times)-1):
			if times[bb+1] - times[bb] > 8:
				hak = bb
		times0 = times[:hak+1]
		times1 = times[hak+1:]
		plt.errorbar(times0, [f_av_dict[voc][t][0] for t in times0],yerr = [np.sqrt(f_av_dict[voc][t][1]) for t in times0],marker='o',linestyle='',markersize=ms,color=voc2color[voc])
		plt.errorbar(times1, [f_av_dict[voc][t][0] for t in times1],yerr = [np.sqrt(f_av_dict[voc][t][1]) for t in times1],marker='o',linestyle='',markersize=ms,color=voc2color[voc])
	else:
		times = sorted(list(f_av_dict[voc].keys()))
		plt.errorbar(times, [f_av_dict[voc][t][0] for t in times],yerr = [np.sqrt(f_av_dict[voc][t][1]) for t in times],marker='o',linestyle='',markersize=ms,color=voc2color[voc])

xtick_labels = ['2021-01-01','2021-05-01','2021-09-01','2022-01-01','2022-05-01']
xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
xtick_labels = ['Jan. $\'$21','May $\'$ 21','Sep. $\'$21','Jan. $\'$22','May $\'$22']
ax.set_xticks(xtick_pos,xtick_labels,rotation=rot,ha='center',fontsize=fs)
plt.ylabel("Fitness gap, $\\delta f(t)$",fontsize=fs)
plt.tick_params(direction='in',labelsize=ls)
plt.ylim([-0.15,0.15])
plt.xlim([Time.dateToCoordinate("2020-12-31"), Time.dateToCoordinate("2022-05-15")-1])

legend_elements = []
legend_elements.append(Line2D([],[],marker='',markersize=ms,color='k',linestyle='-',label='Vaccination', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='',markersize=ms,color='k',linestyle='--',label='Booster', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=color_list[1],linestyle='-',label='Alpha', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=color_list[2],linestyle='-',label='Delta', linewidth=2.0))
# legend_elements.append(Line2D([],[],marker='o',markersize=ms,color='fuchsia',linestyle='',label='Omicron', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=color_list[4],linestyle='-',label='Omicron BA.1', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=color_list[5],linestyle='-',label='Omicron BA.2', linewidth=2.0))
legend_elements.append(Line2D([],[],marker='o',markersize=ms,color=color_list[6],linestyle='-',label='Omicron BA.4/5', linewidth=2.0))

plt.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1.05,1.2),prop={'size':ls})
plt.subplots_adjust(right=0.87)

plt.savefig("figures/Fig4.pdf")
plt.close()


