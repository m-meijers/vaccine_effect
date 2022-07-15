import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from flai.util.Time import Time
import json
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from copy import copy
import scipy.integrate as si
import scipy.stats as ss
import scipy.optimize as so
from collections import defaultdict
import copy
import glob
import sys
from time import time

df = pd.read_csv("output/Fig4_data.txt",'\t',index_col=False)
countries = sorted(list(set(df.country)))
df_s_da = pd.read_csv("output/s_hat_delta_alpha.txt",sep='\t',index_col=False)
df_s_od = pd.read_csv("output/s_hat_omi_delta.txt",sep='\t',index_col=False)
df_s_awt = pd.read_csv("output/s_hat_alpha_wt.txt",sep='\t',index_col=False)
df_s_ba2ba1 = pd.read_csv("output/s_hat_ba2_ba1.txt",sep='\t',index_col=False)
df_s_ba45ba2 = pd.read_csv("output/s_hat_ba45_ba2.txt",sep='\t',index_col=False)
df_od = pd.read_csv("output/df_od_result.txt",'\t',index_col=False)
df_da = pd.read_csv("output/df_da_result.txt",'\t',index_col=False)

gamma_vac_ad = 1.15
gamma_inf_ad = 2.3
gamma_vac_od = 0.29
gamma_inf_od = 0.58
ms = 5
rot = 20
fs = 12
ls = 10

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


plt.figure(figsize=(25, 5 * len(countries)))
index = 1
for c in countries:
	df_c = df.loc[list(df.country == c)]
	df_c = df_c.loc[[int(t) < Time.dateToCoordinate("2022-05-22") and int(t) > Time.dateToCoordinate("2020-12-31") for t in list(df_c.time)]]
	times = np.array(df_c.time)

	ax = plt.subplot(len(countries),2,index)
	plt.plot(times, df_c.R_vac,'k-')
	plt.plot(times, df_c.R_boost,'k--')
	plt.plot(times, df_c.R_delta,'-',color=color_list[2], linewidth=1.0)	
	plt.plot(times, df_c.R_ba1,'-',color=color_list[4], linewidth=1.0)
	plt.plot(times, df_c.R_ba2,'-',color=color_list[5], linewidth=1.0)
	# plt.plot(times, df_c.R_ba45,'-',color='darkorchid')
	plt.title(c)
	plt.ylabel("Immune weight, $Q_k(t)$",fontsize=12)
	xtick_labels = ['2021-01-01','2021-05-01','2021-09-01','2022-01-01','2022-05-01']
	xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
	xtick_labels = ['Jan. $\'$21','May $\'$ 21','Sep. $\'$21','Jan. $\'$22','May $\'$22']
	ax.set_xticks(xtick_pos,xtick_labels,rotation=0,ha='center',fontsize=fs)
	plt.tick_params(direction='in')
	index += 1

	ax = plt.subplot(len(countries),2,index)
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
			gamma_vac = 0.3
			gamma_inf = 0.6
		
		F_wt = - gamma_vac * line.C_wtv - gamma_vac * line.C_wtb -  gamma_inf * line.C_wta  -  gamma_inf * line.C_wtd - gamma_inf * line.C_wto
		F_alpha = - gamma_vac * line.C_av - gamma_vac * line.C_ab -  gamma_inf * line.C_aa  -  gamma_inf * line.C_ad - gamma_inf * line.C_ao
		F_delta = - gamma_vac * line.C_dv - gamma_vac * line.C_db -  gamma_inf * line.C_da   - gamma_inf * line.C_dd - gamma_inf * line.C_do
		F_ba1   = - gamma_vac * line.C_ov - gamma_vac * line.C_ob -  gamma_inf * line.C_oa   - gamma_inf * line.C_od - gamma_inf * line.C_a1a1 - gamma_inf * line.C_a1a2
		F_ba2   = - gamma_vac * line.C_ov - gamma_vac * line.C_a2b - gamma_inf * line.C_oa  -  gamma_inf * line.C_od - gamma_inf * line.C_a2a1 - gamma_inf * line.C_a2a2
		F_ba45   =- gamma_vac * line.C_ov - gamma_vac * line.C_a45b -gamma_inf * line.C_oa -   gamma_inf * line.C_od - gamma_inf * line.C_a45a1 - gamma_inf * line.C_a45a2

		# F_wt = F_alpha
		F_alpha  += dS_0_awt
		F_delta  += dS_0_awt + dS_0_da
		F_ba1    +=	dS_0_awt + dS_0_da + dS_0_od
		F_ba2    +=	dS_0_awt + dS_0_da + dS_0_od + dS_0_ba2
		F_ba45   +=	dS_0_awt + dS_0_da + dS_0_od + dS_0_ba2 + dS_0_ba45

		x = [line.x_alpha + line.x_voc, line.x_delta, line.x_ba1, line.x_ba2, line.x_ba45]
		x.append(1 - np.sum(x))
		F_av = np.dot(np.array(x),np.array([F_alpha, F_delta, F_ba1, F_ba2, F_ba45, F_wt]))

		fitness_plot.append([c,t, line.x_wt,line.x_voc,line.x_alpha, line.x_delta, line.x_ba1, line.x_ba2, line.x_ba45,F_wt - F_av, F_alpha - F_av, F_delta - F_av, F_ba1 - F_av, F_ba2 - F_av, F_ba45 - F_av])
	fitness_plot = pd.DataFrame(fitness_plot, columns=['c','t','x_wt','x_voc','x_alpha','x_delta','x_ba1','x_ba2','x_ba45','f_wt','f_alpha','f_delta','f_ba1','f_ba2','f_ba45'])

	f_wt = fitness_plot.loc[list(fitness_plot.x_wt > 0.05)]
	f_alpha = fitness_plot.loc[list(fitness_plot.x_alpha > 0.01)]
	f_delta = fitness_plot.loc[list(fitness_plot.x_delta > 0.01)]
	if c =='ITALY':
		f_delta = f_delta.loc[list(f_delta.t > 44287)]
		f_alpha = f_alpha.loc[list(f_alpha.t < 44408)]
	f_ba1 = fitness_plot.loc[list(fitness_plot.x_ba1 > 0.01)]
	f_ba2 = fitness_plot.loc[list(fitness_plot.x_ba2 > 0.01)]
	f_ba45 = fitness_plot.loc[list(fitness_plot.x_ba45 > 0.01)]

	plt.title(c)
	plt.plot(f_wt.t, f_wt.f_wt, '-',color=color_list[0], linewidth=1.0)
	plt.plot(f_alpha.t, f_alpha.f_alpha, color=color_list[1], linewidth=1.0)
	plt.plot(f_delta.t, f_delta.f_delta,color=color_list[2], linewidth=1.0)
	plt.plot(f_ba1.t, f_ba1.f_ba1,color=color_list[4], linewidth=1.0)
	plt.plot(f_ba2.t, f_ba2.f_ba2,color=color_list[5], linewidth=1.0)
	plt.plot(f_ba45.t, f_ba45.f_ba45,color=color_list[6], linewidth=1.0)


	if c not in ['BELGIUM','FINLAND','FRANCE']:
		alpha_times = list(sorted(f_hat[c]['ALPHA'].keys()))
		for bb in range(len(alpha_times)-1):
			if alpha_times[bb+1] - alpha_times[bb] > 8:
				hak = bb
		alpha_times0 = alpha_times[:hak+1]
		alpha_times1 = alpha_times[hak+1:]
		plt.errorbar(alpha_times0, [f_hat[c]['ALPHA'][t][0] for t in alpha_times0],yerr = [np.sqrt(f_hat[c]['ALPHA'][t][1]) for t in alpha_times0],marker='o',linestyle='--',color=color_list[1])
		plt.errorbar(alpha_times1, [f_hat[c]['ALPHA'][t][0] for t in alpha_times1],yerr = [np.sqrt(f_hat[c]['ALPHA'][t][1]) for t in alpha_times1],marker='o',linestyle='--',color=color_list[1])
	else:
		alpha_times = list(sorted(f_hat[c]['ALPHA'].keys()))
		plt.errorbar(alpha_times, [f_hat[c]['ALPHA'][t][0] for t in alpha_times],yerr = [np.sqrt(f_hat[c]['ALPHA'][t][1]) for t in alpha_times],marker='o',linestyle='--',color=color_list[1])
	alpha_times = list(sorted(f_hat[c]['DELTA'].keys()))
	for bb in range(len(alpha_times)-1):
		if alpha_times[bb+1] - alpha_times[bb] > 8:
			hak = bb
	alpha_times0 = alpha_times[:hak+1]
	alpha_times1 = alpha_times[hak+1:]
	plt.errorbar(alpha_times0, [f_hat[c]['DELTA'][t][0] for t in alpha_times0],yerr = [np.sqrt(f_hat[c]['DELTA'][t][1]) for t in alpha_times0],marker='o',linestyle='--',color=color_list[2])
	plt.errorbar(alpha_times1, [f_hat[c]['DELTA'][t][0] for t in alpha_times1],yerr = [np.sqrt(f_hat[c]['DELTA'][t][1]) for t in alpha_times1],marker='o',linestyle='--',color=color_list[2])
	alpha_times = list(sorted(f_hat[c]['WT'].keys()))
	plt.errorbar(alpha_times, [f_hat[c]['WT'][t][0] for t in alpha_times],yerr = [np.sqrt(f_hat[c]['WT'][t][1]) for t in alpha_times],marker='o',linestyle='--',color=color_list[0])
	alpha_times = list(sorted(f_hat[c]['BA1'].keys()))
	plt.errorbar(alpha_times, [f_hat[c]['BA1'][t][0] for t in alpha_times],yerr = [np.sqrt(f_hat[c]['BA1'][t][1]) for t in alpha_times],marker='o',linestyle='--',color=color_list[4])
	alpha_times = list(sorted(f_hat[c]['BA2'].keys()))
	plt.errorbar(alpha_times, [f_hat[c]['BA2'][t][0] for t in alpha_times],yerr = [np.sqrt(f_hat[c]['BA2'][t][1]) for t in alpha_times],marker='o',linestyle='--',color=color_list[5])
	
	alpha_times = list(sorted(f_hat[c]['BA22'].keys()))
	plt.errorbar(alpha_times, [f_hat[c]['BA22'][t][0] for t in alpha_times],yerr = [np.sqrt(f_hat[c]['BA22'][t][1]) for t in alpha_times],marker='o',linestyle='--',color=color_list[5])
	alpha_times = list(sorted(f_hat[c]['BA45'].keys()))
	plt.errorbar(alpha_times, [f_hat[c]['BA45'][t][0] for t in alpha_times],yerr = [np.sqrt(f_hat[c]['BA45'][t][1]) for t in alpha_times],marker='o',linestyle='--',color=color_list[6])


	xtick_labels = ['2021-01-01','2021-05-01','2021-09-01','2022-01-01','2022-05-01']
	xtick_pos = [Time.dateToCoordinate(t) for t in xtick_labels]
	xtick_labels = ['Jan. $\'$21','May $\'$ 21','Sep. $\'$21','Jan. $\'$22','May $\'$22']
	ax.set_xticks(xtick_pos,xtick_labels,rotation=0,ha='center',fontsize=fs)
	plt.ylabel("Fitness gap, $\\delta f(t)$",fontsize=fs)
	plt.tick_params(direction='in')

	index += 1

plt.subplots_adjust(bottom=0.02,top=0.98)
plt.savefig("figures/FigS4.pdf")
plt.close()


