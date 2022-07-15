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

def sigmoid_func(t,mean,s):
	val = 1 / (1 + np.exp(-s * (t - mean)))
	return val

def clean_vac(times,vac_vec):
	vac_rep = []
	vac_times = []
	i = 0
	while np.isnan(vac_vec[i]):
		vac_rep.append(0.0)
		vac_times.append(times[i])
		i += 1 
	for i in range(len(vac_vec)):
		if np.isnan(vac_vec[i]):
			continue
		else:
			vac_rep.append(vac_vec[i])
			vac_times.append(times[i])
	v_func = interp1d(vac_times,vac_rep,fill_value = 'extrapolate')
	v_interp = v_func(times)
	return v_interp

def smooth(times,vec,dt):
	smooth = []
	times_smooth = []
	for i in range(dt,len(times)-dt):
		smooth.append(np.mean(vec[i-dt:i+dt]))
		times_smooth.append(times[i])
	return smooth, times_smooth

#Get trajectories for vaccination, delta infection and booster
files = glob.glob("DATA/2022_06_22/freq_traj_*")
countries = [f.split("_")[-1][:-5] for f in files]
meta_df = pd.read_csv("DATA/clean_data.txt",sep='\t',index_col=False)
country2immuno2time = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
country_min_count = defaultdict(lambda: [])
df_s_od = pd.read_csv("output/s_hat_omi_delta.txt",sep='\t',index_col='country')
df_s_da = pd.read_csv("output/s_hat_delta_alpha.txt",sep='\t',index_col='country')
countries_da = sorted(list(set(df_s_da.index)))
countries_od = sorted(list(set(df_s_od.index)))

x_limit = 0.01
for country in countries_od:
	with open("DATA/2022_06_22/freq_traj_" + country.upper() + ".json",'r') as f:
		freq_traj = json.load(f)
	with open("DATA/2022_06_22/multiplicities_" + country.upper() + ".json",'r') as f:
		counts = json.load(f)
	with open("DATA/2022_06_22/multiplicities_Z_" + country.upper() + ".json",'r') as f:
		Z = json.load(f)

	meta_country= meta_df.loc[list(meta_df['location'] == country[0] + country[1:].lower())]
	if len(country) <= 3:
		meta_country= meta_df.loc[list(meta_df['location'] == country)]
	if country == 'SOUTHKOREA':
		meta_country= meta_df.loc[list(meta_df['location'] == 'SouthKorea')]
	if country == 'SOUTHAFRICA':
		meta_country= meta_df.loc[list(meta_df['location'] == 'SouthAfrica')]
	meta_country.index = meta_country['FLAI_time']
	df_c = df_s_od.loc[list(df_s_od.index == country)]

	if max(list(freq_traj['DELTA'].values())) > 0.5 and max(list(freq_traj['OMICRON'].values())) > 0.5:
		dates_delta = list(counts['DELTA'].keys())
		dates_omi = list(counts['OMICRON'].keys())
		dates_delta = [int(a) for a in dates_delta]
		dates_omi = [int(a) for a in dates_omi]
		dates_delta = sorted(dates_delta)
		dates_omi = sorted(dates_omi)
		dates_omi = [a for a in dates_omi if a < 44620 and a > 44469]
		dates_delta = [a for a in dates_delta if a < 44620 and a > 44377]
		tmin = min(set(dates_delta).intersection(set(dates_omi)))
		tmax = max(set(dates_delta).intersection(set(dates_omi)))
		t_range = np.arange(tmin,tmax)
		omi_count = [int(np.exp(counts['OMICRON'][str(a)])) for a in t_range]
		delta_count = [int(np.exp(counts['DELTA'][str(a)])) for a in t_range]
		N_tot = np.array(omi_count) + np.array(delta_count)
		check = 'not_okay'
		for t in t_range:
			x_d = delta_count[t-tmin] / N_tot[t-tmin]
			x_o = omi_count[t-tmin] / N_tot[t-tmin]
			if x_o > x_limit:
				tminnew = t
				check = 'okay'
				break
		tmaxnew = tmax
		for t in t_range:
			x_d = delta_count[t-tmin] / N_tot[t-tmin]
			x_o = omi_count[t-tmin] / N_tot[t-tmin]
			if x_o > 1 - x_limit:
				tmaxnew = t
				break
		tmin = tminnew
		tmax = tmaxnew
		t_range= np.arange(tmin,tmax)

		vac_full = [meta_country.loc[t]['people_fully_vaccinated_per_hundred']/100. for t in meta_country.index]
		vinterp = clean_vac(list(meta_country.index),vac_full)
		country2immuno2time[country]['VAC'] = {list(meta_country.index)[i] : vinterp[i] for i in range(len(list(meta_country.index)))}

		booster = [meta_country.loc[t]['total_boosters_per_hundred']/100. for t in meta_country.index]
		if np.sum(np.isnan(booster)) == len(booster):
			boosterp = np.zeros(len(booster))
		else:
			boosterp = clean_vac(list(meta_country.index), booster)
		country2immuno2time[country]['BOOST'] = {list(meta_country.index)[i]:boosterp[i] for i in range(len(list(meta_country.index)))}
		#correct the cumulative vaccinated: NB: looks different for each time point
		for time_meas in  df_c.FLAI_time:
			for t in list(country2immuno2time[country]['VAC'].keys()):
				if country2immuno2time[country]['VAC'][t] < country2immuno2time[country]['BOOST'][time_meas]:
					country2immuno2time[country][time_meas][t] = 0.0
				else:
					country2immuno2time[country][time_meas][t] = country2immuno2time[country]['VAC'][t] - country2immuno2time[country]['BOOST'][time_meas]

		cases_full = [meta_country.loc[t]['new_cases']/meta_country.loc[t]['population'] for t in list(meta_country.index)]
		cases_full = clean_vac(list(meta_country.index),cases_full)
		country2immuno2time[country]['CASES'] = {list(meta_country.index)[i]:cases_full[i]*100000 for i in range(len(list(meta_country.index)))}

		x_delta = []
		x_omi = []
		for t in list(meta_country.index):
			if str(t) in freq_traj['DELTA'].keys():
				x_delta.append(freq_traj['DELTA'][str(t)])
			else:
				x_delta.append(0.0)
			if str(t) in freq_traj['OMICRON'].keys():
				x_omi.append(freq_traj['OMICRON'][str(t)])
			else:
				x_omi.append(0.0)
		freq_delta = np.array(x_delta)
		freq_omi = np.array(x_omi)

		cases_full_delta = cases_full * freq_delta
		recov_delta = [[np.sum(cases_full_delta[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['RECOV_DELTA_0'] = {a[1]: a[0] for a in recov_delta}

		cases_full_omi = cases_full * freq_omi
		recov_omi = [[np.sum(cases_full_omi[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['RECOV_OMI_0'] = {a[1]: a[0] for a in recov_omi}

for country in countries_da:
	with open("DATA/2022_06_22/freq_traj_" + country.upper() + ".json",'r') as f:
		freq_traj = json.load(f)
	with open("DATA/2022_06_22/multiplicities_" + country.upper() + ".json",'r') as f:
		counts = json.load(f)
	with open("DATA/2022_06_22/multiplicities_Z_" + country.upper() + ".json",'r') as f:
		Z = json.load(f)
	meta_country= meta_df.loc[list(meta_df['location'] == country[0] + country[1:].lower())]
	if len(country) <= 3:
		meta_country= meta_df.loc[list(meta_df['location'] == country)]
	meta_country.index = meta_country['FLAI_time']


	if max(list(freq_traj['ALPHA'].values())) > 0.5 and max(list(freq_traj['DELTA'].values())) > 0.5:
		dates_delta = list(counts['DELTA'].keys())
		dates_alpha = list(counts['ALPHA'].keys())
		dates_delta = [int(a) for a in dates_delta]
		dates_alpha = [int(a) for a in dates_alpha]
		dates_delta = sorted(dates_delta)
		dates_alpha = sorted(dates_alpha)
		dates_alpha = [a for a in dates_alpha if a < 44470]
		dates_delta = [a for a in dates_delta if a > 44255]
		if country=='ITALY':
			dates_delta = [a for a in dates_delta if a > 44287]
		if country=='CA':
			dates_delta = [a for a in dates_delta if a > 44274]

		tmin = min(set(dates_delta).intersection(set(dates_alpha)))
		tmax = max(set(dates_delta).intersection(set(dates_alpha)))
		t_range = np.arange(tmin,tmax)
		alpha_count = [int(np.exp(counts['ALPHA'][str(a)])) for a in t_range]
		delta_count = [int(np.exp(counts['DELTA'][str(a)])) for a in t_range]
		N_tot = np.array(alpha_count) + np.array(delta_count)
		check = 'not_okay'
		for t in t_range:
			x_d = delta_count[t-tmin] / N_tot[t-tmin]
			x_a = alpha_count[t-tmin] / N_tot[t-tmin]
			if x_d > x_limit:
				tminnew = t
				check = 'okay'
				break
		for t in t_range:
			x_d = delta_count[t-tmin] / N_tot[t-tmin]
			x_a = alpha_count[t-tmin] / N_tot[t-tmin]
			if x_d > 1 - x_limit:
				tmaxnew = t
				break
		tmin = tminnew
		tmax = tmaxnew
		t_range= np.arange(tmin,tmax)

		vac_full = [meta_country.loc[t]['people_fully_vaccinated_per_hundred']/100. for t in t_range]
		cases_full = [meta_country.loc[t]['new_cases']/meta_country.loc[t]['population'] for t in list(meta_country.index)]
		cases_full = clean_vac(list(meta_country.index),cases_full)
		country2immuno2time[country]['CASES'] = {list(meta_country.index)[i]:cases_full[i]*100000 for i in range(len(list(meta_country.index)))}
		x_alpha = []
		x_delta = []
		for t in list(meta_country.index):
			if str(t) in freq_traj['DELTA'].keys():
				x_delta.append(freq_traj['DELTA'][str(t)])
			else:
				x_delta.append(0.0)

			if str(t) in freq_traj['ALPHA'].keys():
				x_alpha.append(freq_traj['ALPHA'][str(t)])
			else:
				x_alpha.append(0.0)
		freq_delta = np.array(x_delta)
		freq_alpha = np.array(x_alpha)
		freq_wt = 1 - freq_delta - freq_alpha

		cases_full_alpha = cases_full * freq_alpha
		cases_full_wt = cases_full * freq_wt
		cases_full_delta = cases_full * freq_delta

		recov_delta = [[np.sum(cases_full_delta[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['RECOV_DELTA'] = {a[1]: a[0] for a in recov_delta}
		recov_alpha = [[np.sum(cases_full_alpha[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		recov_wt = [[np.sum(cases_full_wt[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['RECOV_ALPHA'] = {a[1]: a[0] for a in recov_alpha}
		country2immuno2time[country]['RECOV_WT'] = {a[1]: a[0] for a in recov_wt}

		vaccinated = [meta_country.loc[t]['people_fully_vaccinated_per_hundred']/100. for t in meta_country.index]
		vaccinated = clean_vac(list(meta_country.index),vaccinated)
	
		vaccinated = [meta_country.loc[t]['people_fully_vaccinated_per_hundred']/100. for t in list(meta_country.index)]
		vac_full = clean_vac(list(meta_country.index),vaccinated)
		vac_full = [[vac_full[i], list(meta_country.index)[i]] for i in range(len(vac_full)) if not np.isnan(vac_full[i])]
		country2immuno2time[country]['VAC'] = {a[1] : a[0] for a in vac_full}

dT_alpha_vac = 1.8
dT_delta_vac = 3.2
dT_omi_vac   = 47

dT_alpha_booster = 1.8 ##no value -> use the same as vaccinatino
dT_delta_booster = 2.8
dT_omi_booster   = 6.4

dT_alpha_alpha = 1.
dT_delta_alpha = 2.8
dT_omi_alpha   = 33

dT_alpha_delta = 3.5
dT_delta_delta = 1.
dT_omi_delta   = 27.

#Omicron values are taken as symmetric from their counterparts... no data!
dT_alpha_omi = 33.
dT_delta_omi = 27.
dT_omi_omi   = 1.

k=3.0
n50 = np.log10(0.2 * 94) 
time_decay = 90
tau_decay = time_decay
Xdat_od = []
for line in df_s_od.iterrows():
	line = line[1]
	Xdat_od.append([line.name,line.FLAI_time, line.s_hat,line.s_var])
Xdat_od = pd.DataFrame(Xdat_od,columns=['Country','time','s_hat','s_var'])

Xdat_da = []
for line in df_s_da.iterrows():
	line = line[1]
	Xdat_da.append([line.name,line.FLAI_time, line.s_hat,line.s_var])
Xdat_da = pd.DataFrame(Xdat_da,columns=['Country','time','s_hat','s_var'])

def sigmoid_func(t,mean,s):
	val = 1 / (1 + np.exp(-s * (t - mean)))
	return val

def integrate_S_component(country,time_50,tau_decay,country2immuno2time, immunisation,dTiter_1, dTiter_2):
	times = sorted(list(country2immuno2time[country][immunisation].keys()))
	if immunisation == 'VAC_cor':
		country2immuno2time[country][immunisation] = country2immuno2time[country][time_50]
		times = sorted(list(country2immuno2time[country][immunisation].keys()))
		T0 = 223
	elif immunisation == 'VAC':
		T0 = 223
	elif 'RECOV' in immunisation:
		T0 = 94
	elif immunisation == 'BOOST':
		T0 = 223 * 4
	
	time_index = 0
	while country2immuno2time[country][immunisation][times[time_index]] == 0.0:
		time_index += 1

	C_omi = 0.0
	C_delta = 0.0
	while times[time_index] < time_50:
		dt_vac = time_50 - times[time_index]
		weight = country2immuno2time[country][immunisation][times[time_index]] - country2immuno2time[country][immunisation][times[time_index-1]]
		C_1 = sigmoid_func(np.log10(T0) - np.log10(np.exp(dt_vac/tau_decay)) - np.log10(dTiter_1), n50, k)
		C_2 = sigmoid_func(np.log10(T0) - np.log10(np.exp(dt_vac/tau_decay)) - np.log10(dTiter_2), n50, k)
		C_omi += C_1 * weight
		C_delta += C_2 * weight
		time_index += 1

	return C_omi, C_delta

lines = []
for line in Xdat_da.iterrows():
	line = line[1]
	c=line.Country
	time_50 = int(line.time)

	C_delta_vac, C_alpha_vac =   integrate_S_component(c,time_50, tau_decay,country2immuno2time, 'VAC', dT_delta_vac, dT_alpha_vac)
	C_delta_alpha, C_alpha_alpha = integrate_S_component(c,time_50, tau_decay,country2immuno2time, 'RECOV_ALPHA', dT_delta_alpha, 1.0)
	C_delta_delta, C_alpha_delta = integrate_S_component(c,time_50, tau_decay,country2immuno2time, 'RECOV_DELTA', dT_delta_delta, dT_alpha_delta)

	#Get effective weights:
	C_delta_vac_0 = sigmoid_func(np.log10(223) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_delta_vac),n50,k)
	C_alpha_vac_0 = sigmoid_func(np.log10(223) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_alpha_vac),n50,k)
	C_delta_alpha_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_delta_alpha),n50,k)
	C_alpha_alpha_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(1.0),n50,k)
	C_delta_delta_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_delta_delta),n50,k)
	C_alpha_delta_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_alpha_delta),n50,k)

	dC_vac = C_alpha_vac_0 - C_delta_vac_0
	dC_alpha = C_alpha_alpha_0 - C_delta_alpha_0
	dC_delta = C_alpha_delta_0 - C_delta_delta_0

	x_eff_vac = (C_alpha_vac - C_delta_vac) / dC_vac
	x_eff_alpha = (C_alpha_alpha - C_delta_alpha) / dC_alpha
	x_eff_delta = (C_alpha_delta - C_delta_delta) / dC_delta	
	t_range = np.arange(time_50 - 20, time_50)
	cases = np.mean([country2immuno2time[c]['CASES'][t] for t in t_range])

	lines.append([c,time_50, line.s_hat, line.s_var, x_eff_vac, x_eff_alpha, x_eff_delta, cases,dC_vac,dC_alpha,dC_delta])
	
lines = pd.DataFrame(lines,columns=['Country','time','s_hat','s_var','x_eff_vac', 'x_eff_alpha', 'x_eff_delta', 'cases','dC_vac','dC_alpha','dC_delta'])
lines.to_csv("output/Data_da.txt",'\t',index=False)


c2_max_omi = defaultdict(lambda: [])
lines = []
for line in Xdat_od.iterrows():
	line = line[1]
	c=line.Country
	time_50 = int(line.time)

	C_omi_vac, C_delta_vac =   integrate_S_component(c,time_50, tau_decay,country2immuno2time, 'VAC_cor', dT_omi_vac, dT_delta_vac)
	C_omi_delta, C_delta_delta = integrate_S_component(c,time_50, tau_decay,country2immuno2time, 'RECOV_DELTA_0', dT_omi_delta, 1.0)
	C_omi_omi, C_delta_omi = integrate_S_component(c,time_50, tau_decay,country2immuno2time, 'RECOV_OMI_0', 1.0, dT_delta_omi)
	if np.sum(list(country2immuno2time[c]['BOOST'].values())) != 0.0:
		C_omi_booster, C_delta_booster = integrate_S_component(c,time_50, tau_decay,country2immuno2time, 'BOOST', dT_omi_booster, dT_delta_booster)
	else:
		C_omi_booster = 0.0
		C_delta_booster = 0.0
	c2_max_omi[c].append(country2immuno2time[c]['RECOV_OMI_0'][time_50])
	
	#Get effective weights:
	#boost Large part of the change is also decay in selection in vaccinated individuals!
	C_omi_vac0, C_delta_vac0 =   integrate_S_component(c,time_50, tau_decay,country2immuno2time, 'VAC', dT_omi_vac, dT_delta_vac)
	C_delta_vac_0 = sigmoid_func(np.log10(223) - np.log10(np.exp(210/tau_decay)) - np.log10(dT_delta_vac),n50,k)
	C_omi_vac_0 = sigmoid_func(np.log10(223) - np.log10(np.exp(210/tau_decay)) - np.log10(dT_omi_vac),n50,k)
	C_delta_vac_1 = sigmoid_func(np.log10(223) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_delta_vac),n50,k)
	C_omi_vac_1 = sigmoid_func(np.log10(223) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_omi_vac),n50,k)
	C_delta_boost_0 = sigmoid_func(np.log10(223*4) - np.log10(np.exp(30/tau_decay)) - np.log10(dT_delta_booster),n50,k)
	C_omi_boost_0 = sigmoid_func(np.log10(223*4) - np.log10(np.exp(30/tau_decay)) - np.log10(dT_omi_booster),n50,k)
	dC_boost =  C_delta_boost_0 - C_omi_boost_0# - (C_delta_vac_0 - C_omi_vac_0)	

	C_delta_delta_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(1.0),n50,k)
	C_omi_delta_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_omi_delta),n50,k)
	dC_delta = C_delta_delta_0 - C_omi_delta_0

	C_delta_omi_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_delta_omi),n50,k)
	C_omi_omi_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(1.0),n50,k)
	dC_omi = C_delta_omi_0 - C_omi_omi_0
	dC_vac = (C_delta_vac_1 - C_omi_vac_1)

	x_eff_vac = (C_delta_vac - C_omi_vac) / dC_vac
	x_eff_vac0 = (C_delta_vac0 - C_omi_vac0) / dC_vac
	x_eff_boost = (C_delta_booster - C_omi_booster) / dC_boost
	x_eff_delta = (C_delta_delta - C_omi_delta) / dC_delta	
	x_eff_omi = (C_delta_omi - C_omi_omi) / dC_omi	

	t_range = np.arange(time_50 - 15, time_50)
	cases = np.mean([country2immuno2time[c]['CASES'][t] for t in t_range])


	lines.append([c,time_50,line.s_hat,line.s_var,x_eff_vac,x_eff_vac0,x_eff_boost, x_eff_delta, x_eff_omi, cases,dC_vac,dC_boost,dC_delta,dC_omi])
lines = pd.DataFrame(lines,columns=['Country','time','s_hat','s_var','x_eff_vac','x_eff_vac0','x_eff_boost', 'x_eff_delta', 'x_eff_omi', 'cases','dC_vac','dC_boost','dC_delta','dC_omi'])
country2pop = []
for c in c2_max_omi.keys():
	max_o = max(c2_max_omi[c])
	if max_o < 0.01:
		country2pop.append(c)
mask = [c not in country2pop for c in list(lines.Country)]
lines = lines.loc[mask]
lines.to_csv("output/Data_od.txt",'\t',index=False)


