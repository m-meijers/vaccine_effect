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
country2trange = defaultdict(lambda: defaultdict())
country2time_50 = defaultdict(lambda: defaultdict())
df_s_od = pd.read_csv("output/df_od_result.txt",sep='\t',index_col='Country')
df_s_ad = pd.read_csv("output/df_da_result.txt",sep='\t',index_col='Country')
countries_da = sorted(list(set(df_s_ad.index)))
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
		t_50 = 0
		for t in t_range:
			x_o = omi_count[t-tmin] / N_tot[t-tmin]
			if x_o > 0.5:
				t_50 = t
				break
		tmin = tminnew
		tmax = tmaxnew
		t_range= np.arange(tmin,tmax)
		country2time_50['OD'][country] = t_50

		country2trange['OD'][country] = t_range

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
		for time_meas in  t_range:
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
		for t in t_range:
			x_d = delta_count[t-tmin] / N_tot[t-tmin]
			x_a = alpha_count[t-tmin] / N_tot[t-tmin]
			if x_d > x_a:
				t_50 = t
				break
		tmin = tminnew
		tmax = tmaxnew
		t_range= np.arange(tmin,tmax)
		country2time_50['DA'][country] = t_50

		country2trange['DA'][country] = t_range

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



VOCs = ['ALPHA','BETA','GAMMA','DELTA','EPSILON','KAPPA','LAMBDA']
df_s_awt = pd.read_csv("output/s_hat_alpha_wt.txt",'\t',index_col=False)
countries_awt = sorted(list(set(df_s_awt.country)))

Population_immunity_AWT = []
for country in countries_awt:
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


	if max(list(freq_traj['ALPHA'].values())) > 0.5:
		dates_alpha = list(counts['ALPHA'].keys())
		dates_alpha = [int(a) for a in dates_alpha]
		dates_alpha = sorted(dates_alpha)
		dates_alpha = [a for a in dates_alpha if a < 44470]
		tmin = min(dates_alpha)
		tmax = max(dates_alpha)
		t_range = np.arange(tmin,tmax)
		alpha_count = [int(np.exp(counts['ALPHA'][str(a)])) for a in t_range]
		wt_count = [int(np.exp(Z[str(a)])) - int(np.exp(counts['ALPHA'][str(a)])) for a in t_range]
		N_tot = np.array(alpha_count) + np.array(wt_count)
		alpha_freq = np.array(alpha_count) / np.array(N_tot)
		check = 'not_okay'
		for t in t_range:
			x_wt = wt_count[t-tmin] / N_tot[t-tmin]
			x_a = alpha_count[t-tmin] / N_tot[t-tmin]
			if x_a > x_limit:
				tminnew = t
				check = 'okay'
				break
		for t in t_range:
			x_wt = wt_count[t-tmin] / N_tot[t-tmin]
			x_a = alpha_count[t-tmin] / N_tot[t-tmin]
			if x_a > max(alpha_freq)-0.01:
				tmaxnew = t
				break
		tmin = tminnew
		tmax = tmaxnew
		t_range= np.arange(tmin,tmax)

		country2time_50['AWT'][country] = t_50
		country2trange['AWT'][country] = t_range

		x_alpha = []
		x_wt = []
		freq_wt_dict = defaultdict(lambda: 0.0)
		for t in list(meta_country.index):
			if str(t) in freq_traj['ALPHA'].keys():
				x_alpha.append(freq_traj['ALPHA'][str(t)])
			else:
				x_alpha.append(0.0)
			x_voc = 0.0
			for voc in VOCs:
				if voc in freq_traj.keys():
					if str(t) in freq_traj[voc].keys():
						x_voc += freq_traj[voc][str(t)]
			x_wt.append(1-x_voc)
			freq_wt_dict[str(t)] = 1 - x_voc

		freq_alpha = np.array(x_alpha)
		freq_wt = np.array(x_wt)

		cases_full = [meta_country.loc[t]['new_cases']/meta_country.loc[t]['population'] for t in list(meta_country.index)]
		cases_full = clean_vac(list(meta_country.index),cases_full)
		cases_full_alpha = cases_full * freq_alpha
		recov_alpha = [[np.sum(cases_full_alpha[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['RECOV_ALPHA'] = {a[1]: a[0] for a in recov_alpha}
		cases_full_wt = cases_full * freq_wt
		recov_wt = [[np.sum(cases_full_wt[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['RECOV_WT'] = {a[1]: a[0] for a in recov_wt}




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

dT_alpha_omi = 33.
dT_delta_omi = 27.
dT_omi_omi   = 1.

k=3.0 
n50 = np.log10(0.2 * 94) 
time_decay = 90
tau_decay = time_decay

file = open("output/result_da.txt",'r')
coefs_da = file.readline().split()
file.close()
coefs_da = [float(c) for c in coefs_da]

file = open("output/result_od.txt",'r')
coefs_od = file.readline().split()
file.close()
coefs_od = [float(c) for c in coefs_od]


df_da_result = pd.read_csv("output/df_da_result.txt",'\t',index_col=False)
df_od_result = pd.read_csv("output/df_od_result.txt",'\t',index_col=False)

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

def approx_immune_weight(country,time_50,tau_decay,country2immuno2time, immunisation):
	times = sorted(list(country2immuno2time[country][immunisation].keys()))
	time_index = 0
	while country2immuno2time[country][immunisation][times[time_index]] == 0.0:
		time_index += 1

	R_k = 0.0
	while times[time_index] < time_50:
		dt_vac = time_50 - times[time_index]
		weight = country2immuno2time[country][immunisation][times[time_index]] - country2immuno2time[country][immunisation][times[time_index-1]]
		R_k += np.exp(- dt_vac / tau_decay) * weight
		time_index += 1

	return R_k



country2fit_error = defaultdict(lambda:defaultdict(lambda: 0.0))

Population_immunity_DA = []
for c in countries_da:
	t_range = country2trange['DA'][c]
	with open("DATA/2022_06_22/multiplicities_" + c.upper() + ".json",'r') as f:
		counts = json.load(f)
	with open("DATA/2022_06_22/multiplicities_Z_" + c.upper() + ".json",'r') as f:
		Z = json.load(f)
	with open("DATA/2022_06_22/freq_traj_" + c.upper() + ".json",'r') as f:
		freq_traj = json.load(f)
	alpha_count = [int(np.exp(counts['ALPHA'][str(a)])) for a in t_range]
	delta_count = [int(np.exp(counts['DELTA'][str(a)])) for a in t_range]
	Z_count = [int(np.exp(Z[str(a)])) for a in t_range]
	N_tot = np.array(alpha_count) + np.array(delta_count)
	for t in t_range:
		ac = alpha_count[t-t_range[0]]
		dc = delta_count[t-t_range[0]]
		other_c = N_tot[t-t_range[0]] - dc - ac
		totcount = Z_count[t-t_range[0]]
		x_d = dc / N_tot[t-t_range[0]]
		x_a = ac / N_tot[t-t_range[0]]
		x_other = other_c/N_tot[t-t_range[0]]
		freq_d = freq_traj['DELTA'][str(t)]
		freq_a = freq_traj['ALPHA'][str(t)]
		freq_other = 1 - freq_d - freq_a
		var_d = 1/totcount**2 * dc * (1 - dc/totcount)
		var_a = 1/totcount**2 * ac * (1 - ac/totcount)
		var_other = 1/totcount**2 * other_c * (1 - other_c/totcount)




	
		C_delta_vac, C_alpha_vac =   integrate_S_component(c,t, tau_decay,country2immuno2time, 'VAC', dT_delta_vac, dT_alpha_vac)
		C_delta_alpha, C_alpha_alpha = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_ALPHA', dT_delta_alpha, 1.0)
		C_delta_delta, C_alpha_delta = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_DELTA', dT_delta_delta, dT_alpha_delta)

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

		R_eff_vac = (C_alpha_vac - C_delta_vac) / dC_vac
		R_eff_alpha = (C_alpha_alpha - C_delta_alpha) / dC_alpha
		R_eff_delta = (C_alpha_delta - C_delta_delta) / dC_delta	
		# R_vac = approx_immune_weight(c,t,tau_decay,country2immuno2time,'VAC')
		# R_alpha = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_ALPHA')
		# R_delta = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_DELTA')

		R_vac = country2immuno2time[c]['VAC'][t]
		R_alpha = country2immuno2time[c]['RECOV_ALPHA'][t]
		R_delta = country2immuno2time[c]['RECOV_DELTA'][t]


		Population_immunity_DA.append([c, t,freq_a, freq_d,freq_other, x_a, x_d, var_a, var_d,var_other, R_eff_vac, R_eff_alpha, R_eff_delta, R_vac, R_alpha, R_delta, C_delta_vac, C_alpha_vac, C_delta_alpha, C_alpha_alpha, C_delta_delta, C_alpha_delta])
Population_immunity_DA = pd.DataFrame(Population_immunity_DA, columns=['country','time','freq_alpha','freq_delta','freq_other','x_alpha','x_delta','var_a','var_d','var_other','R_eff_vac','R_eff_alpha','R_eff_delta','R_vac','R_alpha','R_delta','Cdv','Cav','Cda','Caa','Cdd','Cad'])

model_x_delta = []
model_x_alpha = []
for c in countries_da:
	t_range = country2trange['DA'][c]
	t_50 = country2time_50['DA'][c]
	pop_c = Population_immunity_DA.loc[list(Population_immunity_DA.country == c)]
	s0 = df_da_result.loc[list(df_da_result.Country == c)].iloc[0].s_0
	s_model = coefs_da[0] * np.array(pop_c.R_eff_vac) + coefs_da[1] * np.array(pop_c.R_eff_alpha) + coefs_da[2] * np.array(pop_c.R_eff_delta) + s0
	ratio = np.log(np.array(pop_c.x_delta)) - np.log(np.array(pop_c.x_alpha))
	#find optimum starting
	def infer_model(t_50):
		index_start = list(t_range).index(t_50)
		DATA = list(pop_c.x_delta)
		model_list_0 = [DATA[index_start]]
		t_list = np.arange(t_50+1, t_range[-1]+1)
		for i in range(len(t_list)):
			model_list_0.append(model_list_0[-1] + s_model[i+index_start] * model_list_0[-1] * (1 - model_list_0[-1]))
		t_list = np.arange(t_50,t_range[0],-1)
		model_list_1 = [DATA[index_start]]
		for i in range(len(t_list)):
			model_list_1.append(model_list_1[-1] - s_model[index_start - i]* model_list_1[-1] * (1 - model_list_1[-1]))
		model_list_1 = model_list_1[-1:0:-1]
		m_x_delta = model_list_1 + model_list_0
		m_x_delta = np.array(m_x_delta)
		m_x_alpha = 1 - np.array(m_x_delta)
		return m_x_delta, m_x_alpha
	mxd, mxa = infer_model(t_50)
	E = np.mean([e**2 for e in np.log(mxd/mxa) - ratio])
	mxd_plus, mxa_plus = infer_model(t_50+1)
	mxd_min, mxa_min = infer_model(t_50-1)
	Eplus = np.mean([e**2 for e in np.log(mxd_plus/mxa_plus) - ratio])
	Emin = np.mean([e**2 for e in np.log(mxd_min/mxa_min) - ratio])
	if Eplus < E:
		Enew = Eplus
		step = +1
		while Enew < E:
			E = Enew
			step += 1
			mxd_plus, mxa_plus = infer_model(t_50+step)
			Enew = np.mean([e**2 for e in np.log(mxd_plus/mxa_plus) - ratio])
		step -= 1
	elif Emin < E:
		Enew = Emin 
		step = -1
		while Enew < E:
			E = Enew
			step -= 1
			mxd_min, mxa_min = infer_model(t_50+step)
			Enew = np.mean([e**2 for e in np.log(mxd_min/mxa_min) - ratio])
		step += 1
	m_x_delta, m_x_alpha = infer_model(t_50+step)
	model_x_delta += list(m_x_delta)
	model_x_alpha += list(m_x_alpha)
	country2fit_error['DA'][c] = E

Population_immunity_DA['model_x_alpha'] = model_x_alpha
Population_immunity_DA['model_x_delta'] = model_x_delta

Population_immunity_OD = []
for c in countries_od:
	t_range = country2trange['OD'][c]
	with open("DATA/2022_06_22/multiplicities_" + c.upper() + ".json",'r') as f:
		counts = json.load(f)
	with open("DATA/2022_06_22/freq_traj_" + c.upper() + ".json",'r') as f:
		freq_traj = json.load(f)
	with open("DATA/2022_06_22/multiplicities_Z_" + c.upper() + ".json",'r') as f:
		Z = json.load(f)
	delta_count = [int(np.exp(counts['DELTA'][str(a)])) for a in t_range]
	omi_count = [int(np.exp(counts['OMICRON'][str(a)])) for a in t_range]
	Z_count = [int(np.exp(Z[str(a)])) for a in t_range]
	N_tot = np.array(delta_count) + np.array(omi_count)
	for t in t_range:
		dc = delta_count[t-t_range[0]]
		oc = omi_count[t-t_range[0]]
		totcount = Z_count[t-t_range[0]]
		x_d = dc / N_tot[t-t_range[0]]
		x_o = oc / N_tot[t-t_range[0]]
		freq_d = freq_traj['DELTA'][str(t)]
		freq_o = freq_traj['OMICRON'][str(t)]
		var_d = 1/totcount**2 * dc * (1 - dc/totcount)
		var_o = 1/totcount**2 * oc * (1 - oc/totcount)


		C_omi_vac, C_delta_vac =   integrate_S_component(c,t, tau_decay,country2immuno2time, 'VAC_cor', dT_omi_vac, dT_delta_vac)
		if np.sum(list(country2immuno2time[c]['BOOST'].values())) != 0.0:
			C_omi_booster, C_delta_booster = integrate_S_component(c,t, tau_decay,country2immuno2time, 'BOOST', dT_omi_booster, dT_delta_booster)
		else:
			C_omi_booster = 0.0
			C_delta_booster = 0.0
		C_omi_delta, C_delta_delta = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_DELTA_0', dT_omi_delta, 1.0)
		C_omi_omi, C_delta_omi = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_OMI_0', 1.0, dT_delta_omi)

		C_delta_vac_1 = sigmoid_func(np.log10(223) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_delta_vac),n50,k)
		C_omi_vac_1 = sigmoid_func(np.log10(223) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_omi_vac),n50,k)
		C_delta_delta_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(1.0),n50,k)
		C_omi_delta_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_omi_delta),n50,k)
		C_delta_omi_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(dT_delta_omi),n50,k)
		C_omi_omi_0 = sigmoid_func(np.log10(94) - np.log10(np.exp(62/tau_decay)) - np.log10(1.0),n50,k)
		C_omi_boost_0 = sigmoid_func(np.log10(223*4) - np.log10(np.exp(30/tau_decay)) - np.log10(dT_omi_booster),n50,k)
		C_delta_boost_0 = sigmoid_func(np.log10(223*4) - np.log10(np.exp(30/tau_decay)) - np.log10(dT_delta_booster),n50,k)


		dC_delta = C_delta_delta_0 - C_omi_delta_0
		dC_omi = C_delta_omi_0 - C_omi_omi_0
		dC_vac = (C_delta_vac_1 - C_omi_vac_1)
		dC_boost =  C_delta_boost_0 - C_omi_boost_0

		R_eff_vac = (C_delta_vac - C_omi_vac) / dC_vac
		R_eff_boost = (C_delta_booster - C_omi_booster) / dC_boost
		R_eff_delta = (C_delta_delta - C_omi_delta) / dC_delta	
		R_eff_omi = (C_delta_omi - C_omi_omi) / dC_omi	

		# R_vac = approx_immune_weight(c,t,tau_decay,country2immuno2time,'VAC_cor')
		# R_boost = approx_immune_weight(c,t,tau_decay,country2immuno2time,'BOOST')
		# R_delta = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_DELTA_0')
		# R_omi = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_OMI_0')

		R_vac = country2immuno2time[c]['VAC'][t]
		R_boost = country2immuno2time[c]['BOOST'][t]
		R_delta = country2immuno2time[c]['RECOV_DELTA_0'][t]
		R_omi = country2immuno2time[c]['RECOV_OMI_0'][t]

		Population_immunity_OD.append([c, t,freq_d,freq_o, x_d, x_o,var_d,var_o, R_eff_vac, R_eff_boost, R_eff_delta, R_eff_omi, R_vac, R_boost, R_delta, R_omi, C_omi_vac, C_delta_vac,C_omi_booster,C_delta_booster, C_omi_delta,C_delta_delta, C_omi_omi,C_delta_omi])
Population_immunity_OD = pd.DataFrame(Population_immunity_OD, columns=['country','time','freq_delta','freq_omi','x_delta','x_omi','var_d','var_o','R_eff_vac','R_eff_boost','R_eff_delta','R_eff_omi','R_vac','R_boost','R_delta','R_omi','Cov','Cdv','Cob','Cdb','Cod','Cdd','Coo','Cdo'])

model_x_delta3 = []
model_x_omi = []
for c in countries_od:
	pop_c = Population_immunity_OD.loc[list(Population_immunity_OD.country == c)]
	t_range = country2trange['OD'][c]
	s0 = df_od_result.loc[list(df_od_result.Country == 'ITALY')].iloc[0].s_0
	s_model = coefs_od[0] * np.array(pop_c.R_eff_vac) + coefs_od[1] * np.array(pop_c.R_eff_boost) + coefs_od[2] * np.array(pop_c.R_eff_delta) + coefs_od[3] * np.array(pop_c.R_eff_omi) + s0
	t_50 = country2time_50['OD'][c]
	ratio = np.log(np.array(pop_c.x_omi)) - np.log(np.array(pop_c.x_delta))
	#find optimum starting
	def infer_model(t_50):
		index_start = list(t_range).index(t_50)
		DATA = list(pop_c.x_omi)
		model_list_0 = [DATA[index_start]]
		t_list = np.arange(t_50+1, t_range[-1]+1)
		for i in range(len(t_list)):
			model_list_0.append(model_list_0[-1] + s_model[i+index_start] * model_list_0[-1] * (1 - model_list_0[-1]))
		t_list = np.arange(t_50,t_range[0],-1)
		model_list_1 = [DATA[index_start]]
		for i in range(len(t_list)):
			model_list_1.append(model_list_1[-1] - s_model[index_start - i]* model_list_1[-1] * (1 - model_list_1[-1]))
		model_list_1 = model_list_1[-1:0:-1]
		m_x_omi = model_list_1 + model_list_0
		m_x_omi = np.array(m_x_omi)
		m_x_delta = 1 - np.array(m_x_omi)
		return m_x_omi, m_x_delta
	mxo, mxd = infer_model(t_50)
	E = np.mean([e**2 for e in np.log(mxo/mxd) - ratio])
	mxo_plus, mxd_plus = infer_model(t_50+1)
	mxo_min, mxd_min = infer_model(t_50-1)
	Eplus = np.mean([e**2 for e in np.log(mxo_plus/mxd_plus) - ratio])
	Emin = np.mean([e**2 for e in np.log(mxo_min/mxd_min) - ratio])
	if Eplus < E:
		Enew = Eplus
		step = +1
		while Enew < E:
			E = Enew
			step += 1
			mxo_plus, mxd_plus = infer_model(t_50+step)
			Enew = np.mean([e**2 for e in np.log(mxo_plus/mxd_plus) - ratio])
		step -= 1
	elif Emin < E:
		Enew = Emin 
		step = -1
		while Enew < E:
			E = Enew
			step -= 1
			mxo_min, mxd_min = infer_model(t_50+step)
			Enew = np.mean([e**2 for e in np.log(mxo_min/mxd_min) - ratio])
		step += 1
	m_x_omi, m_x_delta = infer_model(t_50+step)
	model_x_delta3 += list(m_x_delta)
	model_x_omi += list(m_x_omi)
	country2fit_error['OD'][c] = E

Population_immunity_OD['model_x_delta'] = model_x_delta3
Population_immunity_OD['model_x_omi'] = model_x_omi

df_s_awt = pd.read_csv("output/s_hat_alpha_wt.txt",'\t',index_col=False)
countries_awt = sorted(list(set(df_s_awt.country)))

Population_immunity_DA.to_csv("output/Pop_C_DA.txt",'\t',index=False)
Population_immunity_OD.to_csv("output/Pop_C_OD.txt",'\t',index=False)

