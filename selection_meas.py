import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flai.util.Time import Time
import json
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from collections import defaultdict
import glob


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

VOCs = ['ALPHA','BETA','GAMMA','DELTA','EPSILON','KAPPA','LAMBDA']
colors = ['b','g','r','c','m','y','k','lime','salmon','lime']

x_limit = 0.01
dt = 40.0
min_count = 500

country_min_count = defaultdict(lambda: [])
print("=====================Alpha - Delta ============================")
files = glob.glob("DATA/2022_06_22/freq_traj_*")
countries = [f.split("_")[-1][:-5] for f in files]
meta_df = pd.read_csv("DATA/clean_data.txt",sep='\t',index_col=False)
country2max_recov =   defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: defaultdict())))

country2immuno2time = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
lines = []
for country in countries:
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
		if country=='ISRAEL':
			dates_delta = [a for a in dates_delta if a > 44262]
		if country=='PORTUGAL':
			dates_delta = [a for a in dates_delta if a > 44282]
		if country=='TURKEY':
			dates_delta = [a for a in dates_delta if a > 44306]
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

		Ztot =np.array([int(np.exp(Z[str(t)])) for t in t_range])
		country_min_count[country].append(min(Ztot))
		if np.sum(Ztot > min_count) != len(Ztot):
			print("Low sequence count", country)
			continue

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

		cases_full = [meta_country.loc[t]['new_cases']/meta_country.loc[t]['population'] for t in list(meta_country.index)]
		cases_full = clean_vac(list(meta_country.index),cases_full)

		cases_full_alpha = cases_full * freq_alpha
		cases_full_wt = cases_full * freq_wt
		cases_full_delta = cases_full * freq_delta

		recov_tot = [[np.sum(cases_full_alpha[t_range[0]-list(meta_country.index)[0]:t-list(meta_country.index)[0]]),np.sum(cases_full_delta[t_range[0]-list(meta_country.index)[0]:t-list(meta_country.index)[0]]), t] for t in t_range]
		recov_df  = pd.DataFrame(recov_tot,columns=['recov_alpha','recov_delta','t'])
		country2max_recov['AD']['ALPHA'][country] = max(recov_df.recov_alpha)
		country2max_recov['AD']['DELTA'][country] = max(recov_df.recov_delta)

		

		recov_delta = [[np.sum(cases_full_delta[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['RECOV_DELTA'] = {a[1]: a[0] for a in recov_delta}
		recov_alpha = [[np.sum(cases_full_alpha[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		recov_wt = [[np.sum(cases_full_wt[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['RECOV_ALPHA'] = {a[1]: a[0] for a in recov_alpha}
		country2immuno2time[country]['RECOV_WT'] = {a[1]: a[0] for a in recov_wt}
		recovered  = [meta_country.loc[t]['total_cases'] / meta_country.loc[t]['population'] for t in t_range]
		cases = [meta_country.loc[t]['new_cases'] / meta_country.loc[t]['population'] * 1000000 for t in t_range]
		vaccinated = [meta_country.loc[t]['people_fully_vaccinated_per_hundred']/100. for t in t_range]

		vaccinated = clean_vac(t_range,vaccinated)
		country2immuno2time[country]['VAC'] = {t_range[i] : vaccinated[i] for i in range(len(t_range))}
		
		alpha_count = [int(np.exp(counts['ALPHA'][str(a)])) for a in t_range]
		delta_count = [int(np.exp(counts['DELTA'][str(a)])) for a in t_range]


		t1 = 0 
		t2 = int(t1 + dt)
		while t2 + tmin < tmax:
			FLAI_time = int((t1 + t2 + 2*tmin)/2)
			N_tot1 = alpha_count[t1] + delta_count[t1]
			N_tot2 = alpha_count[t2] + delta_count[t2]
			if alpha_count[t1] < 10 or alpha_count[t2] < 10 or delta_count[t1] < 10 or delta_count[t2] < 10:
				t1 = t1 + 7
				t2 = int(t1 + dt)
				print("t<10",country)
				continue
			p_t1 = [alpha_count[t1] / N_tot1, delta_count[t1] / N_tot1]
			p_t2 = [alpha_count[t2] / N_tot2, delta_count[t2] / N_tot2]

			x_alpha_hat_t1 = np.random.binomial(N_tot1,p_t1[0],1000) / N_tot1
			x_delta_hat_t1 = np.ones(len(x_alpha_hat_t1)) - x_alpha_hat_t1
			x_alpha_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_alpha_hat_t1])
			x_delta_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_delta_hat_t1])

			x_alpha_hat_t2 = np.random.binomial(N_tot2,p_t2[0],1000) / N_tot2
			x_delta_hat_t2 = np.ones(len(x_alpha_hat_t2)) - x_alpha_hat_t2
			x_alpha_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_alpha_hat_t2])
			x_delta_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_delta_hat_t2])

			result = (np.log(x_delta_hat_t2 / x_alpha_hat_t2) - np.log(x_delta_hat_t1 / x_alpha_hat_t1)) / dt
			s_hat = np.mean(result)
			s_hat_var = np.var(result)

			vac_av = np.mean(vaccinated[t1:t2])
			recov = np.mean(recovered[t1:t2])
			cases_av = np.mean(cases[t1:t2])
			t_range_here = np.arange(t1 + tmin,t2 + tmin)
			delta_recov = np.mean([country2immuno2time[country]['RECOV_DELTA'][int(t)] for t in t_range_here])
			alpha_recov = np.mean([country2immuno2time[country]['RECOV_ALPHA'][int(t)] for t in t_range_here])
			t_range_here = np.arange(t1 + tmin,t2 + tmin)
			
			
			if str(FLAI_time) in freq_traj['DELTA'].keys():
				x_delta = freq_traj['DELTA'][str(FLAI_time)]
			else:
				x_delta = 0.0

			if str(FLAI_time) in freq_traj['ALPHA'].keys():
				x_alpha = freq_traj['ALPHA'][str(FLAI_time)]
			else:
				x_alpha = 0.0
			freq_wt = 1 - x_delta - x_alpha


			lines.append([country,FLAI_time, Time.coordinateToStringDate(int(t1+tmin)),Time.coordinateToStringDate(int(t2+tmin)), np.round(s_hat,3), 
			np.round(s_hat_var,7), np.round(vac_av,3), np.round(recov,3),delta_recov,alpha_recov, alpha_count[t1], alpha_count[t2],delta_count[t1],delta_count[t2],np.round(cases_av,2),freq_wt, x_delta, x_alpha])

			t1 = t1 + 7
			t2 = int(t1 + dt)
	else:
		print("no 50%: ", country)
		country_min_count[country].append(0)

lines = pd.DataFrame(lines,columns = ['country','FLAI_time','t1','t2','s_hat','s_var','vaccinated','recovered','delta_recov','alpha_recov','alpha_count_t1','alpha_count_t2','delta_count_t1','delta_count_t2','av_cases','x_wt','x_delta','x_alpha'])
country2pop = []
for country in list(set(lines.country)):
	df_c = lines.loc[list(lines.country == country)]
	if len(df_c) < 6:
		country2pop.append(country)
		
		
mask = [c not in country2pop for c in list(lines.country)]
lines = lines.loc[mask]
savename = 'output/s_hat_delta_alpha.txt'
lines.to_csv(savename,'\t',index=False)

print("=====================Delta - Omicron ============================")
# country2max_recov = defaultdict(lambda:defaultdict())
dt = 30.0
min_count = 750
country2immuno2time = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
lines = []
for country in countries:
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
		tmin = tminnew
		tmax = tmaxnew
		t_range= np.arange(tmin,tmax)

		Ztot =np.array([int(np.exp(Z[str(t)])) for t in t_range])
		country_min_count[country].append(min(Ztot))
		if np.sum(Ztot > min_count) != len(Ztot):
			print("Low sequence count",country)
			continue

		vac_full = [meta_country.loc[t]['people_fully_vaccinated_per_hundred']/100. for t in meta_country.index]
		vinterp = clean_vac(list(meta_country.index),vac_full)
		country2immuno2time[country]['VAC'] = {list(meta_country.index)[i] : vinterp[i] for i in range(len(list(meta_country.index)))}

		booster = [meta_country.loc[t]['total_boosters_per_hundred']/100. for t in meta_country.index]
		if np.sum(np.isnan(booster)) == len(booster):
			boosterp = np.zeros(len(booster))
		else:
			boosterp = clean_vac(list(meta_country.index), booster)
		country2immuno2time[country]['BOOST'] = {list(meta_country.index)[i]:boosterp[i] for i in range(len(list(meta_country.index)))}
		cases_full = [meta_country.loc[t]['new_cases']/meta_country.loc[t]['population'] for t in list(meta_country.index)]
		cases_full = clean_vac(list(meta_country.index),cases_full)
		country2immuno2time[country]['CASES'] = {list(meta_country.index)[i]:cases_full[i] for i in range(len(list(meta_country.index)))}



		x_delta = []
		x_omi = []
		for t in list(meta_country.index):
			if str(t) in freq_traj['DELTA'].keys():
				x_delta.append(freq_traj['DELTA'][str(t)])
			else:
				x_delta.append(0.0)
			if str(t) in freq_traj['OMICRON'].keys() and t > Time.dateToCoordinate("2021-10-01"):
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

		recov_tot = [[np.sum(cases_full_delta[t_range[0]-list(meta_country.index)[0]:t-list(meta_country.index)[0]]),np.sum(cases_full_omi[t_range[0]-list(meta_country.index)[0]:t-list(meta_country.index)[0]]), t] for t in t_range]
		recov_df  = pd.DataFrame(recov_tot,columns=['recov_delta','recov_omi','t'])
		country2max_recov['DO']['DELTA'][country] = max(recov_df.recov_delta)
		country2max_recov['DO']['OMI'][country] = max(recov_df.recov_omi)

		omi_count = [int(np.exp(counts['OMICRON'][str(a)])) for a in t_range]
		delta_count = [int(np.exp(counts['DELTA'][str(a)])) for a in t_range]
		
		t1 = 0 
		t2 = int(t1 + dt)
		while t2 + tmin < tmax:
			FLAI_time = int((t1 + t2 + 2*tmin)/2)
			N_tot1 = omi_count[t1] + delta_count[t1]
			N_tot2 = omi_count[t2] + delta_count[t2]
			if omi_count[t1] < 10 or omi_count[t2] < 10 or delta_count[t1] < 10 or delta_count[t2] < 10:
				t1 = t1 + 7
				t2 = int(t1 + dt)
				print("t<10",country)
				continue
			p_t1 = [omi_count[t1] / N_tot1, delta_count[t1] / N_tot1]
			p_t2 = [omi_count[t2] / N_tot2, delta_count[t2] / N_tot2]

			x_omi_hat_t1 = np.random.binomial(N_tot1,p_t1[0],1000) / N_tot1
			x_delta_hat_t1 = np.ones(len(x_omi_hat_t1)) - x_omi_hat_t1
			x_omi_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_omi_hat_t1])
			x_delta_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_delta_hat_t1])

			x_omi_hat_t2 = np.random.binomial(N_tot2,p_t2[0],1000) / N_tot2
			x_delta_hat_t2 = np.ones(len(x_omi_hat_t2)) - x_omi_hat_t2
			x_omi_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_omi_hat_t2])
			x_delta_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_delta_hat_t2])

			result = (np.log(x_omi_hat_t2 / x_delta_hat_t2) - np.log(x_omi_hat_t1/x_delta_hat_t1)) / dt

			s_hat = np.mean(result)
			s_hat_var = np.var(result)

			for t in list(country2immuno2time[country]['VAC'].keys()):
				if country2immuno2time[country]['VAC'][t] < country2immuno2time[country]['BOOST'][FLAI_time]:
					country2immuno2time[country]['VAC_cor'][t] = 0.0
				else:
					country2immuno2time[country]['VAC_cor'][t] = country2immuno2time[country]['VAC'][t] - country2immuno2time[country]['BOOST'][FLAI_time]

			t_range_here = np.arange(t1 + tmin,t2 + tmin)
			vac_av = np.mean([country2immuno2time[country]['VAC'][int(t)] for t in t_range_here])
			vac_cor_av = np.mean([country2immuno2time[country]['VAC_cor'][int(t)] for t in t_range_here])
			recov = np.mean([country2immuno2time[country]['RECOV_DELTA_0'][int(t)] for t in t_range_here])
			cases_av = np.mean([country2immuno2time[country]['CASES'][int(t)] * 1000000 for t in t_range_here])
			boosted_av = np.mean([country2immuno2time[country]['BOOST'][int(t)] for t in t_range_here])
			delta_recov = np.mean([country2immuno2time[country]['RECOV_DELTA_0'][int(t)] for t in t_range_here])
			omi_recov = np.mean([country2immuno2time[country]['RECOV_OMI_0'][int(t)] for t in t_range_here])


			if str(FLAI_time) in freq_traj['DELTA'].keys():
				x_delta = freq_traj['DELTA'][str(FLAI_time)]
			else:
				x_delta= 0.0
			if str(FLAI_time) in freq_traj['OMICRON'].keys():
				x_omi = freq_traj['OMICRON'][str(FLAI_time)]
			else:
				x_omi = 0.0
			

			lines.append([country,FLAI_time, Time.coordinateToStringDate(int(t1+tmin)),Time.coordinateToStringDate(int(t2+tmin)), np.round(s_hat,3), 
			np.round(s_hat_var,7), np.round(vac_av,3),np.round(vac_cor_av,3), np.round(recov,3),np.round(boosted_av,3), np.round(delta_recov,3), np.round(omi_recov,3),omi_count[t1], omi_count[t2],delta_count[t1],delta_count[t2],np.round(cases_av,2),x_delta, x_omi])

			t1 = t1 + 7
			t2 = int(t1 + dt)
	else:
		print("No 50%:", country)
		country_min_count[country].append(0)

lines = pd.DataFrame(lines,columns = ['country','FLAI_time','t1','t2','s_hat','s_var','vaccinated','vac_cor','recovered','boosted','delta_recov','omi_recov','omi_count_t1','omi_count_t2','delta_count_t1','delta_count_t2','av_cases','x_delta','x_omi'])
country2pop = ['SWEDEN','CROATIA'] #DROP SWEDEN BC LACK OF BOOSTING DATA
for country in list(set(lines.country)):
	df_c = lines.loc[list(lines.country == country)]
	print(country, len(df_c))
	if len(df_c) < 4:
		country2pop.append(country)

mask = [c not in country2pop for c in list(lines.country)]
lines = lines.loc[mask]
savename='output/s_hat_omi_delta.txt'
lines.to_csv(savename,'\t',index=False)

df_da = []
for country in country2max_recov['AD']['ALPHA'].keys():
	df_da.append([country, country2max_recov['AD']['ALPHA'][country],country2max_recov['AD']['DELTA'][country]])
df_da = pd.DataFrame(df_da,columns=['country','alpha','delta'])
df_da['tot'] = np.array(df_da.delta) + np.array(df_da.alpha)
df_da = df_da.sort_values(by='tot',ascending=False)
df_od = []
for country in country2max_recov['DO']['DELTA'].keys():
	df_od.append([country, country2max_recov['DO']['DELTA'][country],country2max_recov['DO']['OMI'][country]])
df_od = pd.DataFrame(df_od,columns=['country','delta','omi'])
df_od['tot'] = np.array(df_od.delta) + np.array(df_od.omi)
df_od = df_od.sort_values(by='tot',ascending=False)

VOCs = ['ALPHA','BETA','GAMMA','DELTA','EPSILON','KAPPA','LAMBDA']
colors = ['b','g','r','c','m','y','k','lime','salmon','lime']

print("=====================1 - Alpha ============================")

x_limit = 0.01
dt = 30
min_count = 500

country_min_count = defaultdict(lambda: [])

df_delta_alpha = pd.read_csv("s_hat_delta_alpha.txt",'\t',index_col=False)
# countries = sorted(list(set(df_delta_alpha.country)))
lines = []
for country in countries:
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

		Ztot =np.array([int(np.exp(Z[str(t)])) for t in t_range])
		country_min_count[country].append(min(Ztot))
		if np.sum(Ztot > min_count) != len(Ztot):
			print("Low sequence count",country)
			continue

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
		country2immuno2time[country]['CASES'] = {list(meta_country.index)[i]:cases_full[i] for i in range(len(list(meta_country.index)))}
		cases_full_alpha = cases_full * freq_alpha
		recov_alpha = [[np.sum(cases_full_alpha[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['ALPHA'] = {a[1]: a[0] for a in recov_alpha}
		cases_full_wt = cases_full * freq_wt
		recov_wt = [[np.sum(cases_full_wt[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
		country2immuno2time[country]['WT'] = {a[1]: a[0] for a in recov_wt}
		vaccinated = [meta_country.loc[t]['people_fully_vaccinated_per_hundred']/100. for t in t_range]
		if not np.sum([np.isnan(a) for a in vaccinated]) == len(vaccinated):
			vaccinated = clean_vac(t_range,vaccinated)
			country2immuno2time[country]['VAC'] = {t_range[i] : vaccinated[i] for i in range(len(t_range))}

		alpha_count = [int(np.exp(counts['ALPHA'][str(a)])) for a in t_range]
		VOC_counts = defaultdict(lambda: [])
		for t in t_range:
			for VOC in VOCs:
				if VOC in counts.keys():
					if str(t) in counts[VOC].keys():
						VOC_counts[VOC].append(int(np.exp(counts[VOC][str(t)])))
					else:
						VOC_counts[VOC].append(0.0)

		wt_count = np.array([int(np.exp(Z[str(a)])) for a in t_range])
		for VOC in VOC_counts.keys():
			wt_count = wt_count - np.array(VOC_counts[VOC])
		wt_count = wt_count - np.array(alpha_count)

		t1 = 0 
		t2 = t1 + dt
		while t2 + tmin < tmax:
			N_tot1 = alpha_count[t1] + wt_count[t1]
			N_tot2 = alpha_count[t2] + wt_count[t2]
			if alpha_count[t1] < 10 or alpha_count[t2] < 10 or wt_count[t1] < 10 or wt_count[t2] < 10:
				t1 = t1 + 7
				t2 = t1 + dt
				print("t<10",country)
				continue
			p_t1 = [alpha_count[t1] / N_tot1, wt_count[t1] / N_tot1]
			p_t2 = [alpha_count[t2] / N_tot2, wt_count[t2] / N_tot2]


			x_alpha_hat_t1 = np.random.binomial(N_tot1,p_t1[0],1000) / N_tot1
			x_other_hat_t1 = np.ones(len(x_alpha_hat_t1)) - x_alpha_hat_t1
			x_alpha_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_alpha_hat_t1])
			x_other_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_other_hat_t1])

			x_alpha_hat_t2 = np.random.binomial(N_tot2,p_t2[0],1000) / N_tot2
			x_other_hat_t2 = np.ones(len(x_alpha_hat_t2)) - x_alpha_hat_t2
			x_alpha_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_alpha_hat_t2])
			x_other_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_other_hat_t2])

			result = (np.log(x_alpha_hat_t2/x_other_hat_t2) - np.log(x_alpha_hat_t1 / x_other_hat_t1)) / float(dt)
			s_hat = np.mean(result)
			s_hat_var = np.var(result)
			FLAI_time = int((t1 + t2 + 2 * tmin)/2)
			cases_av = np.mean([country2immuno2time[country]['CASES'][int(t)] * 1000000 for t in t_range_here])

			if str(FLAI_time) in freq_traj['ALPHA'].keys():
				x_alpha = freq_traj['ALPHA'][str(FLAI_time)]
			else:
				x_alpha = 0.0
			x_voc = 0.0
			for voc in VOCs:
				if voc in freq_traj.keys():
					if str(FLAI_time) in freq_traj[voc].keys():
						x_voc += freq_traj[voc][str(FLAI_time)]
			x_wt = 1-x_voc
			x_voc = 1 - x_wt - x_alpha

			lines.append([country,FLAI_time,Time.coordinateToStringDate(int(t1+tmin)),Time.coordinateToStringDate(int(t2+tmin)), np.round(s_hat,3), 
				np.round(s_hat_var,7), alpha_count[t1], alpha_count[t2],wt_count[t1],wt_count[t2],tmin,tmax, x_wt, x_alpha, x_voc])

			t1 = t1 + 7
			t2 = t1 + dt
	else:
		country_min_count[country].append(0)

lines = pd.DataFrame(lines,columns = ['country','FLAI_time','t1','t2','s_hat','s_var','alpha_count_t1','alpha_count_t2','wt_count_t1','wt_count_t2','tmin','tmax','x_wt','x_alpha','x_voc'])
country2pop = []
for country in list(set(lines.country)):
	df_c = lines.loc[list(lines.country == country)]
	if len(df_c) < 4:
		country2pop.append(country)
mask = [c not in country2pop for c in list(lines.country)]
lines = lines.loc[mask]
savename = 'output/s_hat_alpha_wt.txt'
lines.to_csv(savename,'\t',index=False)


print("=====================o1 - o2============================")

df_da = pd.read_csv("output/df_da_result.txt",'\t',index_col=False)
df_s_da = pd.read_csv("output/s_hat_delta_alpha.txt",sep='\t',index_col=False)

df_od = pd.read_csv("doutput/f_od_result.txt",'\t',index_col=False)
df_s_od = pd.read_csv("output/s_hat_omi_delta.txt",sep='\t',index_col=False)

meta_df = pd.read_csv("DATA/clean_data.txt",sep='\t',index_col=False)   

countries_da = sorted(list(set(df_da.Country)))
countries_od = sorted(list(set(df_od.Country)))
countries = sorted(list(set(countries_da).intersection(set(countries_od))))


x_limit = 0.01
dt = 40.0
min_count = 500

country_min_count = defaultdict(lambda: [])

country2immuno2time = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
lines = []
for country in countries:
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


    dates_ba2 = list(counts['OMICRON BA.2'].keys())
    dates_ba1 = list(counts['OMICRON BA.1'].keys())
    dates_ba2 = [int(a) for a in dates_ba2]
    dates_ba1 = [int(a) for a in dates_ba1]
    dates_ba2 = sorted(dates_ba2)
    dates_ba1 = sorted(dates_ba1)
    dates_ba1 = [a for a in dates_ba1 if a < 44702]
    dates_ba2 = [a for a in dates_ba2 if a < 44702] #before 2022-05-22
    if country == 'NY' or country == 'USA':
        dates_ba2 = [a for a in dates_ba2 if a < 44702 and a > 44534] #before 2022-05-22

    tmin = min(set(dates_ba2).intersection(set(dates_ba1)))
    tmax = max(set(dates_ba2).intersection(set(dates_ba1)))
    t_range = np.arange(tmin,tmax)
    ba1_count = [int(np.exp(counts['OMICRON BA.1'][str(a)])) for a in t_range]
    ba2_count = [int(np.exp(counts['OMICRON BA.2'][str(a)])) for a in t_range]
    N_tot = np.array(ba1_count) + np.array(ba2_count)
    check = 'not_okay'
    for t in t_range:
        x_ba2 = ba2_count[t-tmin] / N_tot[t-tmin]
        x_ba1 = ba1_count[t-tmin] / N_tot[t-tmin]
        if x_ba2 > x_limit:
            tminnew = t
            check = 'okay'
            break
    for t in t_range:
        x_ba2 = ba2_count[t-tmin] / N_tot[t-tmin]
        x_ba1 = ba1_count[t-tmin] / N_tot[t-tmin]
        if x_ba2 > 1 - x_limit:
            tmaxnew = t
            break
    tmin = tminnew
    tmax = tmaxnew
    t_range= np.arange(tmin,tmax)

    Ztot =np.array([int(np.exp(Z[str(t)])) for t in t_range])
    country_min_count[country].append(min(Ztot))
    if np.sum(Ztot > min_count) != len(Ztot):
        continue

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
    for time_meas in  list(meta_country.index):
        for t in list(country2immuno2time[country]['VAC'].keys()):
            if country2immuno2time[country]['VAC'][t] < country2immuno2time[country]['BOOST'][time_meas]:
                country2immuno2time[country][time_meas][t] = 0.0
            else:
                country2immuno2time[country][time_meas][t] = country2immuno2time[country]['VAC'][t] - country2immuno2time[country]['BOOST'][time_meas]
    x_delta = []
    x_omi = []
    x_alpha = []
    x_ba1 = []
    x_ba2 = []
    x_ba4 = []
    x_ba5 = []
    for t in list(meta_country.index):
        if str(t) in freq_traj['DELTA'].keys():
            x_delta.append(freq_traj['DELTA'][str(t)])
        else:
            x_delta.append(0.0)
        if str(t) in freq_traj['OMICRON'].keys():
            x_omi.append(freq_traj['OMICRON'][str(t)])
        else:
            x_omi.append(0.0)
        if str(t) in freq_traj['ALPHA'].keys():
            x_alpha.append(freq_traj['ALPHA'][str(t)])
        else:
            x_alpha.append(0.0)

        if str(t) in freq_traj['OMICRON BA.1'].keys():
            x_ba1.append(freq_traj['OMICRON BA.1'][str(t)])
        else:
            x_ba1.append(0.0)
        if str(t) in freq_traj['OMICRON BA.2'].keys():
            x_ba2.append(freq_traj['OMICRON BA.2'][str(t)])
        else:
            x_ba2.append(0.0)
        if str(t) in freq_traj['OMICRON BA.4'].keys():
            x_ba4.append(freq_traj['OMICRON BA.4'][str(t)])
        else:
            x_ba4.append(0.0)
        if str(t) in freq_traj['OMICRON BA.5'].keys():
            x_ba5.append(freq_traj['OMICRON BA.5'][str(t)])
        else:
            x_ba5.append(0.0)
    
    ba1_count = [int(np.exp(counts['OMICRON BA.1'][str(a)])) for a in t_range]
    ba2_count = [int(np.exp(counts['OMICRON BA.2'][str(a)])) for a in t_range]


    t1 = 0 
    t2 = int(t1 + dt)
    while t2 + tmin < tmax:
        FLAI_time = int((t1 + t2 + 2*tmin)/2)
        N_tot1 = ba1_count[t1] + ba2_count[t1]
        N_tot2 = ba1_count[t2] + ba2_count[t2]
        if ba1_count[t1] < 10 or ba1_count[t2] < 10 or ba2_count[t1] < 10 or ba2_count[t2] < 10:
            t1 = t1 + 7
            t2 = int(t1 + dt)
            continue
        p_t1 = [ba1_count[t1] / N_tot1, ba2_count[t1] / N_tot1]
        p_t2 = [ba1_count[t2] / N_tot2, ba2_count[t2] / N_tot2]

        x_ba1_hat_t1 = np.random.binomial(N_tot1,p_t1[0],1000) / N_tot1
        x_ba2_hat_t1 = np.ones(len(x_ba1_hat_t1)) - x_ba1_hat_t1
        x_ba1_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_ba1_hat_t1])
        x_ba2_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_ba2_hat_t1])

        x_ba1_hat_t2 = np.random.binomial(N_tot2,p_t2[0],1000) / N_tot2
        x_ba2_hat_t2 = np.ones(len(x_ba1_hat_t2)) - x_ba1_hat_t2
        x_ba1_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_ba1_hat_t2])
        x_ba2_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_ba2_hat_t2])

        result = (np.log(x_ba2_hat_t2 / x_ba1_hat_t2) - np.log(x_ba2_hat_t1 / x_ba1_hat_t1)) / dt
        s_hat = np.mean(result)
        s_hat_var = np.var(result)


        t_range_here = np.arange(t1 + tmin,t2 + tmin)
        if str(FLAI_time) in freq_traj['OMICRON BA.1'].keys():
            x_ba1 = freq_traj['OMICRON BA.1'][str(FLAI_time)]
        else:
            x_ba1 = 0.0
        if str(FLAI_time) in freq_traj['OMICRON BA.2'].keys():
            x_ba2 = freq_traj['OMICRON BA.2'][str(FLAI_time)]



        lines.append([country,FLAI_time, Time.coordinateToStringDate(int(t1+tmin)),Time.coordinateToStringDate(int(t2+tmin)), np.round(s_hat,3), 
        np.round(s_hat_var,7), ba1_count[t1], ba1_count[t2],ba2_count[t1],ba2_count[t2],tmin,tmax,x_ba1,x_ba2])

        t1 = t1 + 7
        t2 = int(t1 + dt)


lines = pd.DataFrame(lines,columns = ['country','FLAI_time','t1','t2','s_hat','s_var','ba1_count_t1','ba1_count_t2','ba2_count_t1','ba2_count_t2','tmin','tmax','x_ba1','x_ba2'])
savename = 'output/s_hat_ba2_ba1.txt'
lines.to_csv(savename,'\t',index=False)

print("=====================o2 - o4============================")

x_limit = 0.01
dt = 30.0
min_count = 500
country_min_count = defaultdict(lambda: [])

country2immuno2time = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
lines = []
for country in countries:
    with open("DATA/2022_06_22/freq_traj_" + country.upper() + ".json",'r') as f:
        freq_traj = json.load(f)
    with open("DATA/2022_06_22/multiplicities_" + country.upper() + ".json",'r') as f:
        counts = json.load(f)
    with open("DATA/2022_06_22/multiplicities_Z_" + country.upper() + ".json",'r') as f:
        Z = json.load(f)

    dates_ba2 = list(counts['OMICRON BA.2'].keys())
    dates_ba4 = list(counts['OMICRON BA.4'].keys())
    dates_ba5 = list(counts['OMICRON BA.5'].keys())
    dates_ba2 = [int(a) for a in dates_ba2]
    dates_ba4 = [int(a) for a in dates_ba4]
    dates_ba5 = [int(a) for a in dates_ba5]
    dates_ba2 = sorted(dates_ba2)
    dates_ba4 = sorted(dates_ba4)
    dates_ba5 = sorted(dates_ba5)
    dates_ba4 = [a for a in dates_ba4 if a < Time.dateToCoordinate("2022-05-23")]
    dates_ba5 = [a for a in dates_ba5 if a < Time.dateToCoordinate("2022-05-23")]
    dates_ba45 = sorted(list(set(dates_ba4 + dates_ba5)))
    dates_ba2 = [a for a in dates_ba2 if a < Time.dateToCoordinate("2022-05-23")] #before 2022-05-22
    if country == 'NY' or country == 'USA':
        dates_ba2 = [a for a in dates_ba2 if a < Time.dateToCoordinate("2022-05-23") and a > 44534] #before 2022-05-22

    tmin = min(set(dates_ba2).intersection(set(dates_ba45)))
    tmax = max(set(dates_ba2).intersection(set(dates_ba45)))
    t_range = np.arange(tmin,tmax)

    ba2_count = [int(np.exp(counts['OMICRON BA.2'][str(a)])) for a in t_range]
    ba45_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts['OMICRON BA.4'].keys():
            c += np.exp(counts['OMICRON BA.4'][str(t)])
        if str(t) in counts['OMICRON BA.5'].keys():
            c += np.exp(counts['OMICRON BA.5'][str(t)])
        ba45_count.append(int(c))

    N_tot = np.array(ba2_count) + np.array(ba45_count)


    check = 'not_okay'
    for t in t_range:
        x_ba2 = ba2_count[t-tmin] / N_tot[t-tmin]
        x_ba45 = ba45_count[t-tmin] / N_tot[t-tmin]
        if x_ba45 > x_limit:
            tminnew = t
            check = 'okay'
            break

    tmin = tminnew
    tmax = Time.dateToCoordinate("2022-05-22")
    t_range= np.arange(tmin,tmax)


    ba2_count = [int(np.exp(counts['OMICRON BA.2'][str(a)])) for a in t_range]
    ba45_count = []
    for t in t_range:
        c = 0.0
        if str(t) in counts['OMICRON BA.4'].keys():
            c += np.exp(counts['OMICRON BA.4'][str(t)])
        if str(t) in counts['OMICRON BA.5'].keys():
            c += np.exp(counts['OMICRON BA.5'][str(t)])
        ba45_count.append(int(c))
    N_tot = np.array(ba2_count) + np.array(ba45_count)

    t1 = t_range[0] - tmin
    t2 = t_range[-1] - tmin
    dt = float(t2 -t1)


    FLAI_time = int((t1 + t2 + 2*tmin)/2)
    N_tot1 = ba45_count[t1] + ba2_count[t1]
    N_tot2 = ba45_count[t2] + ba2_count[t2]

    p_t1 = [ba45_count[t1] / N_tot1, ba2_count[t1] / N_tot1]
    p_t2 = [ba45_count[t2] / N_tot2, ba2_count[t2] / N_tot2]

    x_ba45_hat_t1 = np.random.binomial(N_tot1,p_t1[0],1000) / N_tot1
    x_ba2_hat_t1 = np.ones(len(x_ba45_hat_t1)) - x_ba45_hat_t1
    x_ba45_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_ba45_hat_t1])
    x_ba2_hat_t1 = np.array([x if x != 0 else 1/(N_tot1+1) for x in x_ba2_hat_t1])

    x_ba45_hat_t2 = np.random.binomial(N_tot2,p_t2[0],1000) / N_tot2
    x_ba2_hat_t2 = np.ones(len(x_ba45_hat_t2)) - x_ba45_hat_t2
    x_ba45_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_ba45_hat_t2])
    x_ba2_hat_t2 = np.array([x if x != 0 else 1/(N_tot2+1) for x in x_ba2_hat_t2])

    result = (np.log(x_ba45_hat_t2 / x_ba2_hat_t2) - np.log(x_ba45_hat_t1 / x_ba2_hat_t1)) / dt
    s_hat = np.mean(result)
    s_hat_var = np.var(result)

    t_range_here = np.arange(t1 + tmin,t2 + tmin)


    if str(FLAI_time) in freq_traj['OMICRON BA.2'].keys():
        x_ba2 = freq_traj['OMICRON BA.2'][str(FLAI_time)]
    else:
        x_ba2 = 0.0

    x_ba4 = 0.0
    x_ba5 = 0.0
    if str(FLAI_time) in freq_traj['OMICRON BA.4'].keys():
        x_ba4 = freq_traj['OMICRON BA.4'][str(FLAI_time)]
    if str(FLAI_time) in freq_traj['OMICRON BA.5'].keys():
        x_ba5 = freq_traj['OMICRON BA.5'][str(FLAI_time)]
    x_ba45 = x_ba4 + x_ba5

    lines.append([country,FLAI_time, Time.coordinateToStringDate(int(t1+tmin)),Time.coordinateToStringDate(int(t2+tmin)), np.round(s_hat,3), 
    np.round(s_hat_var,7), ba2_count[t1], ba2_count[t2],ba45_count[t1],ba45_count[t2],tmin,tmax,x_ba2,x_ba45])

    t1 = t1 + 7
    t2 = int(t1 + dt)

lines = pd.DataFrame(lines,columns = ['country','FLAI_time','t1','t2','s_hat','s_var','ba2_count_t1','ba2_count_t2','ba45_count_t1','ba45_count_t2','tmin','tmax','x_ba2','x_ba45'])
savename = 'output/s_hat_ba45_ba2.txt'
lines.to_csv(savename,'\t',index=False)
