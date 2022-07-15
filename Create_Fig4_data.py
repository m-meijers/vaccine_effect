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
import scipy.integrate as integrate

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

VOCs = ['ALPHA','DELTA','OMICRON','BETA','EPSILON','IOTA','MU','GAMMA']

df_da = pd.read_csv("output/df_da_result.txt",'\t',index_col=False)
df_s_da = pd.read_csv("output/s_hat_delta_alpha.txt",sep='\t',index_col=False)

df_od = pd.read_csv("output/df_od_result.txt",'\t',index_col=False)
df_s_od = pd.read_csv("output/s_hat_omi_delta.txt",sep='\t',index_col=False)

meta_df = pd.read_csv("DATA/clean_data.txt",sep='\t',index_col=False)   

countries_da = sorted(list(set(df_da.Country)))
countries_od = sorted(list(set(df_od.Country)))

countries = sorted(list(set(countries_da).intersection(set(countries_od))))

country2immuno2time = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: 0.0)))
for country in countries:
    x_limit = 0.01
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

    #GET T_RANGE
    t_alpha = sorted(list(freq_traj['ALPHA'].keys()))
    t_delta = sorted(list(freq_traj['DELTA'].keys()))
    t_omi = sorted(list(freq_traj['OMICRON'].keys()))
    t_delta = [a for a in t_delta if int(a) > 44255]
    t_omi = [a for a in t_omi if int(a) < 44620 and int(a) > 44469]
    if country=='ITALY':
        t_delta = [a for a in t_delta if int(a) > 44287]
    for t in t_alpha:
        if freq_traj['ALPHA'][t] > 0.01:
            tmin_alpha = t
            break
    for t in t_delta:
        if freq_traj['DELTA'][t] > 0.01:
            tmin_delta = t
            break
    for t in t_omi:
        if freq_traj['OMICRON'][t] > 0.01:
            tmin_omi = t
            break

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
    x_wt = []
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
        x_voc = 0.0
        for voc in VOCs:
            if voc in freq_traj.keys():
                if str(t) in freq_traj[voc].keys():
                    x_voc += freq_traj[voc][str(t)]
        x_wt.append(1-x_voc)

        
    freq_wt = np.array(x_wt)
    freq_delta = np.array(x_delta)
    freq_omi = np.array(x_omi)
    freq_alpha = np.array(x_alpha)
    freq_ba1 = np.array(x_ba1)
    freq_ba2 = np.array(x_ba2)
    freq_ba45 = np.array(x_ba4) + np.array(x_ba5)

    cases_full = [meta_country.loc[t]['new_cases']/meta_country.loc[t]['population'] for t in list(meta_country.index)]
    cases_full = clean_vac(list(meta_country.index),cases_full)
    cases_full_alpha = cases_full * freq_alpha
    cases_full_wt = cases_full * freq_wt
    cases_full_delta = cases_full * freq_delta
    cases_full_omi = cases_full * freq_omi
    cases_full_ba1 = cases_full * freq_ba1
    cases_full_ba2 = cases_full * freq_ba2
    cases_full_ba45 = cases_full * freq_ba45

    recov_delta = [[np.sum(cases_full_delta[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_omi = [[np.sum(cases_full_omi[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_ba1 = [[np.sum(cases_full_ba1[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_ba2 = [[np.sum(cases_full_ba2[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_ba45 = [[np.sum(cases_full_ba45[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_alpha = [[np.sum(cases_full_alpha[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    recov_wt = [[np.sum(cases_full_wt[0:t-list(meta_country.index)[0]]), t] for t in list(meta_country.index)]
    country2immuno2time[country]['RECOV_DELTA'] = {a[1]: a[0] for a in recov_delta}
    country2immuno2time[country]['RECOV_OMI'] = {a[1]: a[0] for a in recov_omi}
    country2immuno2time[country]['RECOV_ALPHA'] = {a[1]: a[0] for a in recov_alpha}
    country2immuno2time[country]['RECOV_WT'] = {a[1]: a[0] for a in recov_wt}
    country2immuno2time[country]['RECOV_BA1'] = {a[1]: a[0] for a in recov_ba1}
    country2immuno2time[country]['RECOV_BA2'] = {a[1]: a[0] for a in recov_ba2}
    country2immuno2time[country]['RECOV_BA45'] = {a[1]: a[0] for a in recov_ba45}


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
gamma_vac_od = 0.29
gamma_inf_od = 0.58

k=3.0 #2.2-- 4.2 -> sigma = 1/1.96
n50 = np.log10(0.2 * 94) #0.14 -- 0.28 -> sigma 0.06/1.96
time_decay = 90
# time_decay = 75
tau_decay = time_decay



def sigmoid_func(t,mean=n50,s=k):
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
        W = sigmoid_func(np.log10(94) - np.log10(np.exp(dt_vac/tau_decay)), n50, k)
        # R_k += np.exp(- dt_vac / tau_decay) * weight
        R_k += W * weight
        time_index += 1

    return R_k


VOC_clades = ['ALPHA','DELTA','OMICRON BA.1','OMICRON BA.2','OMICRON BA.45']

data_set = []
data_fitness = []
for c in countries:
    meta_country= meta_df.loc[list(meta_df['location'] == c[0] + c[1:].lower())]
    if len(c) <= 3:
        meta_country= meta_df.loc[list(meta_df['location'] == c)]
    meta_country.index = meta_country['FLAI_time']
    with open("DATA/2022_06_22/freq_traj_" + c.upper() + ".json",'r') as f:
        freq_traj = json.load(f)
    with open("DATA/2022_06_22/multiplicities_" + c.upper() + ".json",'r') as f:
        counts = json.load(f)
    with open("DATA/2022_06_22/multiplicities_Z_" + c.upper() + ".json",'r') as f:
        Z = json.load(f)
    for t in list(meta_country.index):
        if t > Time.dateToCoordinate("2022-05-23"):
            continue
        C_delta_vac, C_alpha_vac =   integrate_S_component(c,t, tau_decay,country2immuno2time, 'VAC_cor', dT_delta_vac, dT_alpha_vac)
        C_omi_vac, C_wt_vac =   integrate_S_component(c,t, tau_decay,country2immuno2time, 'VAC_cor', dT_omi_vac, dT_wt_vac)
        C_omi_vac0, C_delta_vac0 =   integrate_S_component(c,t, tau_decay,country2immuno2time, 'VAC', dT_omi_vac, dT_delta_vac)
        if np.sum(list(country2immuno2time[c]['BOOST'].values())) != 0.0:
            C_omi_booster, C_delta_booster = integrate_S_component(c,t, tau_decay,country2immuno2time, 'BOOST', dT_omi_booster, dT_delta_booster)
            C_wt_booster, C_alpha_booster = integrate_S_component(c,t, tau_decay,country2immuno2time, 'BOOST', dT_wt_booster, dT_alpha_booster)
            C_ba2_booster, C_ba45_booster = integrate_S_component(c,t, tau_decay,country2immuno2time, 'BOOST', dT_ba2_booster, dT_ba45_booster)
        else:
            C_omi_booster = 0.0
            C_delta_booster = 0.0
            C_ba2_booster = 0.0
            C_ba45_booster= 0.0
            C_wt_booster = 0.0

        C_delta_alpha, C_alpha_alpha = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_ALPHA', dT_delta_alpha, 1.0)
        C_omi_alpha, C_wt_alpha = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_ALPHA', dT_omi_alpha, dT_wt_alpha)

        C_delta_delta, C_alpha_delta = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_DELTA', dT_delta_delta, dT_alpha_delta)
        C_omi_delta, C_wt_delta = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_DELTA', dT_omi_delta, dT_wt_delta)

        C_omi_omi, C_delta_omi = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_OMI', 1.0, dT_delta_omi)
        C_wt_omi, C_alpha_omi = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_OMI', dT_wt_omi, dT_alpha_omi)

        C_ba1_ba1,_ = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_BA1', 1., dT_ba45_ba1)
        C_ba2_ba1, C_ba45_ba1 = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_BA1', dT_ba2_ba1, dT_ba45_ba1)
        C_ba1_ba2, C_ba45_ba2 = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_BA2', dT_ba1_ba2, dT_ba45_ba2)
        C_ba2_ba2, _ = integrate_S_component(c,t, tau_decay,country2immuno2time, 'RECOV_BA2', 1., 2.)

        R_wt  = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_WT')
        R_vac = approx_immune_weight(c,t,tau_decay,country2immuno2time,'VAC_cor')
        R_boost = approx_immune_weight(c,t,tau_decay,country2immuno2time,'BOOST')
        R_alpha = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_ALPHA')
        R_delta = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_DELTA')
        R_omi = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_OMI')
        R_ba1 = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_BA1')
        R_ba2 = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_BA2')
        R_ba45 = approx_immune_weight(c,t,tau_decay,country2immuno2time,'RECOV_BA45')

        if str(t) in freq_traj['DELTA'].keys():
            x_delta = freq_traj['DELTA'][str(t)]
        else:
            x_delta = 0.0
        if str(t) in freq_traj['OMICRON'].keys():
            x_omi = freq_traj['OMICRON'][str(t)]
        else:
            x_omi = 0.0
        if str(t) in freq_traj['ALPHA'].keys():
            x_alpha = freq_traj['ALPHA'][str(t)]
        else:
            x_alpha = 0.0

        if str(t) in freq_traj['OMICRON BA.1'].keys():
            x_ba1 = freq_traj['OMICRON BA.1'][str(t)]
        else:
            x_ba1 = 0.0
        if str(t) in freq_traj['OMICRON BA.2'].keys():
            x_ba2 = freq_traj['OMICRON BA.2'][str(t)]
        else:
            x_ba2 = 0.0
        if str(t) in freq_traj['OMICRON BA.4'].keys():
            x_ba4 = freq_traj['OMICRON BA.4'][str(t)]
        else:
            x_ba4 = 0.0
        if str(t) in freq_traj['OMICRON BA.5'].keys():
            x_ba5 = freq_traj['OMICRON BA.5'][str(t)]
        else:
            x_ba5 = 0.0
        x_voc = 0.0
        for voc in VOCs:
            if voc in freq_traj.keys():
                if str(t) in freq_traj[voc].keys():
                    x_voc += freq_traj[voc][str(t)]
        x_wt = 1-x_voc

        data_set.append([c, t, x_wt, 1-x_wt-x_alpha-x_delta-x_omi, x_alpha, x_delta, x_omi, x_ba1, x_ba2, x_ba4 + x_ba5, C_alpha_vac, C_delta_vac, C_omi_vac,C_wt_vac, C_omi_vac0, C_delta_vac0, C_alpha_booster, C_delta_booster, C_omi_booster,C_ba2_booster,C_ba45_booster, C_wt_booster, C_alpha_alpha,
         C_delta_alpha, C_omi_alpha,C_wt_alpha, C_alpha_delta, C_delta_delta, C_omi_delta,C_wt_delta, C_alpha_omi, C_delta_omi,C_wt_omi, C_omi_omi,C_ba1_ba1, C_ba2_ba1, C_ba45_ba1,C_ba1_ba2,C_ba2_ba2,C_ba45_ba2,R_wt,R_vac,R_boost,
         R_alpha,R_delta,R_omi,R_ba1, R_ba2])


df = pd.DataFrame(data_set,columns=['country','time','x_wt','x_voc','x_alpha','x_delta','x_omi','x_ba1','x_ba2','x_ba45','C_av','C_dv','C_ov','C_wtv','C_ov0','C_dv0','C_ab','C_db','C_ob','C_a2b','C_a45b','C_wtb','C_aa','C_da','C_oa','C_wta','C_ad','C_dd','C_od','C_wtd','C_ao','C_do','C_wto','C_oo',
    'C_a1a1','C_a2a1','C_a45a1','C_a1a2','C_a2a2','C_a45a2','R_wt','R_vac','R_boost','R_alpha','R_delta','R_omi','R_ba1','R_ba2'])
df.to_csv("output/Fig4_data.txt",'\t',index=False)


c_channels = ['C_av','C_dv','C_ov','C_wtv','C_ov0','C_dv0','C_ab','C_db','C_ob','C_a2b','C_a45b','C_wtb','C_aa','C_da','C_oa','C_wta','C_ad','C_dd','C_od','C_wtd','C_ao','C_do','C_wto','C_oo',
    'C_a1a1','C_a2a1','C_a45a1','C_a1a2','C_a2a2','C_a45a2']
f_channels = VOC_clades + ['SD_' + voc for voc in VOC_clades]
#Average the R weights over all the countries
time2R = defaultdict(lambda: defaultdict(lambda: []))
df = pd.read_csv("output/Fig4_data.txt",'\t',index_col=False)
for c in countries:
    df_c = df.loc[list(df.country == c)]

    for line in df_c.iterrows():
        line = line[1]
        time2R['R_wt'][line.time].append(line.R_wt)
        time2R['R_vac'][line.time].append(line.R_vac)
        time2R['R_boost'][line.time].append(line.R_boost)
        time2R['R_alpha'][line.time].append(line.R_alpha)
        time2R['R_delta'][line.time].append(line.R_delta)
        time2R['R_omi'][line.time].append(line.R_omi)
        time2R['R_ba1'][line.time].append(line.R_ba1)
        time2R['R_ba2'][line.time].append(line.R_ba2)
        time2R['x_wt'][line.time].append(line.x_wt)
        time2R['x_voc'][line.time].append(line.x_voc)
        time2R['x_alpha'][line.time].append(line.x_alpha)
        time2R['x_delta'][line.time].append(line.x_delta)
        time2R['x_ba1'][line.time].append(line.x_ba1)
        time2R['x_ba2'][line.time].append(line.x_ba2)
        time2R['x_ba45'][line.time].append(line.x_ba45)

        for C in c_channels:
            time2R[C][line.time].append(float(line[C]))


R_av = []
for t in sorted(list(time2R['R_vac'].keys())):
    X = [np.mean(time2R['x_wt'][t]),np.mean(time2R['x_voc'][t]),np.mean(time2R['x_alpha'][t]), np.mean(time2R['x_delta'][t]),np.mean(time2R['x_ba1'][t]),np.mean(time2R['x_ba2'][t]),np.mean(time2R['x_ba45'][t])]
    X = list(np.array(X) / np.sum(X))
    C_all = []
    for CC in c_channels:
        C_all.append(np.mean(time2R[CC][t]))

    R_av.append([int(t)] + X  + [np.mean(time2R['R_wt'][t]), np.mean(time2R['R_vac'][t]), np.mean(time2R['R_boost'][t]), np.mean(time2R['R_alpha'][t]), np.mean(time2R['R_delta'][t]), np.mean(time2R['R_omi'][t]), 
        np.mean(time2R['R_ba1'][t]), np.mean(time2R['R_ba2'][t]),np.mean(time2R['R_ba45'][t])] + C_all)# + F_all)
R_av = pd.DataFrame(R_av,columns=['time','x_wt','x_voc','x_alpha','x_delta','x_ba1','x_ba2','x_ba45','R_wt','R_vac','R_boost','R_alpha','R_delta','R_omi','R_ba1','R_ba2','R_ba45'] + c_channels)# + f_channels)
R_av.to_csv("output/R_average.txt",'\t',index=False)

