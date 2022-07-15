import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from flai.util.Time import Time
from sklearn import linear_model
from matplotlib.lines import Line2D
import scipy.stats as ss
import json
import scipy.optimize as so

df_da = pd.read_csv("output/Data_da.txt",sep='\t',index_col=False)
df_od = pd.read_csv("output/Data_od.txt",sep='\t',index_col=False)

countries_da = sorted(list(set(df_da.Country)))
countries_ad = countries_da
countries_od = sorted(list(set(df_od.Country)))

country_variance = []
for c in countries_da:
    df_c = df_da.loc[list(df_da.Country == c)]
    meansvar = np.mean(df_c.s_var)
    country_variance.append(meansvar)
median_svar_da = np.median(country_variance)

country_variance = []
for c in countries_od:
    df_c = df_od.loc[list(df_od.Country == c)]
    meansvar = np.mean(df_c.s_var)
    country_variance.append(meansvar)
median_svar_od = np.median(country_variance)

df_da['s_var'] = np.array(df_da['s_var'] + median_svar_da)
df_od['s_var'] = np.array(df_od['s_var'] + median_svar_od)

country_coef = []
for i in range(len(countries_da)): 
    country_coef.append([])

c = df_da.iloc[0].Country
country_order = [c]
index = 0
for line in df_da.iterrows():
    line=line[1]
    c_here = line.Country
    if c_here != c:
        index += 1
        c = c_here    
        country_order.append(c)
    for i in range(len(countries_da)):
        if i == index:
            country_coef[i].append(1)
        else:
            country_coef[i].append(0)
for i in range(len(country_coef)):
    country_coef[i] += list(np.zeros(len(df_od)))
country_coef_da = np.array(country_coef)


country_coef = []
for i in range(len(countries_od)):
    country_coef.append(list(np.zeros(len(df_da))))
c = df_od.iloc[0].Country
country_order = [c]
index = 0
for line in df_od.iterrows():
    line=line[1]
    c_here = line.Country
    if c_here != c:
        index += 1
        c = c_here    
        country_order.append(c)
    for i in range(len(countries_od)):
        if i == index:
            country_coef[i].append(1)
        else:
            country_coef[i].append(0)
country_coef_od = np.array(country_coef)

### CONSTRUCT THE X matrix
dC_vac_da = df_da.iloc[0].dC_vac
dC_alpha_da = df_da.iloc[0].dC_alpha
dC_delta_da = df_da.iloc[0].dC_delta

dC_vac_od = df_od.iloc[0].dC_vac
dC_boost_od = df_od.iloc[0].dC_boost
dC_delta_od = df_od.iloc[0].dC_delta
dC_omi_od = df_od.iloc[0].dC_omi



Y = np.array(list(df_da['s_hat']) + list(df_od['s_hat']))
W = np.array(list(1/np.array(df_da['s_var'])) + list(1 / np.array(df_od['s_var'])))
LR = linear_model.LinearRegression(fit_intercept=False,positive=True) #COEF: [gamma_vac_da, gamma_vac_od, gamma_omi_delta, gamma_boost_od,...s_0_da..., ...s_0_od...]

####
kappa_da = 2. #prevents runaway
X = [list(dC_vac_da * np.array(df_da['x_eff_vac']) + kappa_da * dC_alpha_da * np.array(df_da['x_eff_alpha']) + kappa_da * dC_delta_da * np.array(df_da['x_eff_delta'])) + list(np.zeros(len(df_od)))]
X += [list(np.zeros(len(df_da))) + list(np.array(df_od['x_eff_vac']) * dC_vac_od + np.array(df_od['x_eff_boost']) * dC_boost_od), list(np.zeros(len(df_da))) + list(np.array(df_od['x_eff_delta']) * dC_delta_od +  np.array(df_od['x_eff_omi']) * dC_omi_od)]
X = np.array(X)
X = np.append(X,country_coef_da,axis=0)
X = np.append(X, country_coef_od,axis=0)
X = X.transpose()

LR.fit(X,Y,sample_weight=W)
R = LR.score(X,Y,sample_weight=W)

s_model = LR.coef_ * X 
s_model = s_model.sum(axis=1) + LR.intercept_

df_da['s_model'] = s_model[:len(df_da)]
df_da['s_0']  = np.dot(X[:len(df_da),3:3+len(countries_da)], LR.coef_[3:3+len(countries_da)])

df_od['s_model'] = s_model[len(df_da):]
df_od['s_0']  = np.dot(X[len(df_da):,3+len(countries_da):], LR.coef_[3 + len(countries_da):])


L_da = np.sum(ss.norm.logpdf(np.array(df_da['s_model']),np.array(df_da['s_hat']),np.sqrt(np.array(df_da.s_var))))
L_od = np.sum(ss.norm.logpdf(np.array(df_od['s_model']),np.array(df_od['s_hat']),np.sqrt(np.array(df_od.s_var))))

s0_ad_mean = np.mean(LR.coef_[3:len(countries_da)])
s0_ad_sd = np.sqrt(np.var(LR.coef_[3:len(countries_da)]))
s0_od_mean = np.mean(LR.coef_[3+len(countries_da):])
s0_od_sd = np.sqrt(np.var(LR.coef_[3+ len(countries_da):]))

BIC = (len(LR.coef_)) * np.log(len(df_da) + len(df_od)) - 2 * (L_da + L_od) 

print(LR.coef_[0], LR.coef_[1], LR.coef_[2],L_da, L_od, L_da + L_od, BIC)
selection_list = defaultdict(lambda: defaultdict(lambda: []))

for c in countries_ad:
    df_c = df_da.loc[list(df_da.Country == c)]
    s0 = df_c.iloc[0].s_0
    s_vac = np.array(df_c.x_eff_vac) * LR.coef_[0] * dC_vac_da
    s_alpha = np.array(df_c.x_eff_alpha) * LR.coef_[0] * dC_alpha_da * kappa_da
    s_delta = np.array(df_c.x_eff_delta) * LR.coef_[0] * dC_delta_da * kappa_da

    selection_list['s0']['means'].append(s0)
    selection_list['s_vac']['means'].append(np.mean(s_vac))
    selection_list['s_alpha']['means'].append(np.mean(s_alpha))
    selection_list['s_delta']['means'].append(np.mean(s_delta))
    selection_list['s_vac']['var'].append(np.sqrt(np.var(s_vac)))
    selection_list['s_alpha']['var'].append(np.sqrt(np.var(s_alpha)))
    selection_list['s_delta']['var'].append(np.sqrt(np.var(s_delta)))
    
    
a = [np.mean(selection_list['s0']['means']),np.mean(selection_list['s_vac']['means']),np.mean(selection_list['s_alpha']['means']),np.mean(selection_list['s_delta']['means'])]
b = [np.mean(selection_list['s0']['var']),np.mean(selection_list['s_vac']['var']),np.mean(selection_list['s_alpha']['var']),np.mean(selection_list['s_delta']['var'])]
print("s_0_ad:")
print(a[0], np.sqrt(np.var(selection_list['s0']['means'])))
print("svac, salpha, sdelta")
print(a[1:])
for c in countries_od:
    df_c = df_od.loc[list(df_od.Country == c)]
    s0 = df_c.iloc[0].s_0
    s_vac0 = np.array(df_c.x_eff_vac0) * LR.coef_[1] * dC_vac_od
    s_vac = np.array(df_c.x_eff_vac) * LR.coef_[1] * dC_vac_od
    # s_boost = np.array(df_c.x_eff_boost) * LR.coef_[1] * dC_boost_od
    s_boost = np.array(df_c.x_eff_boost) * LR.coef_[1] * dC_boost_od
    s_boost_eff = s_vac - s_vac0 + s_boost
    s_delta = np.array(df_c.x_eff_delta) * LR.coef_[2] * dC_delta_od
    s_omi = np.array(df_c.x_eff_omi) * LR.coef_[2] * dC_omi_od
    
    selection_list['OD_s0']['means'].append(s0)
    selection_list['OD_s_vac']['means'].append(np.mean(s_vac0))
    selection_list['OD_s_boost']['means'].append(np.mean(s_boost_eff))
    selection_list['OD_s_delta']['means'].append(np.mean(s_delta))
    selection_list['OD_s_omi']['means'].append(np.mean(s_omi))
    selection_list['OD_s_vac']['var'].append(np.sqrt(np.var(s_vac)))
    selection_list['OD_s_boost']['var'].append(np.sqrt(np.var(s_boost)))
    selection_list['OD_s_delta']['var'].append(np.sqrt(np.var(s_delta)))
    selection_list['OD_s_omi']['var'].append(np.sqrt(np.var(s_omi)))

a = [np.mean(selection_list['OD_s0']['means']),np.mean(selection_list['OD_s_vac']['means']),np.mean(selection_list['OD_s_boost']['means']),np.mean(selection_list['OD_s_delta']['means']),np.mean(selection_list['OD_s_omi']['means'])]
b = [np.mean(selection_list['OD_s0']['var']),np.mean(selection_list['OD_s_vac']['var']),np.mean(selection_list['OD_s_boost']['var']),np.mean(selection_list['OD_s_delta']['var']),np.mean(selection_list['OD_s_omi']['var'])]
print("s_0_od:")
print(a[0], np.sqrt(np.var(selection_list['OD_s0']['means'])))
print("svac,sboost, sdelta, s_omi")
print(a[1:])


df_da.to_csv("df_da_result.txt",'\t',index=False)
df_od.to_csv("df_od_result.txt",'\t',index=False)

file = open("result_da.txt",'w')
file.write(str(LR.coef_[0] * dC_vac_da) + " " + str(LR.coef_[0] * dC_alpha_da * kappa_da) + " " + str(LR.coef_[0] * dC_delta_da * kappa_da) + "\n")
file.close()

file = open("result_od.txt",'w')
file.write(str(LR.coef_[1] * dC_vac_od) + " " + str(LR.coef_[1] * dC_boost_od) + " " + str(LR.coef_[2] * dC_delta_od) + " " + str(LR.coef_[2] * dC_omi_od))
file.close()

LL_da = ss.norm.logpdf(np.array(df_da['s_model']),np.array(df_da['s_hat']),np.sqrt(np.array(df_da.s_var)))
LL_od = ss.norm.logpdf(np.array(df_od['s_model']),np.array(df_od['s_hat']),np.sqrt(np.array(df_od.s_var)))
df_da['L'] = LL_da
df_od['L'] = LL_od

df_da.to_csv("df_da_result.txt",'\t',index=False)
df_od.to_csv("df_od_result.txt",'\t',index=False)