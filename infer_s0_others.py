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


countries_da = sorted(list(set(df_da.Country)))
countries_od = sorted(list(set(df_od.Country)))
countries_ba2 = sorted(list(set(countries_da).intersection(set(countries_od))))
countries_ba4 = sorted(list(set(countries_da).intersection(set(countries_od))))

gamma_vac_ad = 1.15
gamma_inf_ad = 2.3
gamma_vac_od = 0.29
gamma_inf_od = 0.58
ms = 5
rot = 20
fs = 12
ls = 10


s0_awt_list = []
s0_ad_list = []
s0_do_list = []
s0_ba2_list = []
s0_ba45_list = []

s_vac_awt_list = []
s_vac_ad_list = []
s_vac_do_list = []

s_bst_do_list  = []
s_bst_o12_list = []
s_bst_o24_list = []

s_alpha_awt_list = []
s_alpha_ad_list = []

s_delta_ad_list = []
s_delta_do_list = []

s_omi_do_list = []
s_o1_o12_list = []
s_o2_o12_list = []
s_o1_o24_list = []
s_o2_o24_list = []

for c in countries:
	df_c = df.loc[list(df.country == c)]
	# df_c = df_c.loc[[int(t) < Time.dateToCoordinate("2022-05-23") and int(t) > Time.dateToCoordinate("2020-12-31") for t in list(df_c.time)]]
	df_c.index = df_c.time

	s_antigenic = []
	gamma_vac = 1.15
	gamma_inf = 2.3
	s_vac = [] 
	s_alpha = []
	s_wt = []
	df_s = df_s_awt.loc[list(df_s_awt.country == c)]
	if len(df_s) != 0:
		for line in df_s.iterrows():
			line = line[1]
			C = df_c.loc[line.FLAI_time]
			C_bar_wt =  - gamma_vac * C.C_wtv - gamma_vac * C.C_wtb -  gamma_inf * C.C_wta  -  gamma_inf * C.C_wtd - gamma_inf * C.C_wto
			C_bar_alpha = - gamma_vac * C.C_av - gamma_vac * C.C_ab -  gamma_inf * C.C_aa  -  gamma_inf * C.C_ad - gamma_inf * C.C_ao
			s_antigenic.append(C_bar_alpha - C_bar_wt)
			s_vac.append(-gamma_vac * (C.C_av  - C.C_wtv))
			s_alpha.append(-gamma_inf * (C.C_aa  - C.C_wta))
		s0_awt = np.mean(np.array(df_s.s_hat) - np.array(s_antigenic))
		s_vac_awt_list.append(np.mean(s_vac))
		s_alpha_awt_list.append(np.mean(s_alpha))
		s0_awt_list.append(s0_awt)

	s_antigenic = []
	s_vac = [] 
	s_alpha = []
	s_delta = []
	gamma_vac = 1.15
	gamma_inf = 2.3
	df_s = df_s_da.loc[list(df_s_da.country == c)]
	if len(df_s) != 0:
		for line in df_s.iterrows():
			line = line[1]
			C = df_c.loc[line.FLAI_time]
			C_bar_delta =  - gamma_vac * C.C_dv - gamma_vac * C.C_db -  gamma_inf * C.C_da  -  gamma_inf * C.C_dd - gamma_inf * C.C_do
			C_bar_alpha = - gamma_vac * C.C_av - gamma_vac * C.C_ab -  gamma_inf * C.C_aa  -  gamma_inf * C.C_ad - gamma_inf * C.C_ao
			s_antigenic.append(C_bar_delta - C_bar_alpha)
			s_vac.append(-gamma_vac * (C.C_dv  - C.C_av))
			s_alpha.append(-gamma_inf * (C.C_da  - C.C_aa))
			s_delta.append(-gamma_inf * (C.C_dd  - C.C_ad))
		s0_ad = np.mean(np.array(df_s.s_hat) - np.array(s_antigenic))
		s0_ad_list.append(s0_ad)
		s_vac_ad_list.append(np.mean(s_vac))
		s_alpha_ad_list.append(np.mean(s_alpha))
		s_delta_ad_list.append(np.mean(s_delta))
	

	s_antigenic = []
	s_vac = [] 
	s_bst = [] 
	s_delta = []
	s_omi = []
	gamma_vac = 0.29
	gamma_inf = 0.58
	df_s = df_s_od.loc[list(df_s_od.country == c)]
	if len(df_s) != 0:
		for line in df_s.iterrows():
			line = line[1]
			C = df_c.loc[line.FLAI_time]
			C_bar_delta =  - gamma_vac * C.C_dv - gamma_vac * C.C_db -  gamma_inf * C.C_da  -  gamma_inf * C.C_dd - gamma_inf * C.C_do
			C_bar_omi = - gamma_vac * C.C_ov - gamma_vac * C.C_ob -  gamma_inf * C.C_oa  -  gamma_inf * C.C_od - gamma_inf * C.C_oo
			s_antigenic.append(C_bar_omi - C_bar_delta)

			s_vac.append(-gamma_vac * (C.C_ov0  - C.C_dv0))
			s_bst.append(gamma_vac * (C.C_ov0  - C.C_dv0) - gamma_vac * (C.C_ov  - C.C_dv) - gamma_vac * (C.C_ob- C.C_db))
			s_omi.append(-gamma_inf * (C.C_oo  - C.C_do))
			s_delta.append(-gamma_inf * (C.C_od  - C.C_dd))

		s0_do = np.mean(np.array(df_s.s_hat) - np.array(s_antigenic))
		s0_do_list.append(s0_do)
		s_vac_do_list.append(np.mean(s_vac))
		s_bst_do_list.append(np.mean(s_bst))
		s_omi_do_list.append(np.mean(s_omi))
		s_delta_do_list.append(np.mean(s_delta))

	s_antigenic = []
	gamma_vac = 0.29
	gamma_inf = 0.58
	s_bst = []
	s_o1 = []
	s_o2 = []
	df_s = df_s_ba2ba1.loc[list(df_s_ba2ba1.country == c)]
	if len(df_s) != 0:
		for line in df_s.iterrows():
			line = line[1]
			C = df_c.loc[line.FLAI_time]
			F_ba1   = - gamma_vac * C.C_ov - gamma_vac * C.C_ob -  gamma_inf * C.C_oa   - gamma_inf * C.C_od - gamma_inf * C.C_a1a1 - gamma_inf * C.C_a1a2
			F_ba2   = - gamma_vac * C.C_ov - gamma_vac * C.C_a2b - gamma_inf * C.C_oa  -  gamma_inf * C.C_od - gamma_inf * C.C_a2a1 - gamma_inf * C.C_a2a2
			s_antigenic.append(F_ba2 - F_ba1)

			s_bst.append(-gamma_vac * (C.C_a2b  - C.C_ob))
			s_o1.append(-gamma_inf * (C.C_a2a1  - C.C_a1a1))
			s_o2.append(-gamma_inf * (C.C_a2a2  - C.C_a1a2))
		s0_ba2 = np.mean(np.array(df_s.s_hat) - np.array(s_antigenic))
		s0_ba2_list.append(s0_ba2)
		s_bst_o12_list.append(np.mean(s_bst))
		s_o1_o12_list.append(np.mean(s_o1))
		s_o2_o12_list.append(np.mean(s_o2))

	s_antigenic = []
	gamma_vac = 0.29
	gamma_inf = 0.58
	s_bst = []
	s_o1 = []
	s_o2 = []
	df_s = df_s_ba45ba2.loc[list(df_s_ba45ba2.country == c)]
	if len(df_s) != 0:
		for line in df_s.iterrows():
			line = line[1]
			C = df_c.loc[line.FLAI_time]
			F_ba2   = - gamma_vac * C.C_ov - gamma_vac * C.C_a2b - gamma_inf * C.C_oa  -  gamma_inf * C.C_od - gamma_inf * C.C_a2a1 - gamma_inf * C.C_a2a2
			F_ba45   =- gamma_vac * C.C_ov - gamma_vac * C.C_a45b -gamma_inf * C.C_oa -   gamma_inf * C.C_od - gamma_inf * C.C_a45a1 - gamma_inf * C.C_a45a2
			s_antigenic.append(F_ba45 - F_ba2)
			s_bst.append(-gamma_vac * (C.C_a45b - C.C_a2b))
			s_o1.append(-gamma_inf * (C.C_a45a1 - C.C_a2a1))
			s_o2.append(-gamma_inf * (C.C_a45a2 - C.C_a2a2))
		s0_ba45 = np.mean(np.array(df_s.s_hat) - np.array(s_antigenic))
		s0_ba45_list.append(s0_ba45)
		s_bst_o24_list.append(np.mean(s_bst))
		s_o1_o24_list.append(np.mean(s_o1))
		s_o2_o24_list.append(np.mean(s_o2))


print("1 - Alpha:")
print("s_vac = ", np.mean(s_vac_awt_list), " s_alpha = ", np.mean(s_alpha_awt_list), " s_0 = ", np.mean(s0_awt_list))
print("Alpha - Delta:")
print("s_vac = ", np.mean(s_vac_ad_list), " s_alpha = ", np.mean(s_alpha_ad_list), " s_delta = ", np.mean(s_delta_ad_list), " s_0 = ", np.mean(s0_ad_list))
print("Delta - Omicron:")
print("s_vac = ", np.mean(s_vac_do_list),"s_bst = ", np.mean(s_bst_do_list), " s_delta = ", np.mean(s_delta_do_list), " s_omi = ", np.mean(s_omi_do_list), " s_0 = ", np.mean(s0_do_list))
print("o1- o2:")
print("s_bst = ", np.mean(s_bst_o12_list), " s_o1 = ", np.mean(s_o1_o12_list), " s_o2 = ", np.mean(s_o2_o12_list), " s_0 = ", np.mean(s0_ba2_list))
print("o2 - o4:")
print("s_bst = ", np.mean(s_bst_o24_list), " s_o1 = ", np.mean(s_o1_o24_list), " s_o2 = ", np.mean(s_o2_o24_list), " s_0 = ", np.mean(s0_ba45_list))

print(np.mean(s0_awt_list), np.mean(s0_ba2_list), np.mean(s0_ba45_list))

