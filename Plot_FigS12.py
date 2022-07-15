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

month_dict = {1:'Jan.',2:'Feb.',3:'Mar.',4:'Apr.',5:'May',6:'Jun.',7:'Jul.',8:'Aug.',9:'Sept.',10:'Oct.',11:'Nov.',12:'Dec.'}


df_da = pd.read_csv("output/df_da_result.txt",'\t',index_col=False)
df_od = pd.read_csv("output/df_od_result.txt",'\t',index_col=False)
countries_da = sorted(list(set(df_da.Country)))
countries_od = sorted(list(set(df_od.Country)))

df_da = pd.read_csv("output/Pop_C_DA.txt",sep='\t',index_col=False)
df_od = pd.read_csv("output/Pop_C_OD.txt",sep='\t',index_col=False)

df_s_da = pd.read_csv("output/s_hat_delta_alpha.txt",sep='\t',index_col=False)
df_s_od = pd.read_csv("output/s_hat_omi_delta.txt",sep='\t',index_col=False)

country_variance = []
for c in countries_da:
    df_c = df_s_da.loc[list(df_s_da.country == c)]
    meansvar = np.mean(df_c.s_var)
    country_variance.append(meansvar)
median_svar_da = np.median(country_variance)

country_variance = []
for c in countries_od:
    df_c = df_s_od.loc[list(df_s_od.country == c)]
    meansvar = np.mean(df_c.s_var)
    country_variance.append(meansvar)
median_svar_od = np.median(country_variance)

df_s_da['s_var'] = np.array(df_s_da['s_var'] + median_svar_da)
df_s_od['s_var'] = np.array(df_s_od['s_var'] + median_svar_od)

ratio = 0.5
fs=12
lp = 10
ls = 10
rot=20
lw = 1

gamma_vac = 1.15
gamma_inf = 2.3


plt.figure(figsize= (21, len(countries_da) * 3))
index = 1
for c in countries_da:
	df_c = df_da.loc[list(df_da.country == c)]
	df_c_s = df_s_da.loc[list(df_s_da.country==c)]
	t_range = list(df_c.time)

	ax = plt.subplot(len(countries_da), 3, index)
	plt.plot(t_range, df_c.freq_alpha, color = color_list[1],linewidth=lw)
	plt.plot(t_range, df_c.freq_delta, color = color_list[2],linewidth=lw)
	plt.plot(t_range, df_c.freq_other, color = color_list[0],linewidth=lw)
	plt.fill_between(t_range, np.array(df_c.freq_alpha) - np.sqrt(np.array(df_c.var_a)),np.array(df_c.freq_alpha) + np.sqrt(np.array(df_c.var_a)),color=color_list[1],alpha=0.4)
	plt.fill_between(t_range, np.array(df_c.freq_delta) - np.sqrt(np.array(df_c.var_d)),np.array(df_c.freq_delta) + np.sqrt(np.array(df_c.var_d)),color=color_list[2],alpha=0.4)
	plt.fill_between(t_range, np.array(df_c.freq_other) - np.sqrt(np.array(df_c.var_other)),np.array(df_c.freq_other) + np.sqrt(np.array(df_c.var_other)),color=color_list[0],alpha=0.4)

	plt.ylabel("Frequency, $x(t)$",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.02,1.02])
	ax.set_yticks([0,0.5,1.0],['0.0','0.5','1.0'],fontsize=ls)
	plt.title(c)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = list(np.arange(t_range[0],t_range[-1],30)) + [t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
	ax.tick_params(direction="in",width=0.3)

	index += 1
	ax = plt.subplot(len(countries_da), 3, index)
	plt.plot(t_range, df_c.R_vac,'-',color='k')
	plt.plot(t_range,df_c.R_alpha,'--',color='k')
	plt.plot(t_range, df_c.Cav, '-',color=color_list[1])
	plt.plot(t_range, df_c.Cdv, '-',color=color_list[2])
	plt.plot(t_range, df_c.Caa,'--',color=color_list[1])
	plt.plot(t_range, df_c.Cda,'--',color=color_list[2])
	plt.fill_between(t_range, df_c.Cdv,df_c.Cav,color=color_list[2],alpha=0.3,linewidth=0.0)
	plt.fill_between(t_range, df_c.Cda,df_c.Caa,color=color_list[2],alpha=0.3,linewidth=0.0)
	legend_elements = []
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle='-',linewidth=1.0,label='Vaccination'))
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle='--',linewidth=1.0,label='Alpha recovery'))
	plt.legend(handles = legend_elements,loc='upper left', prop={'size':ls-1})

	plt.ylabel("Population Immunity",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.02,0.6])
	# ax.set_yticks([0,0.5,1.0],['0.0','0.5','1.0'],fontsize=ls)
	plt.title(c)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = list(np.arange(t_range[0],t_range[-1],30)) + [t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
	ax.tick_params(direction="in",width=0.3)

	index += 1
	ax = plt.subplot(len(countries_da), 3, index)
	times = np.array(df_c_s.FLAI_time) 
	plt.errorbar(times,np.array(df_c_s.s_hat) - np.mean(df_c_s.s_hat), np.sqrt(np.array(df_c_s.s_var)) * 1.96,alpha=0.7,color='k', fmt='o-')

	F_alpha = -gamma_vac * np.array(df_c.Cav) - gamma_inf * np.array(df_c.Caa) - gamma_inf * np.array(df_c.Cad)
	F_delta = -gamma_vac * np.array(df_c.Cdv) - gamma_inf * np.array(df_c.Cda) - gamma_inf * np.array(df_c.Cdd)
	plt.plot(t_range,F_delta - F_alpha - np.mean(F_delta - F_alpha),'r--')
	plt.title(c)
	plt.ylabel("Selection coefficient,$\\Delta s_{\\delta \\alpha}$",fontsize=fs,labelpad=lp)
	# ax.set_ylim([-0.4,0.4])
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = list(np.arange(t_range[0],t_range[-1],30)) + [t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
	ax.tick_params(direction="in",width=0.3)	

	index += 1

plt.subplots_adjust(bottom=0.01,top=0.99,hspace=0.3)

plt.savefig("figures/FigS1.pdf")
plt.close()

ratio = 0.5
ratio2 = 0.25
gamma_vac = 0.29
gamma_inf = 0.58
fig = plt.figure(figsize= (21, len(countries_od) * 3))

gs0 = gridspec.GridSpec(len(countries_od),3,figure=fig,wspace=0.3)
index = 0
for c in countries_od:
	df_c = df_od.loc[list(df_od.country == c)]
	df_c_s = df_s_od.loc[list(df_s_od.country==c)]
	t_range = list(df_c.time)

	# ax = plt.subplot(len(countries_od), 3, index)

	ax = fig.add_subplot(gs0[index,0])

	plt.plot(t_range, df_c.freq_omi, color=color_list[4],linewidth=lw)
	plt.plot(t_range, df_c.freq_delta, color=color_list[2],linewidth=lw)
	plt.fill_between(t_range, np.array(df_c.freq_omi) - np.sqrt(np.array(df_c.var_o)),np.array(df_c.freq_omi) + np.sqrt(np.array(df_c.var_o)),color=color_list[4],alpha=0.4,linewidth=0.0)
	plt.fill_between(t_range, np.array(df_c.freq_delta) - np.sqrt(np.array(df_c.var_d)),np.array(df_c.freq_delta) + np.sqrt(np.array(df_c.var_d)),color=color_list[2],alpha=0.4,linewidth=0.0)
	plt.ylabel("Frequency, $x(t)$",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.02,1.02])
	ax.set_yticks([0,0.5,1.0],['0.0','0.5','1.0'],fontsize=ls)
	plt.title(c)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = list(np.arange(t_range[0],t_range[-1],30)) + [t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
	ax.tick_params(direction="in",width=0.3)


	
	
	
	# ax = fig.add_subplot(gs0[index,1])
	gs00 = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs0[index,1],hspace=0.05)
	ax = fig.add_subplot(gs00[0,0])
	t_range = list(df_c.time)
	plt.plot(t_range, df_c.R_vac,'k-')
	plt.plot(t_range, df_c.R_boost,'k-.')
	plt.plot(t_range, df_c.Cdv,  '-',color = color_list[2],linewidth=lw)
	plt.plot(t_range, df_c.Cov,   '-',color = color_list[4],linewidth=lw)
	plt.plot(t_range, df_c.Cdb, '-.',color = color_list[2],linewidth=lw)
	plt.plot(t_range, df_c.Cob,  '-.',color = color_list[4],linewidth=lw)
	plt.fill_between(t_range, df_c.Cov,df_c.Cdv,color=color_list[4],alpha=0.3,linewidth=0.0)
	plt.fill_between(t_range, df_c.Cob,df_c.Cdb,color=color_list[4],alpha=0.3,linewidth=0.0)
	# plt.ylabel("Population Immunity",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.02,0.6])
	plt.title(c)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = list(np.arange(t_range[0],t_range[-1],30)) + [t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append('')
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio2)
	ax.tick_params(direction="in",width=0.3)
	legend_elements = []
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle='-.',linewidth=1.0,label='Booster'))
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle='-',linewidth=1.0,label='Vaccination'))
	plt.legend(handles = legend_elements,loc='upper left', prop={'size':ls-1})

	ax = fig.add_subplot(gs00[1,0])
	plt.plot(t_range, df_c.R_delta,'k--')
	plt.plot(t_range, df_c.R_omi,'k:')
	plt.plot(t_range, df_c.Cdd, '--',color = color_list[2],linewidth=lw)
	plt.plot(t_range, df_c.Cod,  '--',color = color_list[4],linewidth=lw)
	plt.plot(t_range, df_c.Cdo,  ':',color = color_list[2],linewidth=lw)
	plt.plot(t_range, df_c.Coo,   ':',color = color_list[4],linewidth=lw)
	plt.fill_between(t_range, df_c.Cod,df_c.Cdd,color=color_list[4],alpha=0.3,linewidth=0.0)
	plt.fill_between(t_range, df_c.Cdo,df_c.Coo,color=color_list[2],alpha=0.3,linewidth=0.0)
	plt.ylabel("Population Immunity",fontsize=fs,labelpad=lp)
	ax.set_ylim([-0.02,0.6])
	legend_elements = []
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle='--',linewidth=1.0,label='Delta recovery'))
	legend_elements.append(Line2D([],[],marker='',color='k',linestyle=':',linewidth=1.0,label='Omicron recovery'))
	plt.legend(handles = legend_elements,loc='upper left', prop={'size':ls-1})
	# ax.set_yticks([0,0.5,1.0],['0.0','0.5','1.0'],fontsize=ls)
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = list(np.arange(t_range[0],t_range[-1],30)) + [t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio2)
	ax.tick_params(direction="in",width=0.3)

	
	ax = fig.add_subplot(gs0[index,2])
	times = np.array(df_c_s.FLAI_time) 
	plt.errorbar(times,np.array(df_c_s.s_hat) - np.mean(df_c_s.s_hat), np.sqrt(np.array(df_c_s.s_var)) * 1.96,alpha=0.7,color='k', fmt='o-')

	F_delta = -gamma_vac * np.array(df_c.Cdv) -gamma_vac * np.array(df_c.Cdb) - gamma_inf * np.array(df_c.Cdd) - gamma_inf * np.array(df_c.Cdo)
	F_omi = -gamma_vac * np.array(df_c.Cov) -gamma_vac * np.array(df_c.Cob) - gamma_inf * np.array(df_c.Cod) - gamma_inf * np.array(df_c.Coo)
	plt.plot(t_range,F_omi - F_delta - np.mean(F_omi - F_delta),'r--')
	plt.title(c)
	plt.ylabel("Selection coefficient,$\\Delta s_{o \\delta}$",fontsize=fs,labelpad=lp)
	# ax.set_ylim([-0.4,0.4])
	x_left, x_right = ax.get_xlim()
	y_low, y_high = ax.get_ylim()
	xtick_pos = list(np.arange(t_range[0],t_range[-1],30)) + [t_range[-1]]
	xtick_labels = [Time.coordinateToDate(int(x)) for x in xtick_pos] 
	xtick_labels2 = []
	for d in xtick_labels:
		xtick_labels2.append(month_dict[int(d.month)] + " '" + str(d.day))
	ax.set_xticks(xtick_pos,xtick_labels2,rotation=0,ha='center',fontsize=ls)
	ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
	ax.tick_params(direction="in",width=0.3)	


	index += 1

plt.subplots_adjust(bottom=0.01,top=0.99,hspace=0.3)

plt.savefig("figures/FigS2.pdf")
plt.close()

