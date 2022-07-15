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
df_s_o2 = pd.read_csv("output/s_hat_ba2_ba1.txt",'\t',index_col=False)

df_da = pd.read_csv("output/df_da_result.txt",'\t',index_col=False)
df_s_da = pd.read_csv("output/s_hat_delta_alpha.txt",sep='\t',index_col=False)
C_da = pd.read_csv("output/Pop_C_DA.txt",'\t',index_col=False)

df_od = pd.read_csv("output/df_od_result.txt",'\t',index_col=False)
df_s_od = pd.read_csv("output/s_hat_omi_delta.txt",sep='\t',index_col=False)
C_od = pd.read_csv("output/Pop_C_OD.txt",'\t',index_col=False)

df_s_da = pd.read_csv("output/s_hat_delta_alpha.txt",sep='\t',index_col=False)
df_s_od = pd.read_csv("output/s_hat_omi_delta.txt",sep='\t',index_col=False)



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


plt.figure(figsize=(14,10))

fs=12
lw=1
ls=10
countries_awt = sorted(list(set(df_s_awt.country)))
means = []
slopes = []
times = []
points = []
t_length = []
ax = plt.subplot(221)
for c in countries_awt:
	df_c = df_s_awt.loc[list(df_s_awt.country == c)]
	t_range = np.array(df_c.FLAI_time)
	plt.plot(t_range - np.mean(t_range), np.array(df_c.s_hat),linestyle=country2style[c],marker='',color=country2color[c],alpha=0.6,markersize=ms)
	R = ss.linregress(t_range - np.mean(t_range), np.array(df_c.s_hat))
	means.append(np.mean(df_c.s_hat))
	slopes.append(R.slope)
	times += list(t_range - np.mean(t_range))
	points += list(np.array(df_c.s_hat))
	t_length.append(t_range[-1] - t_range[0])

plt.plot(0,np.mean(means),'k_',markersize=ms+6,linewidth=lw+2)
# plt.errorbar([0],[np.mean(means)],fmt='none',yerr=[np.sqrt(np.var(means))],capsize=0,marker='',color='k',linewidth=lw,elinewidth=lw+1)

x_range= np.linspace(-1/2. * np.mean(t_length),1/2. * np.mean(t_length))
plt.plot(x_range, np.mean(slopes) * x_range + np.mean(means),'k-',linewidth=lw+1)
R = ss.linregress(times, points)
var = x_range[-1] * R.slope - x_range[0] * R.slope
plt.title("increase: " + str(np.round(var,3)) + " Pval: " + format(R.pvalue,'.2E'))
plt.xlabel("Time from midpoint, t [days]",fontsize=fs)
plt.ylabel("Selection coefficient, $\\hat{s}_{\\alpha 614{\\rm G}}(t)$",fontsize=fs)
plt.ylim((0.025,0.2))
plt.xlim((-40,40))
ax.tick_params(direction="in",width=0.3,pad=lp,labelsize=ls)


ax = plt.subplot(222)
means = []
slopes = []
times = []
points = []
t_length = []
for c in countries_da:
	df_c = df_da.loc[list(df_da.Country == c)]
	t_range = np.array(df_c.time)
	plt.plot(t_range - np.mean(t_range), df_c.s_hat,linestyle=country2style[c],marker='',color=country2color[c],alpha=0.6,markersize=ms,linewidth=lw)
	R = ss.linregress(t_range - np.mean(t_range), np.array(df_c.s_hat))
	means.append(np.mean(df_c.s_hat))
	slopes.append(R.slope)
	times += list(t_range - np.mean(t_range))
	points += list(np.array(df_c.s_hat))
	t_length.append(t_range[-1] - t_range[0])

# plt.plot(0,np.mean(means),'k_',markersize=ms+6,linewidth=lw+2)
# plt.errorbar([0],[np.mean(means)],fmt='none',yerr=[np.sqrt(np.var(means))],capsize=0,marker='',color='k',linewidth=lw,elinewidth=lw+1)
x_range= np.linspace(-1/2. * np.mean(t_length),1/2. * np.mean(t_length))
plt.plot(x_range, np.mean(slopes) * x_range + np.mean(means),'k-',linewidth=lw+1)
R = ss.linregress(times, points)
var = x_range[-1] * R.slope - x_range[0] * R.slope
plt.title("variaton: " + str(np.round(var,3)) + " Pval: " + format(R.pvalue,'.2E'))
plt.xlabel("Time from midpoint, $t$ [days]",fontsize=fs)
plt.ylabel("Selection coefficient, $\\hat{s}(t)$",fontsize=fs)
plt.ylim((0.025,0.2))
plt.xlim((-40,40))
ax.tick_params(direction="in",width=0.5,pad=lp,labelsize=ls)

ax = plt.subplot(223)
means1 = []
slopes1 = []
times = []
points = []
t_length = []
for c in countries_od:
	df_c = df_od.loc[list(df_od.Country == c)]
	t_range = np.array(df_c.time)
	plt.plot(t_range - np.mean(t_range), df_c.s_hat,linestyle=country2style[c],marker='',color=country2color[c],alpha=0.6,markersize=ms,linewidth=lw)
	R = ss.linregress(t_range - np.mean(t_range), np.array(df_c.s_hat))
	means1.append(np.mean(df_c.s_hat))
	slopes1.append(R.slope)
	times += list(t_range - np.mean(t_range))
	points += list(-np.array(df_c.s_hat))
	t_length.append(t_range[-1] - t_range[0])

# plt.plot(0,np.mean(means),'k_',markersize=ms+6,linewidth=lw+2)
# plt.errorbar([0],[np.mean(means1)],fmt='none',yerr=[np.sqrt(np.var(means1))],capsize=0,marker='',color='k',linewidth=lw,elinewidth=lw+1)
x_range= np.linspace(-1/2. * np.mean(t_length),1/2. * np.mean(t_length))
R = ss.linregress(times, points)
var = x_range[-1] * R.slope - x_range[0] * R.slope
plt.title("variation: " + str(np.round(var,3)) + " Pval: " + format(R.pvalue,'.2E'))
plt.plot(x_range, np.mean(slopes1) * x_range + np.mean(means1),'k-',linewidth=lw+1)
plt.xlabel("Time from midpoint, $t$ [days]",fontsize=fs)
plt.ylabel("Selection coefficient, $\\hat{s}(t)$",fontsize=fs)
plt.ylim((0.025,0.2))
plt.xlim((-40,40))
ax.tick_params(direction="in",width=0.5,pad=lp,labelsize=ls)

countries_a2 = sorted(list(set(df_s_o2.country)))
means = []
slopes = []
times = []
points = []
t_length = []
ax = plt.subplot(224)
for c in countries_a2:
	df_c = df_s_o2.loc[list(df_s_o2.country == c)]
	t_range = np.array(df_c.FLAI_time)
	plt.plot(t_range - np.mean(t_range), np.array(df_c.s_hat) ,linestyle=country2style[c],marker='',color=country2color[c],alpha=0.6,markersize=ms)
	R = ss.linregress(t_range - np.mean(t_range), np.array(df_c.s_hat))
	means.append(np.mean(df_c.s_hat))
	slopes.append(R.slope)
	times += list(t_range - np.mean(t_range))
	points += list(np.array(df_c.s_hat))
	t_length.append(t_range[-1] - t_range[0])

plt.plot(0,np.mean(means),'k_',markersize=ms+6,linewidth=lw+2)
# plt.errorbar([0],[np.mean(means)],fmt='none',yerr=[np.sqrt(np.var(means))],capsize=0,marker='',color='k',linewidth=lw,elinewidth=lw+1)
x_range= np.linspace(-1/2. * np.mean(t_length),1/2. * np.mean(t_length))
plt.plot(x_range, np.mean(slopes) * x_range + np.mean(means),'k-',linewidth=lw+1)
R = ss.linregress(times, points)
var = x_range[-1] * R.slope - x_range[0] * R.slope
plt.title("variation: " + str(np.round(var,3)) + " Pval: " + format(R.pvalue,'.2E'))
plt.xlabel("Time from midpoint, t [days]",fontsize=fs)
plt.ylabel("Selection coefficient, $\\hat{s}_{\\alpha 614{\\rm G}}(t)$",fontsize=fs)
plt.ylim((0.025,0.2))
plt.xlim((-40,40))
ax.tick_params(direction="in",width=0.3,pad=lp,labelsize=ls)


legend_elements = []
for c in sorted(all_countries):
	label = c
	if len(label) == 2:
		label = states_short2state[c]
	else:
		label = c[0] + c[1:].lower()
	legend_elements.append(Line2D([],[],marker='',color=country2color[c],linestyle=country2style[c],label=label,markersize=4, linewidth=lw))
plt.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1.15,1.3),prop={'size':ls})
plt.subplots_adjust(right=0.8)
plt.savefig("figures/FigS3.pdf")
plt.close()