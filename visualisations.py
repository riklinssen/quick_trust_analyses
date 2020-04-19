import datetime
import glob
import os
import re
import getpass
from pathlib import Path, PureWindowsPath, WindowsPath
import statsmodels.stats.api as sms
import matplotlib.cm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
from pandas.api.types import CategoricalDtype


# Filepaths
base_path = Path.cwd()
data_path = base_path/"Data"
graphs_path = base_path/"Graphs"
username = getpass.getuser()

# data:
# sourcedata (all cntrs trustitems)
data = pd.read_hdf(data_path/'trust_data.h5')
# averages--> totals_m
totals_m = pd.read_hdf(data_path/'trust_avg_by_inst_cntry.h5')
# percentage often/always trust --> totals_d
totals_d = pd.read_hdf(data_path/"trust_inst_cntry_perc.h5")

# Visualising averages by institutions
vizitems=['cctrust_localgov',
            'cctrust_centralgov',
            'cctrust_internngo',
            'cctrust_localcso',
            'cctrust_tradrelileader',
            'cctrust_media']

# all items are still category in  should be float

vizitems_d = [c + '_d' for c in vizitems]

# sortorder items/overall totals

totals_institutions = data[vizitems].agg(['mean', 'count', 'sem']).T.sort_values(by='mean', ascending=False)
totals_countries = data.groupby('country').mean().mean(axis=1).sort_values(ascending=True)
# labels for countries
newlabels = dict(zip(list(totals_m.index), [nlab.replace(
    '_', '\n(')+')' for nlab in totals_m.index]))
totals_m['labels'] = totals_m.index.map(newlabels)
totals_d['labels'] = totals_d.index.map(newlabels)

# make lists where institution and cntries are sorted to loop over

institutions_l = list(totals_institutions.index)
countries_l = list(totals_countries.index)
# sort frame
sortordermap = {l: i for i, l in enumerate(countries_l)}
totals_m['countrysort'] = totals_m.index.map(sortordermap)
# sort rows
totals_m.sort_values(by='countrysort', inplace=True)
# sort cols done by institutions list
nrows = 1
ncols = len(institutions_l)
idx = pd.IndexSlice

# labels for institutions
itemtitles = ['Traditional\n&\nreligious\nleaders', 'Local\ngovernment',
              "Local\ncso's", "Int.\nngo's", 'Central\ngovernment', 'Media']

coltitles = {k: title for (k, title) in zip(institutions_l, itemtitles)}
# set informative rowlabels (countries)
totals_m.set_index(totals_m['labels'], inplace=True)


# Oxfam colors
hex_values = ['#E70052',  # rood
              '#F16E22',  # oranje
              '#E43989',  # roze
              '#630235',  # Bordeax
              '#61A534',  # oxgroen
              '#53297D',  # paars
              '#0B9CDA',  # blauw
              '#0C884A'  # donkergroen
              ]
colornames= hex_values[1: -1]

# plot: Institutions (average)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                       sharey = 'row', figsize = (6, 5.5))
axs=fig.axes
# averages
for i, (inst, color) in enumerate(zip(institutions_l, colornames)):
    means=totals_m.loc[:, idx[inst, 'mean']]
    axs[i].scatter(y = means.index, x = means,  color = color, clip_on = False)
# spines
    axs[i].spines['left'].set_position(('outward', 1))
    axs[i].spines['bottom'].set_position(('outward', 5))
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
# titles
    axs[i].set_title(coltitles[inst], size = 9, color = color)
# x-axis
    axs[i].set_xlim(1, 4)
    axs[i].set_xticks([i+1 for i in range(4)])
    xlabels=['never-1', '2', '3', 'always-4']
    axs[i].set_xticklabels(xlabels, size = 'small', rotation = 80)
# y-axis
    axs[i].tick_params(axis = 'y', which = 'both', length = 0)
# grid
    axs[i].yaxis.grid(linestyle = 'dotted', color = 'gray')
# yticklabels
for tick in axs[0].yaxis.get_major_ticks():
    tick.label.set_fontsize(9)
    tick.label.set_fontweight('bold')

# titles
fig.suptitle("Trust in institutions, by project", y = 1.1,fontsize = 'x-large', horizontalalignment = 'right')

plt.figtext(s = "Question: How often do you trust [institution] to do the right thing?\n(1-never-2-Not very often-3-most of the time-4-always)\nData collected Oct-2010, Feb. 2020\nDots represent average by country-project and institution",
            size = 'medium', color = 'gray', x = 0, y=0.95, horizontalalignment='left')
# footnote
plt.figtext(s = "Source: Oxfam Novib: SP-surveys on citizens' voice, n=6413\nTrust in traditional and religious leaders was seperated for all countries except Uganda.\nValues represent the mean of trust in traditional and trust in religious leaders.\nNo data available for OPT-trust in media",
            size = 'small', x = 0, y = 0, color='grey')
plt.subplots_adjust(bottom = 0.3, top = 0.8)
fig.savefig(graphs_path/'Trust_averages.svg')



# plot by cntry percentages
# transpose totals_d for easy sorting &labeling

totals_perc=totals_d[vizitems_d].T
#labels replace newline chars
instlabels={k + '_d': v.replace('\n', " ") for k,v in coltitles.items()}
totals_perc['instlabels']=totals_perc.index.map(instlabels)
nrows=2
ncols=4
fig, ax=plt.subplots(nrows = nrows, ncols = ncols,
                       sharey = 'row', figsize = (6, 6*1.61))
axs=fig.axes
# averages
for i, (cnt, color) in enumerate(zip(countries_l, colornames)):
    means=totals_perc.loc[cnt]
    axs[i].scatter(y = means.index, x = means,  color = color, clip_on=False)
