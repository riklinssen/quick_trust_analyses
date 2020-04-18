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


# test run on first files
base_path = Path.cwd()
data_path = base_path/"Data"
graphs_path = base_path/"Graphs"
username = getpass.getuser()

files = pd.read_csv(data_path/"links.csv", sep=';', header=0)

filelist = [f.replace('RikL', username) for f in files.link_to_file]
files['name'] = files.cntry + '_'+files.toc


# check which cols are in each set.
cntrs = []
trustvars = []
for f, cntry in zip(filelist, files.name):
    checkcols = pd.read_stata(Path(f))
    cntrs.append(cntry)
    cols = [c for c in checkcols.columns if 'cctrust' in c or 'period' in c]
    cols2 = [c for c in cols if 'out_' not in c]
    trustvars.append(cols2)
dataoverview = dict(zip(cntrs, trustvars))

# make a dict of frames
frames = {}

for f, cntry in zip(filelist, files.name):
    df = pd.read_stata(Path(f), columns=list(dataoverview[cntry]))
    # keep endline only
    if cntry not in ['Myanmar_r2f', 'Cambodia_r2f']:    # mya has endline data only
        ids = df[df['period'].cat.codes == 0].index
        df.drop(ids, inplace=True)
   # rename Burundi federal government.
    if cntry == 'Burundi_r2f':
        df = df.rename(columns={'cctrust_federalgov': 'cctrust_centralgov'})
    # rename cultural leaders in Uganda drop empty cols for tradrelileaders
    if 'Uganda' in cntry:
        df = df.rename(columns={'cctrust_cultleader': 'cctrust_tradleader'})
        df = df.drop(columns='cctrust_tradrelileader')

    frames[cntry] = df

# cambodia gets \t after each item remove that:
for c in dataoverview['Cambodia_r2f']:
    print(frames['Cambodia_r2f'][c].value_counts(dropna=False))
    frames['Cambodia_r2f'][c] = frames['Cambodia_r2f'][c].str.replace("\t", "")
    print(frames['Cambodia_r2f'][c].value_counts(dropna=False))


trustitems = [
    'cctrust_localgov',
    'cctrust_centralgov',
    'cctrust_bigcomp',
    'cctrust_internngo',
    'cctrust_localcso',
    'cctrust_tradrelileader',
    'cctrust_tradleader',
    'cctrust_relileader',
    'cctrust_commvolun',
    'cctrust_media']

for cntry in cntrs:
    for item in trustitems:
        try:
            #print('iscat:', frames[cntry][item].dtype.name=='category')
            frames[cntry][item] = frames[cntry][item].map(
                {
                    '1. All the time': 4,
                    '2. Most of the time': 3,
                    '3. Not very often': 2,
                    '4. Never': 1,
                    'All the time': 4,
                    'Most of the time': 3,
                    'Not very often': 2,
                    'Never': 1,
                })
        except KeyError:
            #print(cntry, item, 'NOT IN DATASET')
            frames[cntry][item] = 'not in dataset'

for item in trustitems:
    for cntry in cntrs:
        print('-------', item, '-------')
        print('-------', cntry, '-------')
        print(frames[cntry][item].value_counts(dropna=False))


# combine traditional and religious leaders in single col.

# make frames with dummies most of the time/all the time=1-->frames_d
trustitems_a = [c for c in trustitems if 'cctrust_tradrelileader' not in c]

dums = [i + '_d' for i in trustitems_a]

frames_d = {}
for cntry in cntrs:
    if all(frames[cntry]['cctrust_tradrelileader'] == 'not in dataset'):
        frames[cntry]['cctrust_tradrelileader_m'] = frames[cntry][[
            'cctrust_tradleader', 'cctrust_relileader']].mean(axis=1)
        print('---', cntry, '---')
        print(frames[cntry]
              ['cctrust_tradrelileader_m'].value_counts(dropna=False))


# make frames with dummies most of the time/all the time=1-->frames_d
trustitems_a = [c for c in trustitems if 'cctrust_tradrelileader' not in c]

dums = [i + '_d' for i in trustitems_a]

frames_d = {}
for cntry in cntrs:
    df = frames[cntry]
    for item, d in zip(trustitems_a, dums):
        df[d] = df[item].map(
            {1: 0, 1.5: 0, 2: 0, 2.5: 0, 3: 1, 3.5: 1, 4: 1, np.nan: np.nan})
    frames_d[cntry] = df.loc[:, dums]


for cntry in cntrs:
    if 'cctrust_tradrelileader' in frames[cntry].columns:
        frames_d[cntry]['cctrust_tradrelileader_d'] = frames[cntry]['cctrust_tradrelileader'].map(
            {1: 0, 1.5: 0, 2: 0, 2.5: 0, 3: 1, 3.5: 1, 4: 1, np.nan: np.nan})
    if 'cctrust_tradrelileader_m' in frames[cntry].columns:
        frames_d[cntry]['cctrust_tradrelileader_d'] = frames[cntry]['cctrust_tradrelileader_m'].map(
            {1: 0, 1.5: 0, 2: 0, 2.5: 0, 3: 1, 3.5: 1, 4: 1, np.nan: np.nan})
        # add mean to tradrelileader in case this does not occur
        frames[cntry]['cctrust_tradrelileader'] = frames[cntry]['cctrust_tradrelileader_m']
        # set media to non in opt
    if cntry == 'OPT_f4d':
        frames[cntry]['cctrust_media'] = np.nan


for cntry in cntrs:
    print('--', cntry, '--')
    print(frames_d[cntry]
          ['cctrust_tradrelileader_d'].value_counts(dropna=False))
    print(frames[cntry]['cctrust_tradleader'].value_counts(dropna=False))
    print(frames[cntry]['cctrust_relileader'].value_counts(dropna=False))
    print(frames[cntry]['cctrust_tradrelileader'].value_counts(dropna=False))
    print(frames[cntry]['cctrust_media'].value_counts(dropna=False))

vizitems = ['cctrust_localgov',
            'cctrust_centralgov',
            'cctrust_internngo',
            'cctrust_localcso',
            'cctrust_tradrelileader',
            'cctrust_media']

# all items are still category in  should be float

vizitems_d = [c + '_d' for c in vizitems]
# make set with totals_d = trust all or most of the time
totals_d = pd.DataFrame(columns=vizitems_d)
for cntry in cntrs:
    totrow_d = pd.Series(frames_d[cntry].mean(), name=cntry)
    totals_d = totals_d.append(totrow_d)

# make set with totals averages
# replace cctrust_tradrelileader with cctrust_tradrelileader_m if not in dataset.

vizframes = {}
# make a set with all items to visualise -->data
for cntry in cntrs:
    vizframes[cntry] = frames[cntry][vizitems].astype('float64')
    vizframes[cntry]['country'] = cntry
    print(vizframes[cntry])
data = pd.concat(vizframes.values(), ignore_index=True)

# make a set with totals and 95 cis

totals_m = data.groupby('country').agg(['mean', 'count', 'sem', lambda lb: sms.DescrStatsW(lb.dropna(
)).tconfint_mean(alpha=0.05)[0], lambda ub:sms.DescrStatsW(ub.dropna()).tconfint_mean(alpha=0.05)[1]])
# add a better column label
totals_m.columns = totals_m.columns.set_levels(
    ['mean', 'count', 'sem', 'lowerbound', 'upperbound'], level=1)

# sortorder items/overall totals (by mean)
totals_institutions = data[vizitems].agg(
    ['mean', 'count', 'sem']).T.sort_values(by='mean', ascending=False)
totals_countries = data.groupby('country').mean().mean(
    axis=1).sort_values(ascending=True)
# labels for countries
newlabels = dict(zip(list(totals_m.index), [nlab.replace(
    '_', '\n(')+')' for nlab in totals_m.index]))
totals_m['labels'] = totals_m.index.map(newlabels)
#totals_d['labels'] = totals_d.index.map(newlabels)


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
              "Local\ncso's", "Int.\nngo's", 'Central\n government', 'Media']

coltitles = {k: title for (k, title) in zip(institutions_l, itemtitles)}
# set informative rowlabels (countries)
totals_m.set_index(totals_m['labels'], inplace=True)

#export data
totals_m.to_csv(data_path/'trust_avg_by_inst_cntry.csv')
# missings to max value of scale solve later

#totals_m.fillna(4, inplace=True)
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
colornames = hex_values[1:-1]

####################plot: Institutions (average)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                       sharey='row', figsize=(6, 5.5))
axs = fig.axes
# averages
for i, (inst, color) in enumerate(zip(institutions_l, colornames)):
    means = totals_m.loc[:, idx[inst, 'mean']]
    axs[i].scatter(y=means.index, x=means,  color=color, clip_on=False)
# spines
    axs[i].spines['left'].set_position(('outward', 1))
    axs[i].spines['bottom'].set_position(('outward', 5))
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
# titles
    axs[i].set_title(coltitles[inst], size=9, color=color)
# x-axis
    axs[i].set_xlim(1, 4)
    axs[i].set_xticks([i+1 for i in range(4)])
    xlabels = ['never-1', '2', '3', 'always-4']
    axs[i].set_xticklabels(xlabels, size='small', rotation=80)
# y-axis
    axs[i].tick_params(axis='y', which='both', length=0)
# grid
    axs[i].yaxis.grid(linestyle='dotted', color='gray')
#yticklabels
for tick in axs[0].yaxis.get_major_ticks():
    tick.label.set_fontsize(9)
    tick.label.set_fontweight('bold') 

# titles
fig.suptitle("Trust in institutions, by project", y=1.1,
             fontsize='x-large', horizontalalignment='right')

plt.figtext(s="Question: How often do you trust [institution] to do the right thing?\n(1-never-2-Not very often-3-most of the time-4-always)\nData collected Oct-2010, Feb. 2020\nDots represent average by country-project and institution",
            size='medium', color='gray', x=0, y=0.95, horizontalalignment='left')
# footnote
plt.figtext(s="Source: Oxfam Novib: SP-surveys on citizens' voice, n=6413\nNo data available for OPT-trust in media",
            size='small', x=0, y=0.1, color='grey')
plt.subplots_adjust(bottom=0.3, top=0.8)
fig.savefig(graphs_path/'Trust_averages.svg')



#####plot by cntry
#transpose totals_d for easy sorting &labeling
totals_d
totals_perc=totals_d[vizitems_d]

nrows=2
ncols=4
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                       sharey='row', figsize=(6, 6*1.61))
axs = fig.axes
# averages
for i, (cnt, color) in enumerate(zip(countries_l, colornames)):
    means = totals_d.loc[cnt]
    axs[i].scatter(y=means.index, x=means,  color=color, clip_on=False)

