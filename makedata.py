import datetime
import getpass
import glob
import os
import re
from pathlib import Path, PureWindowsPath, WindowsPath

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms

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

print('--', 'Burundi r2f', '--')
print(frames['Burundi_r2f']['cctrust_localcso'].value_counts(dropna=False))


#-- Burundi r2f --
# 1. All the time         20
# 2. Most of the time     44
# 3. Not very often      252
# 4. Never               568
# 88. Don't know         104
# Name: cctrust_localcso, dtype: int64

#   --- Burundi R2F---
# 1. All the time         20    0.0226
# 2. Most of the time     44    0.0452   
# 3. Not very often      252    0.2851
# 4. Never               568    0.6418    
#                       ----
#                        884    1
# Don't know             104


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
totals_d['labels'] = totals_d.index.map(newlabels)

#export data
totals_m.to_hdf(data_path/'trust_avg_by_inst_cntry.h5', key='totals_m')
totals_d.to_hdf(data_path/"trust_inst_cntry_perc.h5", key='totals_d')
data.to_hdf(data_path/"trust_data.h5", key='trust_data_individ')

