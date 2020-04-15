import datetime
import glob
import os
import re
import getpass
from pathlib import Path, PureWindowsPath, WindowsPath

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
    if cntry != 'Myanmar_r2f':  # mya has endline data only
        ids = df[df['period'].cat.codes == 0].index
        df.drop(ids, inplace=True)
   # rename Burundi federal government.
    if cntry == 'Burundi_r2f':
        df = df.rename(columns={'cctrust_federalgov': 'cctrust_centralgov'})

    frames[cntry] = df

trustitems = [
    'cctrust_localgov',
    'cctrust_centralgov',
    'cctrust_taxauth',
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
            print('-------', 'ORIGINAL', '-------')
            print('-------', cntry, '-------')
            print('-------', item, '-------')
            #print('iscat:', frames[cntry][item].dtype.name=='category')
            print(frames[cntry][item].value_counts(dropna=False))
            frames[cntry][item] = frames[cntry][item].map(
                {
                    '1. All the time': 4,
                    '2. Most of the time': 3,
                    '3. Not very often': 2,
                    '4. Never': 1,
                    'All the time': 4,
                    'Most of the time': 3,
                    'Not very often': 2,
                    'Never': 1
                })
        except KeyError:
            #print(cntry, item, 'NOT IN DATASET')
            frames[cntry][item] = np.nan

# make frames with dummies most of the time/all the time=1 frames_d
dums=[i + '_d' for i in trustitems]

frames_d = {}

for cntry in cntrs:
    df=frames[cntry]
    for item, d in zip(trustitems,dums):
        df[d]=df[item].map({1: 0, 2:0, 3: 1, 4: 1, np.nan:np.nan})
    frames_d[cntry] = df.loc[:,dums]

totals=pd.DataFrame(columns=dums)
for cntry in cntrs:
    totrow=pd.Series(frames_d[cntry].mean(), name=cntry)
    totals=totals.append(totrow)

#sortorder
#check which insitutes have highest trust on average
sorteddf=pd.DataFrame(totals.mean(), columns=['totalavg'])
orderitems=list(sorteddf.sort_values(by='totalavg', ascending=False).index)


fig, ax=plt.subplots(nrows=len(cntrs),ncols=len(orderitems), sharey='row', sharex='col')

fig.show()
 

# for cntry in cntrs:
#     for item in dums:
#         print('-------', 'dummies: most of the time, all the time', '-------')
#         print(frames_d[cntry][item].value_counts(dropna=False))


# print(frames[cntry][item].value_counts(dropna=False))

# # print(frames[cntry][item].astype('str').value_counts(dropna=False))


# # lets do as strings, remove digits then just recategorize to be sure.
# cat_type = CategoricalDtype(categories=[
#                             "Never", "Not very often", "Most of the time", "All the time"], ordered=True)

# for cntry in cntrs:
#     # period in first col
#     for item in trustitems:
#         # removing digits
#         try:
#             frames[cntry][item] = frames[cntry][item].astype(
#                 'str').str.replace('\d+', '')

#         except KeyError:
#             print(cntry, item, 'NOT IN DATASET')
#             frames[cntry][item] = np.nan

#     print(frames[cntry][item].value_counts(dropna=False))

# # Uganda_f4d   -> no values in categories
# # Uganda_r2f

# # not in data Burundi_r2f cctrust_tradrelileader
# # Myanmar_r2f:cctrust_tradrelileader:OK
# #
# # not in data Niger_cf cctrust_tradrelileader
# # not in data Niger_r2f cctrust_tradrelileader
# # not in data Burundi_r2f cctrust_commvolun


# trustitems = ['cctrust_localgov',
#               'cctrust_centralgov',
#               'cctrust_taxauth',
#               'cctrust_bigcomp',
#               'cctrust_internngo',
#               'cctrust_localcso',
#               'cctrust_tradrelileader',
#               'cctrust_commvolun']


