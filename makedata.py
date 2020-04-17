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
    if cntry not in  ['Myanmar_r2f','Cambodia_r2f']:    # mya has endline data only
        ids = df[df['period'].cat.codes == 0].index
        df.drop(ids, inplace=True)      
   # rename Burundi federal government.
    if cntry == 'Burundi_r2f':
        df = df.rename(columns={'cctrust_federalgov': 'cctrust_centralgov'})
    # rename cultural leaders in Uganda drop empty cols for tradrelileaders
    if 'Uganda' in cntry:
        df=df.rename(columns={'cctrust_cultleader':'cctrust_tradleader'})
        df=df.drop(columns='cctrust_tradrelileader')
        
    frames[cntry]=df

#cambodia gets \t after each item remove that: 
for c in dataoverview['Cambodia_r2f']: 
    print(frames['Cambodia_r2f'][c].value_counts(dropna=False))
    frames['Cambodia_r2f'][c]=frames['Cambodia_r2f'][c].str.replace("\t", "")
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

for cntry in cntrs:
    if all(frames[cntry]['cctrust_tradrelileader']=='not in dataset'): 
        frames[cntry]['cctrust_tradrelileader_m']=frames[cntry][['cctrust_tradleader','cctrust_relileader']].mean(axis=1)
        print('---',cntry,'---')
        print(frames[cntry]['cctrust_tradrelileader_m'].value_counts(dropna=False))


# make frames with dummies most of the time/all the time=1-->frames_d
trustitems_a=[ c for c in trustitems if 'cctrust_tradrelileader' not in c]

dums=[i + '_d' for i in trustitems_a]

frames_d = {}
for cntry in cntrs:
    df=frames[cntry]
    for item, d in zip(trustitems_a,dums):
        df[d]=df[item].map({1: 0, 1.5: 0, 2:0, 2.5:0, 3: 1, 3.5:1, 4: 1, np.nan:np.nan})
    frames_d[cntry] = df.loc[:,dums]

# for d in dums:
#     for cntry in cntrs:
#         print('--', cntry,'--')
#         print(frames_d[cntry][d].value_counts(dropna=False))

for cntry in cntrs:
    if 'cctrust_tradrelileader' in frames[cntry].columns:
        frames_d[cntry]['cctrust_tradrelileader_d']=frames[cntry]['cctrust_tradrelileader'].map({1: 0, 1.5: 0, 2:0, 2.5:0, 3: 1, 3.5:1, 4: 1, np.nan:np.nan})
    if 'cctrust_tradrelileader_m' in frames[cntry].columns:
        frames_d[cntry]['cctrust_tradrelileader_d']=frames[cntry]['cctrust_tradrelileader_m'].map({1: 0, 1.5: 0, 2:0, 2.5:0, 3: 1, 3.5:1, 4: 1, np.nan:np.nan})
        #add mean to tradrelileader in case this does not occur
        frames[cntry]['cctrust_tradrelileader']=frames[cntry]['cctrust_tradrelileader_m']
        # set media to non in opt
    if cntry=='OPT_f4d':
        frames[cntry]['cctrust_media']=np.nan
    

for cntry in cntrs:
    print('--', cntry,'--')
    print(frames_d[cntry]['cctrust_tradrelileader_d'].value_counts(dropna=False))
    print(frames[cntry]['cctrust_tradleader'].value_counts(dropna=False))
    print(frames[cntry]['cctrust_relileader'].value_counts(dropna=False))
    print(frames[cntry]['cctrust_tradrelileader'].value_counts(dropna=False))
    print(frames[cntry]['cctrust_media'].value_counts(dropna=False))



vizitems= ['cctrust_localgov',
    'cctrust_centralgov',
    'cctrust_internngo',
    'cctrust_localcso',
    'cctrust_tradrelileader',
    'cctrust_media']

#all items are still category in  should be float

vizitems_d=[c + '_d' for c in vizitems]
#make set with totals_d = trust all or most of the time
totals_d=pd.DataFrame(columns=vizitems_d)
for cntry in cntrs:
    totrow_d=pd.Series(frames_d[cntry].mean(), name=cntry)
    totals_d=totals_d.append(totrow_d)

#make set with totals averages
# replace cctrust_tradrelileader with cctrust_tradrelileader_m if not in dataset. 

vizframes={}
#make a set with all items to visualise -->data
for cntry in cntrs:
    vizframes[cntry]=frames[cntry][vizitems].astype('float64')
    vizframes[cntry]['country']=cntry
    print(vizframes[cntry])
data=pd.concat(vizframes.values(), ignore_index=True)

#make a set with totals and 95 cis

totals_m=data.groupby('country').agg(['mean', 'count', 'sem', lambda lb: sms.DescrStatsW(lb.dropna()).tconfint_mean(alpha=0.05)[0], lambda ub:sms.DescrStatsW(ub.dropna()).tconfint_mean(alpha=0.05)[1]])
totals_m.columns=totals_m.columns.set_levels(['mean', 'count', 'sem', 'lowerbound', 'upperbound'], level=1)


#sortorder items/overall totals (by mean)
totals_overall=data.agg(['mean', 'count', 'sem']).T.sort_values(by='mean', ascending=False)

#labels for countries
newlabels=dict(zip(list(totals_m.index),[nlab.replace('_', '\n(')+')' for nlab in totals_m.index]))
totals_m['labels']=totals_m.index.map(newlabels)
#labels for items ####################TILL HERE
# itemtitles=dict(zip(
#     sorteditems,
#     ["traditional\nleaders",
#     "religious\nleaders",
#     "comm. \nvolunteers",
#     "local cso's",
#     "international\nngo's",
#     "local gov.",
#     "central gov.",
#     "traditional\n&\nrelig. leaders",
#     "media",
#     "big\ncompanies"]))

#make a list which values to plot and where there's no data. 
#notna the df, take values, ravel these values, then enumerate so we have indices corresponding with plot locs in plotgrid 
to_plot=totals.notna().values.ravel()
#list of items
itemlist1=[[i] * 7 for i in sorteditems]
#flatten
itemlist=[item for sorteditems in itemlist1 for item in sorteditems]
#list of cntrys

cntrlist=sortedcntry*nrows

#anno options for no data. 
anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
                 va='center', ha='center')


fig, ax=plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,20))
#axs=all axes
axs=fig.axes
#toprow.
toprow=axs[:ncols]
for i, title in enumerate(cntrytitles.values()):
    print(i, title)   
    toprow[i].set_title(title, fontsize='medium', rotation=90, loc='center')
#loop over plot indices, in cntrys in rows
for i, (toplot, item, cntry),  in enumerate(zip(to_plot, itemlist, cntrlist)):
    if toplot==False: 
        axs[i].annotate('No Data', **anno_opts)
    else: 
        axs[i].barh(y=0.5, width=totals.at[cntry,item], height=0.2)
        axs[i].set_xlim([0,1])
        axs[i].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        #axs[i].annotate(s=(item cntry), **anno_opts)
    axs[i].yaxis.set_visible(False)
    axs[i].xaxis.set_visible(False)
#first columns
firstcols=[axs[i] for i in [ i for i in range(0,(nrows*ncols), ncols)]]
for ax, title in zip(firstcols, itemtitles.values()):
    ax.set_ylabel(title, fontsize='medium', rotation=0, labelpad=20)
    ax.yaxis.set_label_coords(-0.5,0.3)
    ax.yaxis.set_visible(True)
    ax.set_yticklabels("")

plt.show()


fig, ax=plt.subplots(nrows=1, ncols=ncols, figsize=(10,10), sharey='row', sharex='col')

plt.show()


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


