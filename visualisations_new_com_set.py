
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
graphs_path_final=Path(r"C:\Users\RikL\Box\ONL-IMK\Communication\COVID 19 Response\factsheets\graphs")
# data:

# sourcedata (all cntrs trustitems)
data = pd.read_hdf(data_path/'trust_data_compsets.h5')
# averages--> totals_m
totals_m = pd.read_hdf(data_path/'trust_avg_by_inst_cntry_compsets.h5')
# percentage often/always trust --> totals_d
totals_d = pd.read_hdf(data_path/"trust_inst_cntry_perc_compsets.h5")

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

coltitles = {'cctrust_tradrelileader': 'Traditional\n&\nreligious\nleaders',
 'cctrust_localcso': "Local\ncso's",
 'cctrust_localgov': 'Local\ngovernment',
 'cctrust_internngo': "Int.\nngo's",
 'cctrust_centralgov': 'Central\ngovernment',
 'cctrust_media': 'Media'}

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

plt.figtext(s = "Question: How often do you trust [institution] to do the right thing?\n1-never (left)-2-not very often-3-most of the time-4-always (right)\nData collected Oct-2010, Feb. 2020\nDots represent average by country-project and institution",
            size = 'medium', color = 'gray', x = 0, y=0.95, horizontalalignment='left')
# footnote
plt.figtext(s = "Source: Oxfam Novib: SP-surveys on citizens' voice, n=6413\nTrust in traditional and religious leaders was seperated for all countries except Uganda.\nValues represent the mean of trust in traditional and trust in religious leaders.\nNo data available for OPT-trust in media",
            size = 'small', x = 0, y = 0, color='grey')
plt.subplots_adjust(bottom = 0.3, top = 0.8)
fig.savefig(graphs_path_final/'Trust_averages_compsets.pdf', bbox_inches='tight')



# plot by cntry percentages
# transpose totals_d for easy sorting &labeling

totals_perc=totals_d[vizitems_d].T
#labels replace newline chars
instlabels={k + '_d': v.replace('\n', ' ') for k,v in coltitles.items()}
totals_perc['instlabels']=totals_perc.index.map(instlabels)
totals_perc.set_index('instlabels', inplace=True)
#neat country labels
cntrylabels = dict(zip(list(totals_perc.columns), [
    nlab.replace('_', '\n(')+')' for nlab in list(totals_perc.columns)]
    ))

#reverse the country list so that highest scoring cntrys plotted first. 
countries_l_r=countries_l[::-1]

nrows=8
ncols=1
fig, ax=plt.subplots(nrows = nrows, ncols = ncols, sharex='col', figsize=(6, 18))
axs=fig.axes
# percentages
for i, (cnt, color) in enumerate(zip(countries_l_r, hex_values)):
    
    means=totals_perc.loc[:,cnt].sort_values(ascending=True)
    axs[i].barh(y = means.index, width = means, height=0.5, color = color, clip_on=False)
#plot labels
    bars=[rect for rect in axs[i].get_children() if isinstance(rect, mpatches.Rectangle)]
    for bar in bars: 
        width=bar.get_width()
        percentage="{:.0%}".format(round(width,2))
        offset=0.03
        axs[i].text(width+offset, bar.get_y(), percentage, color=bar.get_facecolor(), size='small')
# x-axis
    axs[i].set_xlim(0, 1)
    axs[i].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

# spines & ticks
    axs[i].spines['left'].set_position(('outward', 1))
    axs[i].spines['bottom'].set_position(('outward', 5))
    axs[i].spines['right'].set_visible(False)
    if  i < 7:
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].tick_params(axis = 'x', which = 'major', length = 0)
    if i==7:
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['bottom'].set_visible(True)
        axs[i].tick_params(axis = 'x', which = 'major', length = 2)

    
# subplottitles
    axs[i].set_title(cntrylabels[cnt], size = 'medium', color = color)

    #axs[i].set_xticklabels(xlabels, size = 'small', rotation = 80)
# grid
    axs[i].xaxis.grid(linestyle = 'dotted', color = 'gray')

# titles
fig.suptitle("Which insitutions are trusted most?\nRanking of often/always trusted institutions, by project", x=-0.16,y = 1.5,fontsize = 'x-large', horizontalalignment = 'left')
plt.figtext(s = "Question: How often do you trust [institution] to do the right thing?\n(1-never-2-Not very often-3-most of the time-4-always)\nData collected Oct-2010, Feb. 2020\nDots represent share of people by country-project\nthat trust institution most of the time or always",
            size = 'medium', color = 'gray', x = -0.16, y=1.23, horizontalalignment='left')
# footnote
plt.figtext(s = "Source: Oxfam Novib: SP-surveys on citizens' voice, n=6413\nTrust in traditional and religious leaders was seperated for all countries except Uganda.\nValues represent the mean of trust in traditional and trust in religious leaders.\nNo data available for OPT-trust in media",
            size = 'small', x = -0.16, y = 0, color='grey')

plt.subplots_adjust(bottom = 0.05, top =1.2, wspace=0.5, left=0.2)

fig.savefig(graphs_path_final/'Trust_ranking_by_all_cntry_compset.pdf', bbox_inches='tight')


#seperate graph for each counttry


# get the max nr of observation for each country on all items as n
nobsbycol=pd.DataFrame(index=totals_m.index)
for col  in coltitles:
    nobsbycol[col]=pd.Series(totals_m.loc[:, idx[col, 'count']])
#now map the index so that it corresponds with countries_l_r
#newlabels reversed keys and vals leads to proper labeling in totals_perc

maxnobs=pd.DataFrame(nobsbycol.max(axis=1)).rename(index={v:k for k,v in newlabels.items()})
maxnobs.columns=['maxnrofobs']
#split countrynames and project

countrynames, project = zip(*(s.split("_") for s in countries_l_r))
countrynames=list(countrynames) # have nice list to iterate over
projectextended=list(pd.Series(project).map({'cf':'Conflict & Fragility', 'f4d': 'Finance for Development', 'r2f': 'Right to Food'}))

for (cnt, color, cntname, project) in zip(countries_l_r, hex_values, countrynames, projectextended):
    means=totals_perc.loc[:,cnt].sort_values(ascending=True)
    fig, ax1=plt.subplots(nrows=1, ncols=1,figsize=(5.3, 2))

    nrobs=maxnobs.at[cnt,'maxnrofobs']
    ax1.barh(y = means.index, width = means, height=0.5, color = color, clip_on=False)
#plot labels
    bars=[rect for rect in ax1.get_children() if isinstance(rect, mpatches.Rectangle)]
    for bar in bars: 
        width=bar.get_width()
        percentage="{:.0%}".format(round(width,2))
        offset=0.03
        ax1.text(width+offset, (bar.get_y()+.25), percentage, color=bar.get_facecolor(), size='small', va='center'), 
#ylabel

# x-axis
    ax1.set_xlim(0, 1)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
# spines
    sns.despine()

    fig.suptitle("Which insitutions are trusted most often?\nRanking of often/always trusted institutions", x=-0.16,y = 1.4,fontsize = 'x-large', horizontalalignment = 'left')
    plt.figtext(s = "Question: How often do you trust [institution] to do the right thing?\n(1-never-2-Not very often-3-most of the time-4-always)\nPercentage of people that trust institution most of the time or always",
            size = 'medium', color = 'gray', x = -0.16, y=0.95, horizontalalignment='left')
# footnote
    plt.figtext(s = "Source: Oxfam Novib: SP-"+str(project)+" surveys on citizens' voice in "+cntname+"\nn= "+str(nrobs),
            size = 'small', x =-0.16, y = -0.2, color='grey')
    filename= cnt + "_trustranking.svg"    
    fig.savefig(graphs_path_final/filename, bbox_inches='tight')
