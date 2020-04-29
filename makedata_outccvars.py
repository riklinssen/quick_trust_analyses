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
    cols = [c for c in checkcols.columns if 'cctrust' in c  or 'period' in c]
    #cols2 = [c for c in cols if 'out_' not in c]
    trustvars.append(cols)
dataoverview = dict(zip(cntrs, trustvars))


##uganda r2f soes not have out_cctrust variables stick to cctrust ones so revert to raw data. 

#check if period is in values
print([cntr for cntr, col in dataoverview.items() if 'period' not in col])

