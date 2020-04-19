# SP - quick analyses trust - 

Code to reproduce visualisations for quick comparative analyses on trust here <add link> --> add link to report here. 

# Technologies
Project is created with: 
- Python 3.7.5 (see requirements.txt for packages used)

# Structure
```
├───Data            --> list of links to dataset locations on Box for each SP endline file (raw data), + intermediate storage of files with estimates/aggregates (in hdf5-format)
├───Graphs          --> visualisations in pdf/png/svg
|
├─makedata.py       --> script to clean data and generates estimates/aggregates
├─visualisation.py  --> script to generate visualisations
└─requirements.txt  --> requirements (packages/libraries) for python env. setup
```
# Datasets used
(Only accessible via ON-Box)
- OPT	~Box\ONL-IMK\2.0 Projects\Current\16-01 SP\OPTI\5. Endline survey\1. Data and data analysis\2. Clean\F4D OPTI Baseline + Endline Analysis.dta
- Burundi	~Box\ONL-IMK\2.0 Projects\Current\16-01 SP\Burundi\4. Endline survey\Data and data analysis\Data\R2F BI Baseline + Endline Analysis (for comparative analysis).dta
- Myanmar	~Box\ONL-IMK\2.0 Projects\Current\16-01 SP\Myanmar\Endline survey\Data and data analysis\Data\MYA Analysis ITT.dta
- Niger	~Box\ONL-IMK\2.0 Projects\Current\16-01 SP\Niger\5. Endline survey\C&F\1. Data and data analysis\2. Clean\C&F NI Baseline + Endline Analysis.dta
- Niger	~Box\ONL-IMK\2.0 Projects\Current\16-01 SP\Niger\5. Endline survey\F4D\1. Data and data analysis\2. Clean\F4D NI Baseline + Endline Analysis.dta
- Uganda	~Box\ONL-IMK\2.0 Projects\Current\16-01 SP\Uganda\Endline survey\F4D\Data and data analysis\Data\F4D UGA Baseline + Endline.dta
- Uganda	~Box\ONL-IMK\2.0 Projects\Current\16-01 SP\Uganda\Endline survey\R2F\Data and data analysis\Data\R2F UGA Baseline + Endline Analysis.dta






