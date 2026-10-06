import pandas as pd, numpy as np, io, os, glob
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')
def read_wq(fn):
    txt = open(fn,'rb').read().decode('big5', errors='replace')
    lines = txt.replace('\r','').split('\n')
    i = next(i for i,l in enumerate(lines) if l.startswith('日期/時間'))
    df = pd.read_csv(io.StringIO('\n'.join(lines[i:])))
    df.columns=[c.strip() for c in df.columns]
    df['dt']=pd.to_datetime(df['日期/時間'],format='%Y/%m/%d %H:%M',errors='coerce')
    df=df.dropna(subset=['dt'])
    for c in df.columns:
        if c not in ('日期/時間','dt'): df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.drop(columns=['日期/時間']).sort_values('dt').reset_index(drop=True)
def load_station(name):
    fs = sorted(glob.glob(os.path.join(RAW, f'376500305I_*({name}).csv')))
    df = pd.concat([read_wq(f) for f in fs]).drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    return df
def read_cwa():
    recs=[]; cols=None
    for fn in sorted(glob.glob(os.path.join(RAW,'LotsDataReports-CWA*.csv'))):
        for line in open(fn,encoding='utf-8-sig'):
            line=line.strip().lstrip('﻿')
            if line.startswith('# stno'): cols=[c.strip() for c in line.lstrip('#').split(',')]
            elif line.startswith('C0M'):
                recs.append(dict(zip(cols,[p.strip() for p in line.split(',')])))
    m=pd.DataFrame(recs).drop_duplicates(['stno','yyyymmddhh'])
    # hour 24 -> next day 00
    ymd=m['yyyymmddhh'].str[:8]; hh=m['yyyymmddhh'].str[8:].astype(int)
    m['dt']=pd.to_datetime(ymd,format='%Y%m%d')+pd.to_timedelta(hh,unit='h')
    for c in ['PS01','TX01','RH01','WD01','WD02','WD07','PP01']: m[c]=pd.to_numeric(m[c],errors='coerce')
    m['rain']=m['PP01'].replace(-9.8,0.05); m.loc[m['rain']<0,'rain']=np.nan
    for a,b in [('TX01','temp'),('RH01','rh'),('WD01','wspd'),('WD02','wdir')]:
        m[b]=m[a].where(m[a]>-9)
    return m.sort_values(['stno','dt']).reset_index(drop=True)
