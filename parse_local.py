import pandas as pd, numpy as np, re, json, io, os

RAW = 'raw'

# ---------- 1. Salinity / water quality (Big5, 5-min) ----------
with open(os.path.join(RAW, '布袋港海域水質自動監測系統資料.csv'), 'rb') as f:
    txt = f.read().decode('big5', errors='replace')
lines = txt.split('\n')
hdr_i = next(i for i, l in enumerate(lines) if l.startswith('日期/時間'))
wq = pd.read_csv(io.StringIO('\n'.join(lines[hdr_i:])))
wq.columns = [c.strip() for c in wq.columns]
wq['dt'] = pd.to_datetime(wq['日期/時間'], format='%Y/%m/%d %H:%M', errors='coerce')
wq = wq.dropna(subset=['dt']).sort_values('dt')
for c in wq.columns:
    if c not in ('日期/時間', 'dt'):
        wq[c] = pd.to_numeric(wq[c], errors='coerce')
print('WQ rows', len(wq), wq['dt'].min(), wq['dt'].max())
print(wq.columns.tolist())
print(wq[['鹽度', '溫度', '溶氧', '葉綠素', '導電度']].describe().round(2))

# ---------- 2. Rainfall CWA C0M750 布袋 (hourly MH format) ----------
recs = []
cols = None
for fn in ['LotsDataReports1.csv', 'LotsDataReports2.csv', 'LotsDataReports3.csv']:
    with open(os.path.join(RAW, fn), encoding='utf-8-sig') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('# stno'):
                cols = [c.strip() for c in line.lstrip('#').split(',')]
            elif line.startswith('C0M750'):
                parts = [p.strip() for p in line.split(',')]
                recs.append(dict(zip(cols, parts)))
met = pd.DataFrame(recs).drop_duplicates(subset=['yyyymmddhh'])
met['dt'] = pd.to_datetime(met['yyyymmddhh'], format='%Y%m%d%H', errors='coerce')
# hour 24 style? check
print('bad dt', met['dt'].isna().sum())
for c in ['PS01', 'TX01', 'RH01', 'WD01', 'WD02', 'WD07', 'PP01']:
    met[c] = pd.to_numeric(met[c], errors='coerce')

SPECIAL = {-999.1: np.nan, -9.6: np.nan, -999.6: np.nan, -9.5: np.nan, -99.5: np.nan,
           -999.5: np.nan, -9999.5: np.nan, -9995: np.nan, -9.7: np.nan, -99.7: np.nan,
           -999.7: np.nan, -9999.7: np.nan, -9997: np.nan}


def clean(s, trace_to_zero=True):
    s = s.copy()
    if trace_to_zero:
        s = s.replace(-9.8, 0.05)  # trace rainfall
    s = s.where(s > -9, np.nan)
    return s


met['rain'] = clean(met['PP01'])
met['temp'] = clean(met['TX01'], False)
met['rh'] = clean(met['RH01'], False)
met['wspd'] = clean(met['WD01'], False)
met = met.dropna(subset=['dt']).sort_values('dt')
print('MET rows', len(met), met['dt'].min(), met['dt'].max(),
      'rain NaN', met['rain'].isna().sum(), 'total rain', met['rain'].sum())

# ---------- 3. Aggregate ----------
wq['date'] = wq['dt'].dt.floor('D')
wq['hour'] = wq['dt'].dt.floor('h')
# CWA hourly value at hh covers the preceding hour (hh-1 -> hh); align to hour label
met['hour'] = met['dt'].dt.floor('h')
met['date'] = (met['dt'] - pd.Timedelta(hours=1)).dt.floor('D')

sal_h = wq.groupby('hour').agg(sal=('鹽度', 'mean'), sal_min=('鹽度', 'min'),
                               temp_w=('溫度', 'mean'), do=('溶氧', 'mean'),
                               chl=('葉綠素', 'mean'), ph=('酸鹼值', 'mean'),
                               cond=('導電度', 'mean'), n=('鹽度', 'size')).reset_index()
met_h = met.set_index('hour')[['rain', 'temp', 'rh', 'wspd']]
hourly = sal_h.set_index('hour').join(met_h, how='outer').reset_index()

daily = pd.DataFrame({
    'sal_mean': wq.groupby('date')['鹽度'].mean(),
    'sal_min': wq.groupby('date')['鹽度'].min(),
    'sal_max': wq.groupby('date')['鹽度'].max(),
    'wtemp': wq.groupby('date')['溫度'].mean(),
    'do': wq.groupby('date')['溶氧'].mean(),
    'chl': wq.groupby('date')['葉綠素'].mean(),
    'ph': wq.groupby('date')['酸鹼值'].mean(),
    'cod': wq.groupby('date')['化學需氧量'].mean(),
    'ss': wq.groupby('date')['懸浮固體'].mean(),
    'n_obs': wq.groupby('date')['鹽度'].size(),
})
rain_d = met.groupby('date')['rain'].sum(min_count=1)
rain_hrs = met.groupby('date')['rain'].apply(lambda s: (s > 0).sum())
daily = daily.join(rain_d.rename('rain')).join(rain_hrs.rename('rain_hours'))
daily = daily.join(met.groupby('date')[['temp', 'rh', 'wspd']].mean())
daily.index.name = 'date'
daily = daily.reset_index()
print(daily.head(10).round(2).to_string())
print('days', len(daily), 'total rain mm', daily['rain'].sum())
print(daily.nlargest(12, 'rain')[['date', 'rain', 'sal_mean', 'sal_min']].round(2).to_string())

hourly.to_csv('hourly.csv', index=False)
daily.to_csv('daily.csv', index=False)
wq.to_pickle('wq.pkl')
