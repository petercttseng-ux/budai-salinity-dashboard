# -*- coding: utf-8 -*-
"""布袋港鹽度 × 降雨量 × 八掌溪徑流量 相關性分析"""
import pandas as pd, numpy as np, json, math
from scipy import stats

# ============================================================
# 0. 讀入
# ============================================================
hourly = pd.read_csv('hourly.csv', parse_dates=['hour'])
wq = pd.read_pickle('wq.pkl')

# ---- 鹽度品管 (QC) ----
# 儀器異常值：導電度=0 或 鹽度落在物理不合理區間
raw_n = len(wq)
bad = (wq['鹽度'] <= 25) | (wq['鹽度'] > 36) | (wq['導電度'] <= 5)
wq.loc[bad, '鹽度'] = np.nan
qc_removed = int(bad.sum())

# 以 5 分鐘資料重建逐時 / 逐日
wq['hour'] = wq['dt'].dt.floor('h')
wq['date'] = wq['dt'].dt.floor('D')

# ============================================================
# 1. 八掌溪逕流量推估模式 (HBV-lite 連續式概念模式)
# ============================================================
A_KM2 = 474.74          # 八掌溪流域面積 (km2)，水利署/維基
K_OROG = 1.50           # 流域面雨量 / 布袋沿海站雨量 (地形增雨係數)
FC = 150.0              # 土壤最大含水量 (mm)
BETA = 2.0              # 入滲遞減指數
LP = 0.7                # 蒸散應力係數
K_QUICK = 1 - math.exp(-1 / 15.0)   # 直接逕流線性水庫，滯時 15 hr
K_SLOW = 1 - math.exp(-1 / (30 * 24))  # 基流線性水庫，滯時 30 day
PERC = 0.35             # 有效降雨進入地下水之比例

h = hourly.copy().sort_values('hour').reset_index(drop=True)
h['rain'] = h['rain'].fillna(0.0)
h['temp'] = h['temp'].interpolate(limit_direction='both')
h['P_basin'] = h['rain'] * K_OROG

# Hamon 潛勢蒸散 (逐時分配)
lat = 23.38
doy = h['hour'].dt.dayofyear
decl = 0.4093 * np.sin(2 * np.pi * doy / 365 - 1.405)
ws = np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(decl), -1, 1))
daylight = 24 * ws / np.pi
es = 6.108 * np.exp(17.27 * h['temp'] / (h['temp'] + 237.3))
pet_day = 0.1651 * (daylight / 12) * (216.7 * es / (h['temp'] + 273.3))  # mm/day
h['PET'] = pet_day / 24.0

SM = 0.45 * FC          # 乾季末期初始土壤含水
Sq, Ss = 0.0, 2.0 / (A_KM2 * 1000 / 3600) if False else 0.0
# 初始基流 ~2 cms  -> mm/hr = cms*3600/(A*1e6)*1000
base0_cms = 2.0
Ss = (base0_cms * 3600 / (A_KM2 * 1e6) * 1000) / K_SLOW

sm_l, q_l, b_l = [], [], []
for P, PET in zip(h['P_basin'].values, h['PET'].values):
    r = P * (min(SM / FC, 1.0) ** BETA)      # 有效降雨
    SM = SM + P - r
    aet = PET * min(SM / (LP * FC), 1.0)
    SM = max(SM - aet, 0.0)
    if SM > FC:
        r += SM - FC
        SM = FC
    Sq += r * (1 - PERC)
    Ss += r * PERC
    qq = Sq * K_QUICK
    qb = Ss * K_SLOW
    Sq -= qq
    Ss -= qb
    sm_l.append(SM); q_l.append(qq); b_l.append(qb)

h['soil'] = sm_l
h['q_direct_mm'] = q_l
h['q_base_mm'] = b_l
h['runoff_mm'] = h['q_direct_mm'] + h['q_base_mm']
# mm/hr -> m3/s
h['Q_cms'] = h['runoff_mm'] / 1000 * (A_KM2 * 1e6) / 3600
h['Qb_cms'] = h['q_base_mm'] / 1000 * (A_KM2 * 1e6) / 3600

# ============================================================
# 2. 逐時 / 逐日資料表
# ============================================================
sal_h = wq.groupby('hour')['鹽度'].mean()
h['sal'] = h['hour'].map(sal_h)

h['date'] = (h['hour'] - pd.Timedelta(hours=1)).dt.floor('D')
d = h.groupby('date').agg(
    rain=('rain', 'sum'),
    rain_basin=('P_basin', 'sum'),
    Q=('Q_cms', 'mean'),
    Qmax=('Q_cms', 'max'),
    Qb=('Qb_cms', 'mean'),
    soil=('soil', 'mean'),
    temp=('temp', 'mean'),
    wspd=('wspd', 'mean'),
    rh=('rh', 'mean'),
    sal=('sal', 'mean'),
).reset_index()
# 逐日鹽度統計改由 5 分鐘原始資料計算（更穩健）
g = wq.groupby('date')
d = d.merge(pd.DataFrame({
    'sal_mean': g['鹽度'].mean(), 'sal_min': g['鹽度'].min(), 'sal_max': g['鹽度'].max(),
    'sal_std': g['鹽度'].std(), 'wtemp': g['溫度'].mean(), 'do': g['溶氧'].mean(),
    'chl': g['葉綠素'].mean(), 'ph': g['酸鹼值'].mean(), 'ss': g['懸浮固體'].mean(),
    'cod': g['化學需氧量'].mean(), 'n_obs': g['鹽度'].count(),
}).reset_index(), on='date', how='left')

d['runoff_vol'] = d['Q'] * 86400 / 1e4      # 萬 m3/day
d['logQ'] = np.log10(d['Q'])
d['rain3'] = d['rain'].rolling(3, min_periods=1).sum()
d['rain7'] = d['rain'].rolling(7, min_periods=1).sum()
d = d.dropna(subset=['sal_mean']).reset_index(drop=True)
print(d[['date', 'rain', 'Q', 'sal_mean', 'sal_min']].describe().round(2).to_string())

# ============================================================
# 3. 相關性分析
# ============================================================
def corr(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return dict(n=int(m.sum()), r=None, p=None, rho=None, prho=None)
    r, p = stats.pearsonr(x[m], y[m])
    rho, pr = stats.spearmanr(x[m], y[m])
    return dict(n=int(m.sum()), r=round(float(r), 3), p=float(p),
                rho=round(float(rho), 3), prho=float(pr))

pairs = {
    '降雨量 vs 鹽度(日均)': corr(d['rain'], d['sal_mean']),
    '降雨量 vs 鹽度(日最低)': corr(d['rain'], d['sal_min']),
    '徑流量 vs 鹽度(日均)': corr(d['Q'], d['sal_mean']),
    'log徑流量 vs 鹽度(日均)': corr(d['logQ'], d['sal_mean']),
    'log徑流量 vs 鹽度(日最低)': corr(d['logQ'], d['sal_min']),
    '降雨量 vs 徑流量': corr(d['rain'], d['Q']),
    '3日累積雨量 vs 鹽度': corr(d['rain3'], d['sal_mean']),
    '7日累積雨量 vs 鹽度': corr(d['rain7'], d['sal_mean']),
    '徑流量 vs 懸浮固體': corr(d['logQ'], d['ss']),
    '徑流量 vs 葉綠素': corr(d['logQ'], d['chl']),
    '鹽度 vs 水溫': corr(d['sal_mean'], d['wtemp']),
}
for k, v in pairs.items():
    print(k, v)

# ---- 落後相關 (lag cross-correlation) ----
def lagged(driver, target, maxlag=10):
    out = []
    for L in range(0, maxlag + 1):
        x = driver.shift(L)
        c = corr(x, target)
        out.append(dict(lag=L, **c))
    return out

lag_rain = lagged(d['rain'], d['sal_mean'])
lag_Q = lagged(d['logQ'], d['sal_mean'])
lag_rain_min = lagged(d['rain'], d['sal_min'])
print('lag rain->sal', [(o['lag'], o['r']) for o in lag_rain])
print('lag logQ->sal', [(o['lag'], o['r']) for o in lag_Q])

# ---- 偏相關：控制水溫後，徑流量 vs 鹽度 ----
def partial(x, y, z):
    m = x.notna() & y.notna() & z.notna()
    x, y, z = x[m], y[m], z[m]
    rxy = stats.pearsonr(x, y)[0]; rxz = stats.pearsonr(x, z)[0]; ryz = stats.pearsonr(y, z)[0]
    r = (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    n = len(x); dof = n - 3
    t = r * math.sqrt(dof / (1 - r ** 2))
    p = 2 * (1 - stats.t.cdf(abs(t), dof))
    return dict(r=round(float(r), 3), p=float(p), n=int(n))

pc = {
    'log徑流量 vs 鹽度 (控制水溫)': partial(d['logQ'], d['sal_mean'], d['wtemp']),
    '降雨量 vs 鹽度 (控制水溫)': partial(d['rain'], d['sal_mean'], d['wtemp']),
    'log徑流量 vs 鹽度 (控制降雨)': partial(d['logQ'], d['sal_mean'], d['rain']),
}
print(pc)

# ---- 複迴歸 ----
X = pd.DataFrame({'logQ': d['logQ'], 'wtemp': d['wtemp'], 'wspd': d['wspd']})
m = X.notna().all(axis=1) & d['sal_mean'].notna()
Xn = np.column_stack([np.ones(m.sum()), X[m].values])
yv = d['sal_mean'][m].values
beta, res, rank, sv = np.linalg.lstsq(Xn, yv, rcond=None)
yhat = Xn @ beta
r2 = 1 - ((yv - yhat) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
n_, k_ = Xn.shape
adj_r2 = 1 - (1 - r2) * (n_ - 1) / (n_ - k_)
se = np.sqrt(((yv - yhat) ** 2).sum() / (n_ - k_) * np.diag(np.linalg.inv(Xn.T @ Xn)))
tvals = beta / se
pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), n_ - k_))
reg = dict(vars=['截距', 'log10(徑流量)', '水溫', '風速'],
           beta=[round(float(b), 4) for b in beta],
           t=[round(float(t), 2) for t in tvals],
           p=[float(p) for p in pvals],
           r2=round(float(r2), 3), adj_r2=round(float(adj_r2), 3), n=int(n_))
print(reg)

# ---- 降雨事件分析 ----
events = []
rain = d['rain'].fillna(0).values
dates = d['date'].values
i = 0
while i < len(d):
    if rain[i] >= 10:
        j = i
        tot = 0
        while j < len(d) and rain[j] > 0:
            tot += rain[j]; j += 1
        if tot >= 20:
            pre = d['sal_mean'][max(0, i - 3):i]
            win = d.iloc[i:min(len(d), j + 5)]
            base = float(pre.mean()) if len(pre) else np.nan
            smin = float(win['sal_mean'].min())
            imin = int(win['sal_mean'].idxmin())
            events.append(dict(
                start=str(pd.Timestamp(dates[i]).date()),
                days=int(j - i), rain=round(float(tot), 1),
                Qpeak=round(float(d['Qmax'][i:min(len(d), j + 3)].max()), 1),
                sal_pre=round(base, 2), sal_min=round(smin, 2),
                drop=round(base - smin, 2),
                lag_days=int((d['date'][imin] - d['date'][i]).days),
                ss_max=round(float(win['ss'].max()), 1),
            ))
        i = j + 1
    else:
        i += 1
print(json.dumps(events, ensure_ascii=False, indent=1))

# ---- 逐時反應時間 (以最大事件) ----
h['sal_i'] = h['sal'].interpolate(limit=6)

summary = dict(
    period=[str(d['date'].min().date()), str(d['date'].max().date())],
    n_days=int(len(d)),
    n_wq_raw=int(raw_n), qc_removed=qc_removed,
    rain_total=round(float(d['rain'].sum()), 1),
    rain_days=int((d['rain'] > 0).sum()),
    rain_max=round(float(d['rain'].max()), 1),
    sal_mean=round(float(d['sal_mean'].mean()), 2),
    sal_min=round(float(d['sal_min'].min()), 2),
    sal_max=round(float(d['sal_max'].max()), 2),
    sal_std=round(float(d['sal_mean'].std()), 3),
    Q_mean=round(float(d['Q'].mean()), 2),
    Q_max=round(float(d['Qmax'].max()), 1),
    Q_min=round(float(d['Q'].min()), 2),
    vol_total=round(float(d['runoff_vol'].sum()), 1),
    runoff_coef=round(float(d['Q'].sum() * 86400 / (d['rain_basin'].sum() / 1000 * A_KM2 * 1e6)), 3),
    basin_area=A_KM2, k_orog=K_OROG,
)
print(summary)

out = dict(
    summary=summary, pairs=pairs, lag_rain=lag_rain, lag_Q=lag_Q,
    lag_rain_min=lag_rain_min, partial=pc, reg=reg, events=events,
    daily=json.loads(d.assign(date=d['date'].dt.strftime('%Y-%m-%d')).round(3).to_json(orient='records')),
    hourly=json.loads(h[['hour', 'rain', 'Q_cms', 'sal']].assign(
        hour=h['hour'].dt.strftime('%Y-%m-%d %H:00')).round(3).to_json(orient='records')),
)
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False)
d.to_csv('daily_final.csv', index=False)
print('saved')
