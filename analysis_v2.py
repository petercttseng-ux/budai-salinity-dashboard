# -*- coding: utf-8 -*-
"""v2：布袋 / 東石 雙站 鹽度 × 降雨 × 河川徑流量（八掌溪 / 朴子溪）相關性分析
資料：嘉義縣環保局 115/10/01 嘉環水字第1150033589號函附件 1–5；中央氣象署 C0M750 布袋、C0M710 東石 逐時資料 2026/05–09
"""
import pandas as pd, numpy as np, json, math, os
from scipy import stats
from load import read_wq, load_station, read_cwa, RAW

OUT = os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT, exist_ok=True)
T0, T1 = pd.Timestamp('2026-05-01 01:00'), pd.Timestamp('2026-10-01 00:00')

# ---------------- 氣象：兩站互補缺漏 ----------------
met = read_cwa()
H = pd.date_range(T0, T1, freq='h')
def st(code):
    return met[met.stno == code].set_index('dt').reindex(H)
mb, md = st('C0M750'), st('C0M710')
fill_log = {}
for name, a, b in [('C0M750', mb, md), ('C0M710', md, mb)]:
    miss = a['rain'].isna()
    fill_log[name] = dict(missing=int(miss.sum()), filled=int((miss & b['rain'].notna()).sum()))
    a['rain_filled'] = a['rain'].fillna(b['rain']).fillna(0.0)
    for c in ['temp', 'rh', 'wspd']:
        a[c] = a[c].fillna(b[c]).interpolate(limit_direction='both')

# ---------------- HBV-lite 徑流模式 ----------------
P = dict(K_OROG=1.5, FC=150.0, BETA=2.0, LP=0.7, TQ=15.0, TS=30 * 24.0, PERC=0.35, BASE0=2.0, SM0=0.45)
def hbv(rain, temp, area):
    k_q = 1 - math.exp(-1 / P['TQ']); k_s = 1 - math.exp(-1 / P['TS'])
    Pb = rain.values * P['K_OROG']
    doy = rain.index.dayofyear.values; lat = 23.4
    decl = 0.4093 * np.sin(2 * np.pi * doy / 365 - 1.405)
    ws = np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(decl), -1, 1))
    t = temp.values; es = 6.108 * np.exp(17.27 * t / (t + 237.3))
    pet = 0.1651 * (24 * ws / np.pi / 12) * (216.7 * es / (t + 273.3)) / 24
    SM = P['SM0'] * P['FC']; Sq = 0.0
    Ss = (P['BASE0'] * 3600 / (area * 1e6) * 1000) / k_s
    q, qb, sm = [], [], []
    for p_, e_ in zip(Pb, pet):
        r = p_ * min(SM / P['FC'], 1) ** P['BETA']; SM += p_ - r
        SM = max(SM - e_ * min(SM / (P['LP'] * P['FC']), 1), 0)
        if SM > P['FC']: r += SM - P['FC']; SM = P['FC']
        Sq += r * (1 - P['PERC']); Ss += r * P['PERC']
        a1 = Sq * k_q; a2 = Ss * k_s; Sq -= a1; Ss -= a2
        q.append(a1 + a2); qb.append(a2); sm.append(SM)
    f = area * 1e6 / 1000 / 3600
    return pd.DataFrame({'Q': np.array(q) * f, 'Qb': np.array(qb) * f, 'soil': sm, 'P_basin': Pb}, index=rain.index)

# ---------------- 統計工具 ----------------
def corr(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 8: return dict(n=int(m.sum()), r=None, p=None, rho=None, prho=None)
    r, p = stats.pearsonr(x[m], y[m]); rho, pr = stats.spearmanr(x[m], y[m])
    return dict(n=int(m.sum()), r=round(float(r), 3), p=float(p), rho=round(float(rho), 3), prho=float(pr))
def partial(x, y, z):
    m = x.notna() & y.notna() & z.notna(); x, y, z = x[m], y[m], z[m]
    if len(x) < 8: return dict(r=None, p=None, n=int(len(x)))
    rxy, rxz, ryz = [stats.pearsonr(a, b)[0] for a, b in [(x, y), (x, z), (y, z)]]
    r = (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)); dof = len(x) - 3
    p = 2 * (1 - stats.t.cdf(abs(r * math.sqrt(dof / (1 - r ** 2))), dof))
    return dict(r=round(float(r), 3), p=float(p), n=int(len(x)))
def ols(d, cols, names):
    X = d[cols]; m = X.notna().all(axis=1) & d['sal_mean'].notna()
    Xn = np.column_stack([np.ones(m.sum()), X[m].values]); y = d['sal_mean'][m].values
    b = np.linalg.lstsq(Xn, y, rcond=None)[0]; yh = Xn @ b; n, k = Xn.shape
    r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    se = np.sqrt(((y - yh) ** 2).sum() / (n - k) * np.diag(np.linalg.inv(Xn.T @ Xn))); t = b / se
    return dict(vars=['截距'] + names, beta=[round(float(v), 4) for v in b], t=[round(float(v), 2) for v in t],
                p=[float(v) for v in 2 * (1 - stats.t.cdf(np.abs(t), n - k))],
                r2=round(float(r2), 3), adj_r2=round(float(1 - (1 - r2) * (n - 1) / (n - k)), 3), n=int(n))
def lagged(a, b, L=10): return [dict(lag=i, **corr(a.shift(i), b)) for i in range(L + 1)]

def events(d):
    ev = []; rain = d['rain'].fillna(0).values; i = 0
    while i < len(d):
        if rain[i] >= 10:
            j = i; tot = 0
            while j < len(d) and rain[j] > 0: tot += rain[j]; j += 1
            if tot >= 20:
                pre = d['sal_mean'].iloc[max(0, i - 3):i].dropna()
                win = d.iloc[i:min(len(d), j + 5)]
                if len(pre) and win['sal_mean'].notna().any():
                    imin = win['sal_mean'].idxmin()
                    ev.append(dict(start=d['date'].iloc[i].strftime('%Y-%m-%d'), days=int(j - i), rain=round(float(tot), 1),
                                   Qpeak=round(float(d['Qmax'].iloc[i:min(len(d), j + 3)].max()), 1),
                                   sal_pre=round(float(pre.mean()), 2), sal_min=round(float(win['sal_mean'].min()), 2),
                                   drop=round(float(pre.mean() - win['sal_mean'].min()), 2),
                                   lag_days=int((d['date'][imin] - d['date'].iloc[i]).days),
                                   ss_max=round(float(win['ss'].max()), 1)))
            i = j + 1
        else: i += 1
    return ev

# ---------------- 單站分析 ----------------
def run_station(key, wq, metdf, river, area, extra_flags=None):
    wq = wq.copy(); raw_n = len(wq)
    bad = (wq['鹽度'] <= 25) | (wq['鹽度'] > 36) | (wq['導電度'] <= 5)
    wq.loc[bad, '鹽度'] = np.nan
    flagged = 0
    if extra_flags is not None:
        fm = extra_flags(wq); flagged = int(fm.sum())
        for c in ['化學需氧量', '懸浮固體', '水中油(總油脂)', '溶氧', '酸鹼值', '葉綠素']: wq.loc[fm, c] = np.nan
    wq.loc[wq['水位'] <= 0, '水位'] = np.nan
    wq['date'] = wq['dt'].dt.floor('D')
    hyd = hbv(metdf['rain_filled'], metdf['temp'], area)
    h = pd.concat([metdf[['rain_filled', 'temp', 'wspd', 'rh']].rename(columns={'rain_filled': 'rain'}), hyd], axis=1)
    h['date'] = (h.index - pd.Timedelta(hours=1)).floor('D')
    d = h.groupby('date').agg(rain=('rain', 'sum'), rain_basin=('P_basin', 'sum'), Q=('Q', 'mean'), Qmax=('Q', 'max'),
                              Qb=('Qb', 'mean'), soil=('soil', 'mean'), temp=('temp', 'mean'), wspd=('wspd', 'mean'), rh=('rh', 'mean'))
    g = wq.groupby('date')
    lvl_h = wq.set_index('dt')['水位'].resample('h').mean()
    tide = lvl_h.groupby(lvl_h.index.floor('D')).agg(lambda s: (s.max() - s.min()) if s.notna().sum() >= 18 else np.nan)
    w = pd.DataFrame({'sal_mean': g['鹽度'].mean(), 'sal_min': g['鹽度'].min(), 'sal_max': g['鹽度'].max(),
                      'sal_std': g['鹽度'].std(), 'wtemp': g['溫度'].mean(), 'do': g['溶氧'].mean(), 'chl': g['葉綠素'].mean(),
                      'ph': g['酸鹼值'].mean(), 'ss': g['懸浮固體'].mean(), 'cod': g['化學需氧量'].mean(),
                      'cond': g['導電度'].mean(), 'n_obs': g['鹽度'].count(), 'tide_rng': tide / 1000.0})
    w.loc[w['n_obs'] < 144, [c for c in w.columns if c != 'n_obs']] = np.nan   # 日資料完整率 < 50 % 不採用
    d = d.join(w, how='left').reset_index()
    d['logQ'] = np.log10(d['Q']); d['rain3'] = d['rain'].rolling(3, min_periods=1).sum(); d['rain7'] = d['rain'].rolling(7, min_periods=1).sum()
    d['runoff_vol'] = d['Q'] * 86400 / 1e4
    full = d.copy()
    d = d[d['sal_mean'].notna()].reset_index(drop=True)

    def block(dd):
        pairs = {
            '日降雨量 → 日均鹽度': corr(dd['rain'], dd['sal_mean']),
            '3日累積雨量 → 日均鹽度': corr(dd['rain3'], dd['sal_mean']),
            '7日累積雨量 → 日均鹽度': corr(dd['rain7'], dd['sal_mean']),
            '日均徑流量 → 日均鹽度': corr(dd['Q'], dd['sal_mean']),
            'log₁₀(徑流量) → 日均鹽度': corr(dd['logQ'], dd['sal_mean']),
            'log₁₀(徑流量) → 日最低鹽度': corr(dd['logQ'], dd['sal_min']),
            '日降雨量 → 日均徑流量': corr(dd['rain'], dd['Q']),
            'log₁₀(徑流量) → 懸浮固體': corr(dd['logQ'], dd['ss']),
            '潮差 → 日均鹽度': corr(dd['tide_rng'], dd['sal_mean']),
        }
        pc = {'log₁₀Q vs 鹽度｜控制潮差': partial(dd['logQ'], dd['sal_mean'], dd['tide_rng']),
              'log₁₀Q vs 鹽度｜控制當日降雨': partial(dd['logQ'], dd['sal_mean'], dd['rain']),
              'log₁₀Q vs 鹽度｜控制水溫': partial(dd['logQ'], dd['sal_mean'], dd['wtemp']),
              '7日雨量 vs 鹽度｜控制潮差': partial(dd['rain7'], dd['sal_mean'], dd['tide_rng'])}
        use_tide = dd['tide_rng'].notna().sum() >= 0.7 * len(dd)
        reg = ols(dd, ['logQ', 'tide_rng', 'wspd'] if use_tide else ['logQ', 'wspd'],
                  ['log₁₀(徑流量)', '潮差 (m)', '風速'] if use_tide else ['log₁₀(徑流量)', '風速'])
        return dict(pairs=pairs, partial=pc, reg=reg, lag_rain=lagged(dd['rain'], dd['sal_mean']),
                    lag_Q=lagged(dd['logQ'], dd['sal_mean']), n=int(len(dd)),
                    period=[dd['date'].min().strftime('%Y-%m-%d'), dd['date'].max().strftime('%Y-%m-%d')])
    main = block(d)
    # lagged analyses need contiguous series: compute on full (with NaN sal) to keep spacing
    main['lag_rain'] = lagged(full['rain'], full['sal_mean']); main['lag_Q'] = lagged(full['logQ'], full['sal_mean'])
    wq_days = full[full['sal_mean'].notna()]
    summ = dict(key=key, river=river, basin_area=area,
                period=main['period'], n_days=main['n'], n_wq_raw=raw_n, qc_removed=int(bad.sum()), flagged_copy=flagged,
                rain_total=round(float(wq_days['rain'].sum()), 1), rain_total_all=round(float(full['rain'].sum()), 1),
                rain_max=round(float(wq_days['rain'].max()), 1),
                sal_mean=round(float(d['sal_mean'].mean()), 2), sal_min=round(float(d['sal_min'].min()), 2),
                sal_max=round(float(d['sal_max'].max()), 2), sal_daily_min=round(float(d['sal_mean'].min()), 2),
                sal_daily_rng=round(float(d['sal_mean'].max() - d['sal_mean'].min()), 2),
                Q_mean=round(float(wq_days['Q'].mean()), 1), Q_max=round(float(full['Qmax'].max()), 1),
                runoff_coef=round(float(full['Q'].sum() * 86400 / (full['rain_basin'].sum() / 1000 * area * 1e6)), 3),
                wtemp_mean=round(float(d['wtemp'].mean()), 1), air_mean=round(float(d['temp'].mean()), 1))
    ev = events(full)
    return dict(summary=summ, main=main, events=ev,
                daily=json.loads(full.assign(date=full['date'].dt.strftime('%Y-%m-%d')).round(3).to_json(orient='records'))), full, block

bd = load_station('布袋'); ds = load_station('東石')
R_bd, F_bd, blk = run_station('budai', bd, mb, '八掌溪', 474.74)
copy_mask = lambda w: (w['dt'] < pd.Timestamp('2026-07-22'))
R_ds, F_ds, _ = run_station('dongshi', ds, md, '朴子溪', 426.60, extra_flags=copy_mask)
# 東石 sensitivity: exclude 7/1–7/21
sub = F_ds[(F_ds['date'] >= '2026-07-22') & F_ds['sal_mean'].notna()].reset_index(drop=True)
R_ds['sens_after0722'] = {k: v for k, v in blk(sub).items() if k in ('pairs', 'partial', 'n', 'period')}
# 布袋 v1 window replication with county data
v1 = F_bd[(F_bd['date'] >= '2026-05-10') & (F_bd['date'] <= '2026-07-15') & F_bd['sal_mean'].notna()].reset_index(drop=True)
R_bd['v1_window'] = {k: v for k, v in blk(v1).items() if k in ('pairs', 'n', 'period')}

# ---------------- 資料品質稽核 ----------------
orig = read_wq(os.path.join(RAW, '布袋港海域水質自動監測系統資料.csv'))
mm = orig.merge(bd, on='dt', suffixes=('_o', '_c'))
j = ds.merge(bd, on='dt', suffixes=('_d', '_b'))
ident = {c: round(float(((j[c + '_d'] - j[c + '_b']).abs() < 1e-9).mean()), 3)
         for c in ['化學需氧量', '懸浮固體', '水中油(總油脂)', '溶氧', '酸鹼值', '葉綠素', '鹽度', '溫度', '導電度', '磷酸鹽']}
qa = dict(
    budai_vs_v1=dict(overlap=int(len(mm)), sal_diff_rows=int(((mm['鹽度_o'] - mm['鹽度_c']).abs() > 0).sum()),
                     sal_diff_days=sorted(mm.loc[(mm['鹽度_o'] - mm['鹽度_c']).abs() > 0, 'dt'].dt.strftime('%m/%d').unique().tolist()),
                     other_identical=bool(((mm['溫度_o'] - mm['溫度_c']).abs().max() == 0))),
    dongshi_budai_copy=dict(overlap=int(len(j)), period='2026-07-01 ～ 2026-07-21', identical_fraction=ident),
    budai_wtemp=dict(wtemp_mean=R_bd['summary']['wtemp_mean'], air_mean=R_bd['summary']['air_mean']),
    dongshi_wtemp=dict(wtemp_mean=R_ds['summary']['wtemp_mean'], air_mean=R_ds['summary']['air_mean']),
    rain_fill=fill_log,
    rain_daily_r=round(float(pd.concat([mb['rain'], md['rain']], axis=1).dropna().resample('D').sum().corr().iloc[0, 1]), 3),
)
monthly = pd.DataFrame({'布袋 C0M750': mb['rain_filled'], '東石 C0M710': md['rain_filled']})
monthly = monthly.groupby((monthly.index - pd.Timedelta(hours=1)).month).sum().round(1)
out = dict(version='v2', generated=pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d'), params=P, qa=qa,
           monthly_rain={int(k): v for k, v in monthly.to_dict(orient='index').items()},
           stations=dict(budai=R_bd, dongshi=R_ds))
json.dump(out, open(os.path.join(OUT, 'results_v2.json'), 'w', encoding='utf-8'), ensure_ascii=False, default=float)
F_bd.assign(station='布袋').to_csv(os.path.join(OUT, 'daily_budai_v2.csv'), index=False, encoding='utf-8-sig')
F_ds.assign(station='東石').to_csv(os.path.join(OUT, 'daily_dongshi_v2.csv'), index=False, encoding='utf-8-sig')

# ---------------- print ----------------
for n, R in [('布袋', R_bd), ('東石', R_ds)]:
    print('=====', n); print(R['summary'])
    for k, v in R['main']['pairs'].items(): print(f"  {k}: r={v['r']} p={v['p']:.3g} n={v['n']}")
    print('  partial', {k: (v['r'], round(v['p'], 4) if v['p'] is not None else None) for k, v in R['main']['partial'].items()})
    print('  reg', R['main']['reg'])
    print('  lagR', [(o['lag'], o['r']) for o in R['main']['lag_rain']]); print('  lagQ', [(o['lag'], o['r']) for o in R['main']['lag_Q']])
    for e in R['events']: print('  ev', e)
print('東石 after 7/22', {k: (v['r'], v['n']) for k, v in R_ds['sens_after0722']['pairs'].items()})
print('東石 after 7/22 partial', {k: v['r'] for k, v in R_ds['sens_after0722']['partial'].items()})
print('布袋 v1 window', {k: v['r'] for k, v in R_bd['v1_window']['pairs'].items()})
print(json.dumps(qa, ensure_ascii=False)); print(monthly)
