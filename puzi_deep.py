# -*- coding: utf-8 -*-
"""東石站 × 朴子溪推估徑流 × 東石降雨：深入分析（v2.1）"""
import pandas as pd, numpy as np, json, os, math, itertools
from scipy import stats
import analysis_v2 as A
from load import load_station

OUT = A.OUT
ds = load_station('東石').copy()
ds.loc[(ds['鹽度'] <= 25) | (ds['鹽度'] > 36) | (ds['導電度'] <= 5), '鹽度'] = np.nan
md = A.md
H = md.index

def godin(s):
    """Godin 24-24-25 h 低通濾波，去除半日與全日潮"""
    return s.rolling(24, center=True).mean().rolling(24, center=True).mean().rolling(25, center=True).mean()

sal_h = ds.set_index('dt')['鹽度'].resample('h').mean().reindex(H)
sal_h_i = sal_h.interpolate(limit=3)
sal_lp = godin(sal_h_i)
hyd = A.hbv(md['rain_filled'], md['temp'], 426.60)
h = pd.DataFrame({'rain': md['rain_filled'], 'Q': hyd['Q'], 'sal': sal_h, 'sal_lp': sal_lp}, index=H)
h['logQ'] = np.log10(h['Q'])
for w in [24, 72, 168, 336]:
    h[f'rainc{w}'] = h['rain'].rolling(w, min_periods=1).sum()
    h[f'Qvol{w}'] = h['Q'].rolling(w, min_periods=1).mean()
# tidal range hourly-aligned (daily)
lvl = ds.set_index('dt')['水位'].where(lambda s: s > 0).resample('h').mean().reindex(H)
tr = lvl.groupby(lvl.index.floor('D')).transform(lambda s: (s.max() - s.min()) / 1000 if s.notna().sum() >= 18 else np.nan)
h['tide'] = tr.rolling(25, center=True, min_periods=12).mean()

def r_(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 30: return None, None, int(m.sum())
    r, p = stats.pearsonr(x[m], y[m]); return round(float(r), 3), float(p), int(m.sum())

def neff(x, y):
    """有效樣本數（Bretherton 1999 lag-1 自相關修正）"""
    m = x.notna() & y.notna(); x, y = x[m], y[m]
    r1x = x.autocorr(1); r1y = y.autocorr(1)
    return max(3, int(len(x) * (1 - r1x * r1y) / (1 + r1x * r1y)))

def p_eff(r, n):
    if r is None: return None
    t = r * math.sqrt((n - 2) / max(1e-9, 1 - r * r)); return float(2 * (1 - stats.t.cdf(abs(t), n - 2)))

periods = {'all': ('2026-07-01', '2026-08-31 23:00'), 'after0722': ('2026-07-22', '2026-08-31 23:00')}
res = {}
for pk, (a, b) in periods.items():
    hh = h.loc[a:b]
    out = {}
    # hourly cross-correlation 0–240 h (every 6 h) on low-passed salinity
    xc_Q, xc_R = [], []
    for L in range(0, 241, 6):
        r, p, n = r_(hh['logQ'].shift(L), hh['sal_lp']); xc_Q.append(dict(lag=L, r=r))
        r2, p2, n2 = r_(hh['rainc72'].shift(L), hh['sal_lp']); xc_R.append(dict(lag=L, r=r2))
    out['xc_Q'] = xc_Q; out['xc_R'] = xc_R
    bq = min([o for o in xc_Q if o['r'] is not None], key=lambda o: o['r'])
    br = min([o for o in xc_R if o['r'] is not None], key=lambda o: o['r'])
    out['best_Q'] = bq; out['best_R'] = br
    # window comparison
    win = {}
    for w in [24, 72, 168, 336]:
        r, p, n = r_(np.log10(hh[f'Qvol{w}']), hh['sal_lp']); ne = neff(np.log10(hh[f'Qvol{w}']), hh['sal_lp'])
        r2, p2, _ = r_(hh[f'rainc{w}'], hh['sal_lp'])
        win[w] = dict(rQ=r, pQ=p_eff(r, ne), rR=r2, pR=p_eff(r2, ne), n=n, neff=ne)
    out['windows'] = win
    # 逐日顯著性：日均低通鹽度 vs 落後 L 日之 7 日平均徑流（有效樣本數修正）
    d = pd.DataFrame({'sal': hh['sal_lp'], 'Qv': hh['Qvol168'], 'R7': hh['rainc168'], 'tide': hh['tide']}).resample('D').mean()
    d['sal'] = d['sal'].where(hh['sal_lp'].resample('D').count() >= 18)
    dl = []
    for L in range(0, 11):
        x = np.log10(d['Qv']).shift(L); r, _, n = r_(x, d['sal']) if False else (None, None, 0)
        m = x.notna() & d['sal'].notna()
        if m.sum() >= 10:
            r = float(stats.pearsonr(x[m], d['sal'][m])[0]); ne = neff(x, d['sal'])
            r2 = float(stats.pearsonr(d['R7'].shift(L)[m], d['sal'][m])[0])
            dl.append(dict(lag=L, r=round(r, 3), p=p_eff(r, ne), rR=round(r2, 3), pR=p_eff(r2, ne), n=int(m.sum()), neff=ne))
    out['daily_lag'] = dl
    out['daily_best'] = min(dl, key=lambda o: o['r'])
    # chain: rain→Q, Q→sal, rain→sal, partial rain→sal | Q
    x, y, z = hh['rainc168'], hh['sal_lp'], np.log10(hh['Qvol168'])
    m = x.notna() & y.notna() & z.notna()
    rxy = stats.pearsonr(x[m], y[m])[0]; rxz = stats.pearsonr(x[m], z[m])[0]; ryz = stats.pearsonr(y[m], z[m])[0]
    pr_rain_Q = (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    pr_Q_rain = (ryz - rxz * rxy) / math.sqrt((1 - rxz ** 2) * (1 - rxy ** 2))
    # tide control (only where tide exists)
    mt = m & hh['tide'].notna()
    def pc3(x, y, z):
        rxy = stats.pearsonr(x, y)[0]; rxz = stats.pearsonr(x, z)[0]; ryz = stats.pearsonr(y, z)[0]
        return (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    out['chain'] = dict(rain_Q=round(float(rxz), 3), Q_sal=round(float(ryz), 3), rain_sal=round(float(rxy), 3),
                        rain_sal_given_Q=round(float(pr_rain_Q), 3), Q_sal_given_rain=round(float(pr_Q_rain), 3),
                        Q_sal_given_tide=round(float(pc3(z[mt], y[mt], hh['tide'][mt])), 3) if mt.sum() > 100 else None,
                        n=int(m.sum()), neff=neff(z, y), n_tide=int(mt.sum()))
    # regression on hourly low-passed salinity: sal_lp ~ log Qvol168 + tide
    if mt.sum() > 100:
        X = np.column_stack([np.ones(mt.sum()), z[mt], hh['tide'][mt]]); yv = y[mt].values
        bta = np.linalg.lstsq(X, yv, rcond=None)[0]; yh = X @ bta
        out['reg_tide'] = dict(beta=[round(float(v), 4) for v in bta], r2=round(float(1 - ((yv - yh) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()), 3), n=int(mt.sum()))
    res[pk] = out

# ---- 參數敏感度：r(log Qvol168, sal_lp) 與 r(log daily Q, daily sal) ----
base = dict(A.P)
grid = dict(K_OROG=[1.0, 1.25, 1.5, 1.75, 2.0], TQ=[8.0, 15.0, 30.0], TS=[15 * 24.0, 30 * 24.0, 60 * 24.0], PERC=[0.2, 0.35, 0.5])
sens = []
hh = h.loc['2026-07-22':'2026-08-31 23:00']
for k, vals in grid.items():
    for v in vals:
        A.P.update(base); A.P[k] = v
        q = A.hbv(md['rain_filled'], md['temp'], 426.60)['Q'].loc[hh.index]
        r1, _, _ = r_(np.log10(q.rolling(168, min_periods=1).mean()), hh['sal_lp'])
        r0, _, _ = r_(np.log10(q), hh['sal_lp'])
        sens.append(dict(param=k, value=v, r_Q=r0, r_Qvol7d=r1, Qmax=round(float(q.max()), 0)))
A.P.update(base)

# ---- 事件歷線：8/18–8/31 逐時 ----
ev = h.loc['2026-08-17':'2026-08-31 23:00', ['rain', 'Q', 'sal', 'sal_lp']].round(3)
event = dict(hour=[t.strftime('%m-%d %H') for t in ev.index], **{c: [None if pd.isna(v) else float(v) for v in ev[c]] for c in ev.columns})
# metrics for event
pre = h.loc['2026-08-17':'2026-08-19 23:00', 'sal_lp'].mean()
post = h.loc['2026-08-20':'2026-08-31 23:00', 'sal_lp']
qpk_t = h.loc['2026-08-20':'2026-08-31', 'Q'].idxmax(); smin_t = post.idxmin()
event_m = dict(rain=round(float(h.loc['2026-08-20':'2026-08-26 23:00', 'rain'].sum()), 1), Qpeak=round(float(h.loc['2026-08-20':'2026-08-31', 'Q'].max()), 0),
               Qpeak_t=qpk_t.strftime('%m-%d %H:00'), sal_pre=round(float(pre), 2), sal_min=round(float(post.min()), 2),
               sal_min_t=smin_t.strftime('%m-%d %H:00'), drop=round(float(pre - post.min()), 2),
               lag_h=int((smin_t - qpk_t).total_seconds() / 3600),
               vol_event=round(float(h.loc['2026-08-20':'2026-08-31 23:00', 'Q'].sum() * 3600 / 1e6), 1))

# ---- 布袋對照（相同方法，逐時低通）----
bd = load_station('布袋').copy(); bd.loc[(bd['鹽度'] <= 25) | (bd['鹽度'] > 36) | (bd['導電度'] <= 5), '鹽度'] = np.nan
mb = A.mb; qb = A.hbv(mb['rain_filled'], mb['temp'], 474.74)['Q']
sb = godin(bd.set_index('dt')['鹽度'].resample('h').mean().reindex(H).interpolate(limit=3))
hb = pd.DataFrame({'logQv': np.log10(qb.rolling(168, min_periods=1).mean()), 'logQ': np.log10(qb), 'sal_lp': sb}).loc['2026-05-01':'2026-07-21 23:00']
xcb = [dict(lag=L, r=r_(hb['logQ'].shift(L), hb['sal_lp'])[0]) for L in range(0, 241, 6)]
cmp_bd = dict(r_Qvol7d=r_(hb['logQv'], hb['sal_lp'])[0], neff=neff(hb['logQv'], hb['sal_lp']), xc_Q=xcb,
              best=min([o for o in xcb if o['r'] is not None], key=lambda o: o['r']))

out = dict(periods=res, sens=sens, event=event, event_m=event_m, budai_compare=cmp_bd,
           note='逐時鹽度以 Godin 24-24-25 h 濾波去除潮汐；p 值以 Bretherton 有效樣本數修正自相關')
json.dump(out, open(os.path.join(OUT, 'puzi_deep.json'), 'w', encoding='utf-8'), ensure_ascii=False, default=float)
for pk, o in res.items():
    print('==', pk, 'bestQ', o['best_Q'], 'bestR', o['best_R'])
    print(' daily', [(o['lag'],o['r'],round(o['p'],4),o['rR'],o['neff']) for o in o['daily_lag']]); print(' chain', o['chain']); print(' reg', o.get('reg_tide'))
print('sens', [(s['param'], s['value'], s['r_Q'], s['r_Qvol7d']) for s in sens])
print('event', event_m); print('budai', {k: v for k, v in cmp_bd.items() if k != 'xc_Q'})
