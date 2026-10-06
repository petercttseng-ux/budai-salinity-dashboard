import pandas as pd, numpy as np, math, json, os
from scipy import stats
OUT=os.path.dirname(os.path.abspath(__file__))
def neff(x,y):
    m=x.notna()&y.notna(); x,y=x[m],y[m]; a=x.autocorr(1); b=y.autocorr(1)
    return max(3,int(len(x)*(1-a*b)/(1+a*b)))
def test(x,y):
    m=x.notna()&y.notna(); r=float(stats.pearsonr(x[m],y[m])[0]); n=int(m.sum()); ne=neff(x,y)
    t=r*math.sqrt((ne-2)/(1-r*r)); return dict(r=round(r,3),n=n,neff=ne,p_naive=float(stats.pearsonr(x[m],y[m])[1]),p_eff=float(2*(1-stats.t.cdf(abs(t),ne-2))))
def pc_resid(x,y,z):
    m=x.notna()&y.notna()&z.notna(); x,y,z=x[m],y[m],z[m]
    rx=x-np.polyval(np.polyfit(z,x,1),z); ry=y-np.polyval(np.polyfit(z,y,1),z)
    return test(pd.Series(rx.values),pd.Series(ry.values))
res={}
for st,f,sub in [('budai','daily_budai_v2.csv',None),('dongshi','daily_dongshi_v2.csv',None),('dongshi_after0722','daily_dongshi_v2.csv','2026-07-22')]:
    d=pd.read_csv(os.path.join(OUT,f),parse_dates=['date'])
    if sub: d=d[d.date>=sub]
    d=d[d.sal_mean.notna()].set_index('date').asfreq('D')
    dQ=d['logQ'].diff(); dS=d['sal_mean'].diff()
    r={'日降雨量':test(d['rain'],d['sal_mean']),'7日累積雨量':test(d['rain7'],d['sal_mean']),'log₁₀徑流量':test(d['logQ'],d['sal_mean']),
       'log₁₀徑流量（落後4日）':test(d['logQ'].shift(4),d['sal_mean']),
       '一階差分 Δlog₁₀Q → Δ鹽度':test(dQ,dS),'一階差分 Δlog₁₀Q（落後4日）':test(dQ.shift(4),dS),
       'log₁₀Q｜潮差':pc_resid(d['logQ'],d['sal_mean'],d['tide_rng']),'7日雨量｜潮差':pc_resid(d['rain7'],d['sal_mean'],d['tide_rng'])}
    res[st]=r
    print('==',st)
    for k,v in r.items(): print(f"  {k}: r={v['r']} n={v['n']} neff={v['neff']} p_naive={v['p_naive']:.3g} p_eff={v['p_eff']:.3g}")
json.dump(res,open(os.path.join(OUT,'neff_check.json'),'w',encoding='utf-8'),ensure_ascii=False)
