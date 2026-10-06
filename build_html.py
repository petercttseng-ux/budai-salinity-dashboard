import json, os
B = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f'{B}/results_v2.json', encoding='utf-8'))
R['puzi'] = json.load(open(f'{B}/puzi_deep.json', encoding='utf-8'))
R['neff'] = json.load(open(f'{B}/neff_check.json', encoding='utf-8'))
keep = ['date','rain','rain3','rain7','Q','logQ','sal_mean','sal_min','sal_max','ss','tide_rng','wtemp','do']
for s in R['stations'].values():
    s['daily'] = [{k: r.get(k) for k in keep} for r in s['daily']]
data = json.dumps(R, ensure_ascii=False, separators=(',', ':'))
html = open(f'{B}/template.html', encoding='utf-8').read().replace('/*__DATA__*/null', data)
html = '<!doctype html>\n<html lang="zh-Hant">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n' + html + '\n</body>\n</html>\n'
open(f'{B}/index.html', 'w', encoding='utf-8').write(html)
print(len(html))
