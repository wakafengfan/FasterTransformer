

import json
from pathlib import Path


data = json.load(Path('res.json').open())
e_cnt = 0
ne_l = [] 
for dic in data:
    ft = dic['ft_res']
    ft_preds, ft_scores = zip(*sorted(map(lambda x: (x[0], float(x[1])/max(1, len(x[0]))), zip(ft['preds'], ft['scores'])), key=lambda x: x[1], reverse=True))
    dic['ft_res']['preds'] = ft_preds
    dic['ft_res']['scores'] = ft_scores

    if dic['pred'] != ft_preds[0]:
        ne_l.append(dic)
    else:
        e_cnt += 1

print(f'{e_cnt / len(data)}')

json.dump(ne_l, Path('res_ne.json').open('w'), ensure_ascii=False, indent=2)

