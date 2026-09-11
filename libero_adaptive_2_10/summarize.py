"""Summarize completed task results; NFE is weighted by action chunks."""
import argparse,csv,json
from pathlib import Path
from runtime import write_json


def summarize(output):
    output=Path(output);groups={};suites={}
    for p in sorted((output/'tasks').glob('*/result.json')):
        r=json.loads(p.read_text());name=r['method']['name']
        for table,key in [(groups,name),(suites,(name,r['suite']))]:
            a=table.setdefault(key,dict(method=name,suite=r['suite'] if table is suites else 'all',tasks=0,episodes=0,successes=0,chunks=0,forwards=0))
            a['tasks']+=1
            for dst,src in [('episodes','total_episodes'),('successes','successes'),('chunks','chunks'),('forwards','forwards')]:a[dst]+=r[src]
    for table in [groups,suites]:
        for a in table.values():
            a['success_rate']=a['successes']/a['episodes'];a['mean_nfe']=a['forwards']/a['chunks']
    for name,table in [('summary',groups),('suite_summary',suites)]:
        rows=list(table.values());write_json(output/f'{name}.json',rows)
        if rows:
            tmp=output/f'{name}.csv.tmp'
            with tmp.open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            tmp.replace(output/f'{name}.csv')
    return list(groups.values())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    print(json.dumps(summarize(parser.parse_args().output),indent=2))
