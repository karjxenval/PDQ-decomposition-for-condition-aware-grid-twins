from pathlib import Path
import argparse,re,json
import pandas as pd

def slug(s):return re.sub(r'[^A-Za-z0-9._-]+','_',str(s)).strip('_') or 'sheet'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('data')); a=ap.parse_args(); src=a.root/'raw'/'era_uganda'; out=a.root/'derived'/'era_uganda'; out.mkdir(parents=True,exist_ok=True); rec=[]
    for p in sorted(src.glob('*')):
        if p.suffix.lower() not in ('.xls','.xlsx','.xlsm'):continue
        try:
            x=pd.ExcelFile(p)
            for sh in x.sheet_names:
                df=pd.read_excel(p,sheet_name=sh,header=None); t=out/f'{slug(p.stem)}__{slug(sh)}.csv'; df.to_csv(t,index=False,header=False); rec.append({'source_file':p.name,'sheet':sh,'rows':len(df),'cols':len(df.columns),'derived_file':t.name})
        except Exception as e:rec.append({'source_file':p.name,'error':repr(e)})
    (out/'sheet_inventory.json').write_text(json.dumps(rec,indent=2),encoding='utf-8'); print(f"Prepared {sum('derived_file' in x for x in rec)} sheets")
if __name__=='__main__':main()
