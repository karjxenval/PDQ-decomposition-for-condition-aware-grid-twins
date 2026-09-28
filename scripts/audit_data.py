from pathlib import Path
import argparse,hashlib,json
import pandas as pd
CHUNK=8*1024*1024

def sha(p):
    h=hashlib.sha256(); f=p.open('rb')
    with f:
        for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('data')); a=ap.parse_args(); raw=a.root/'raw'; out=a.root/'audit'; out.mkdir(parents=True,exist_ok=True); rows=[]
    for p in sorted(raw.rglob('*')):
        if not p.is_file() or p.name.endswith('.part'):continue
        r={'relative_path':str(p.relative_to(a.root)),'source_id':p.relative_to(raw).parts[0],'bytes':p.stat().st_size,'sha256':sha(p),'suffix':p.suffix.lower()}
        try:
            if p.suffix.lower()=='.csv':
                df=pd.read_csv(p,nrows=2000,low_memory=False); r['columns']=list(map(str,df.columns)); r['sample_rows']=len(df); r['sample_missing_fraction']=float(df.isna().mean().mean())
            elif p.suffix.lower() in ('.xls','.xlsx','.xlsm'):r['sheets']=pd.ExcelFile(p).sheet_names
        except Exception as e:r['inspect_error']=repr(e)
        rows.append(r)
    (out/'inventory.json').write_text(json.dumps(rows,indent=2),encoding='utf-8'); pd.DataFrame([{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in r.items()} for r in rows]).to_csv(out/'inventory.csv',index=False); print(f'Wrote {len(rows)} records to {out}')
if __name__=='__main__':main()
