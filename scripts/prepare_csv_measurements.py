from pathlib import Path
import argparse,json
import pandas as pd
TIME_HINTS=('time','timestamp','datetime','date','utc','gps')

def detect_time(df):
    for c in df.columns:
        if any(h in str(c).lower() for h in TIME_HINTS):
            t=pd.to_datetime(df[c],errors='coerce',utc=True)
            if t.notna().mean()>=.8:return c,t
    return None,None

def numeric(df):
    out={}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):out[str(c)]=pd.to_numeric(df[c],errors='coerce')
        else:
            x=pd.to_numeric(df[c],errors='coerce')
            if x.notna().mean()>=.95:out[str(c)]=x
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('data')); ap.add_argument('--max-rows',type=int,default=0); a=ap.parse_args(); raw=a.root/'raw'; out=a.root/'derived'/'measurement_csv'; out.mkdir(parents=True,exist_ok=True); rep=[]
    for p in raw.rglob('*.csv'):
        try:
            df=pd.read_csv(p,nrows=(a.max_rows or None),low_memory=False); tc,t=detect_time(df); num=numeric(df)
            if tc is not None:num.insert(0,'timestamp_utc',t)
            target=out/('__'.join(p.relative_to(raw).parts)+'.parquet'); num.to_parquet(target,index=False); rep.append({'source_csv':str(p.relative_to(raw)),'derived':target.name,'rows':len(num),'columns':list(num.columns),'time_column_original':str(tc) if tc is not None else None,'missing_fraction':float(num.isna().mean().mean()) if num.size else None})
        except Exception as e:rep.append({'source_csv':str(p.relative_to(raw)),'error':repr(e)})
    (out/'schema_report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8'); print(f"Processed {sum('derived' in x for x in rep)} CSVs")
if __name__=='__main__':main()
