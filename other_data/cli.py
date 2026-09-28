from pathlib import Path
import argparse,json,traceback,sys
from .core import Downloader,utc_now
from .adapters import ADAPTERS

def main(argv=None):
    ap=argparse.ArgumentParser(description='Acquire public real-world datasets for DCE validation')
    ap.add_argument('--profile',choices=['core','structural','extended','full'],default='core'); ap.add_argument('--root',type=Path,default=Path('data')); ap.add_argument('--max-gb',type=float,default=5.0); ap.add_argument('--source',action='append',default=[]); ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--catalog',type=Path)
    a=ap.parse_args(argv); here=Path(__file__).resolve().parents[1]; cp=a.catalog or here/'configs'/'sources.json'; cat=json.loads(cp.read_text(encoding='utf-8')); a.root.mkdir(parents=True,exist_ok=True); (a.root/'provenance').mkdir(exist_ok=True); (a.root/'provenance'/'source_catalog.json').write_text(json.dumps(cat,indent=2),encoding='utf-8'); dl=Downloader(a.root,a.max_gb)
    selected=[(sid,s) for sid,s in cat.items() if a.profile in s.get('profiles',[]) and (not a.source or sid in a.source)]; failures=[]
    for sid,src in selected:
        print(f"\n=== {sid}: {src.get('title',sid)} ===",flush=True)
        try:
            rs=ADAPTERS[src['adapter']](dl,sid,src,a.profile,a.dry_run)
            for r in rs: print(f"[{r.status:>20}] {r.bytes/1024**2:9.1f} MB  {r.path}")
            if not rs: print('[metadata/manual] no automatic payload downloaded')
        except Exception as e:
            failures.append({'source':sid,'error':repr(e),'traceback':traceback.format_exc()}); print(f'[FAILED] {sid}: {e}',file=sys.stderr)
    (a.root/'provenance'/'run_summary.json').write_text(json.dumps({'timestamp_utc':utc_now(),'profile':a.profile,'failures':failures},indent=2),encoding='utf-8')
    return 1 if failures else 0
if __name__=='__main__': raise SystemExit(main())
