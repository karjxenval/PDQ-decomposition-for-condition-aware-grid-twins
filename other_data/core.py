from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import hashlib, json, os, shutil, re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CHUNK = 8 * 1024 * 1024

def utc_now():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()

def safe_name(name: str) -> str:
    name = name.strip().replace('\\','_').replace('/','_')
    name = re.sub(r'[^A-Za-z0-9._() +\-]+','_',name)
    return name[:240] or 'download.bin'

def hash_file(path: Path, alg='sha256') -> str:
    h = hashlib.new(alg)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()

def make_session():
    s=requests.Session()
    retry=Retry(total=6,connect=6,read=6,status=6,backoff_factor=1.2,
                status_forcelist=(408,425,429,500,502,503,504),
                allowed_methods=frozenset(['GET','HEAD']),respect_retry_after_header=True)
    a=HTTPAdapter(max_retries=retry,pool_connections=8,pool_maxsize=8)
    s.mount('http://',a); s.mount('https://',a)
    s.headers.update({'User-Agent':'DCE-RealData-Acquisition/1.0 academic-reproducibility'})
    return s

@dataclass
class Result:
    source_id:str; url:str; path:str; status:str; bytes:int=0
    sha256:Optional[str]=None; md5:Optional[str]=None; expected_md5:Optional[str]=None
    etag:Optional[str]=None; last_modified:Optional[str]=None; timestamp_utc:Optional[str]=None
    note:Optional[str]=None

class Downloader:
    def __init__(self, root:Path, max_gb=5.0, timeout=(20,180)):
        self.root=Path(root); self.raw=self.root/'raw'; self.meta=self.root/'metadata'; self.prov=self.root/'provenance'
        for d in (self.raw,self.meta,self.prov): d.mkdir(parents=True,exist_ok=True)
        self.session=make_session(); self.timeout=timeout; self.max_bytes=int(max_gb*1024**3); self.planned=0
        self.manifest=self.prov/'download_manifest.jsonl'
    def record(self,r:Result):
        if not r.timestamp_utc: r.timestamp_utc=utc_now()
        with self.manifest.open('a',encoding='utf-8') as f: f.write(json.dumps(asdict(r),ensure_ascii=False)+'\n')
    def get_json(self,url):
        r=self.session.get(url,timeout=self.timeout); r.raise_for_status(); return r.json()
    def get_text(self,url):
        r=self.session.get(url,timeout=self.timeout); r.raise_for_status(); return r.text
    def save_metadata(self,sid,obj):
        p=self.meta/f'{sid}.json'; p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8'); return p
    def remote_info(self,url):
        try:
            r=self.session.head(url,allow_redirects=True,timeout=self.timeout)
            if r.status_code>=400:return {},url
            return dict(r.headers),r.url
        except requests.RequestException:return {},url
    def download(self,sid,url,filename=None,expected_md5=None,expected_size=None,dry_run=False):
        d=self.raw/sid; d.mkdir(parents=True,exist_ok=True)
        headers,final=self.remote_info(url)
        remote_size=expected_size
        if not remote_size:
            cl=headers.get('Content-Length'); remote_size=int(cl) if cl and cl.isdigit() else None
        if filename is None: filename=Path(urlparse(final).path).name or 'download.bin'
        filename=safe_name(filename); dest=d/filename; part=Path(str(dest)+'.part')
        if dest.exists():
            md5=hash_file(dest,'md5') if expected_md5 else None
            ok=(not expected_md5 or md5.lower()==expected_md5.lower()) and (not expected_size or dest.stat().st_size==expected_size)
            if ok:
                r=Result(sid,final,str(dest),'already_present',dest.stat().st_size,hash_file(dest),md5,expected_md5,headers.get('ETag'),headers.get('Last-Modified')); self.record(r); return r
        if remote_size:
            remaining=max(0,remote_size-(part.stat().st_size if part.exists() else 0))
            if self.planned+remaining>self.max_bytes:
                r=Result(sid,final,str(dest),'skipped_size_guard',note=f'Would exceed --max-gb ({self.max_bytes/1024**3:.1f} GB)'); self.record(r); return r
            self.planned+=remaining
            if shutil.disk_usage(d).free < remaining+256*1024**2: raise OSError(f'Insufficient disk space for {filename}')
        if dry_run:
            r=Result(sid,final,str(dest),'dry_run',remote_size or 0,expected_md5=expected_md5,etag=headers.get('ETag'),last_modified=headers.get('Last-Modified')); self.record(r); return r
        start=part.stat().st_size if part.exists() else 0; req={}; mode='wb'
        if start: req['Range']=f'bytes={start}-'; mode='ab'
        with self.session.get(final,stream=True,headers=req,allow_redirects=True,timeout=self.timeout) as resp:
            if start and resp.status_code==200: raise RuntimeError(f'Server refused resume for {filename}; keeping {start} downloaded bytes. Refusing to restart from zero.')
            resp.raise_for_status()
            with part.open(mode) as f:
                for chunk in resp.iter_content(CHUNK):
                    if chunk:f.write(chunk)
                f.flush(); os.fsync(f.fileno())
        if expected_size and part.stat().st_size!=expected_size: raise IOError(f'Size mismatch: {filename}')
        got=hash_file(part,'md5') if expected_md5 else None
        if expected_md5 and got.lower()!=expected_md5.lower(): raise IOError(f'MD5 mismatch: {filename}')
        os.replace(part,dest)
        r=Result(sid,final,str(dest),'downloaded',dest.stat().st_size,hash_file(dest),got,expected_md5,headers.get('ETag'),headers.get('Last-Modified')); self.record(r); return r
