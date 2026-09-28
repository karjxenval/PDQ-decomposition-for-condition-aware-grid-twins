from urllib.parse import urljoin,urlparse
from pathlib import Path
import re
from bs4 import BeautifulSoup
from .core import utc_now

SHEET_EXTS=('.xls','.xlsx','.xlsm','.csv')

def _md5(s): return s.split(':',1)[1] if s and s.lower().startswith('md5:') else None

def zenodo(dl,sid,src,profile,dry=False):
    meta=dl.get_json(f"https://zenodo.org/api/records/{src['record_id']}"); dl.save_metadata(sid,meta); out=[]
    for f in meta.get('files',[]):
        url=f.get('links',{}).get('content') or f.get('links',{}).get('self')
        if url: out.append(dl.download(sid,url,f.get('key') or f.get('filename'),_md5(f.get('checksum')),f.get('size'),dry))
    return out

def era(dl,sid,src,profile,dry=False):
    found={}; pages=[]
    for page in src.get('pages',[]):
        try: html=dl.get_text(page)
        except Exception as e: pages.append({'page':page,'error':repr(e)}); continue
        soup=BeautifulSoup(html,'html.parser'); links=[]
        for a in soup.find_all('a',href=True):
            href=urljoin(page,a['href']); clean=href.split('?')[0].lower()
            if clean.endswith(SHEET_EXTS):
                name=Path(urlparse(href).path).name; found[href]=name; links.append(href)
        pages.append({'page':page,'spreadsheet_links':links})
    dl.save_metadata(sid,{'source':src,'discovered_at_utc':utc_now(),'pages':pages,'files':[{'url':u,'name':n} for u,n in found.items()]})
    return [dl.download(sid,u,n,dry_run=dry) for u,n in sorted(found.items())]

def lbnl(dl,sid,src,profile,dry=False):
    soup=BeautifulSoup(dl.get_text(src['index']),'html.parser'); pat=src['full_regex'] if profile=='full' else src['extended_regex']; rx=re.compile(pat); found=[]
    for a in soup.find_all('a',href=True):
        name=a.get_text(' ',strip=True) or Path(urlparse(a['href']).path).name
        if rx.search(name): found.append((urljoin(src['index'],a['href']),name))
    dl.save_metadata(sid,{'source':src,'profile':profile,'files':[{'url':u,'name':n} for u,n in found]})
    return [dl.download(sid,u,n,dry_run=dry) for u,n in found]

def _eth_bits(dl,item):
    base='https://www.research-collection.ethz.ch/server/api'; url=f'{base}/core/items/{item}/bundles?size=100'; data=dl.get_json(url); out=[]
    for b in data.get('_embedded',{}).get('bundles',[]):
        href=b.get('_links',{}).get('bitstreams',{}).get('href'); bundle=(b.get('name') or '').upper()
        if not href: continue
        bd=dl.get_json(href+('&' if '?' in href else '?')+'size=100')
        for bit in bd.get('_embedded',{}).get('bitstreams',[]):
            content=bit.get('_links',{}).get('content',{}).get('href')
            if content: out.append({'bundle':bundle,'name':bit.get('name') or 'bitstream.bin','size':bit.get('sizeBytes') or bit.get('size'),'url':content})
    return out

def eth(dl,sid,src,profile,dry=False):
    try: bits=_eth_bits(dl,src['item_uuid']); mode='dspace_rest'
    except Exception as e:
        bits=[]; mode=f'html_fallback_after_{type(e).__name__}'; soup=BeautifulSoup(dl.get_text(src['landing']),'html.parser')
        for a in soup.find_all('a',href=True):
            href=urljoin(src['landing'],a['href'])
            if '/bitstreams/' in href or '/bitstream/' in href: bits.append({'bundle':'UNKNOWN','name':a.get_text(' ',strip=True) or Path(urlparse(href).path).name,'size':None,'url':href})
    pat=src['core_name_regex'] if profile=='core' else src['structural_name_regex'] if profile=='structural' else src['full_name_regex']; rx=re.compile(pat)
    selected=[b for b in bits if rx.search(b['name']) and b.get('bundle','ORIGINAL') in ('ORIGINAL','UNKNOWN','')]
    dl.save_metadata(sid,{'source':src,'profile':profile,'discovery_mode':mode,'bitstreams':bits,'selected':selected})
    return [dl.download(sid,b['url'],b['name'],expected_size=int(b['size']) if str(b.get('size','')).isdigit() else None,dry_run=dry) for b in selected]

def manual(dl,sid,src,profile,dry=False):
    dl.save_metadata(sid,{'source':src,'status':'manual_access_or_terms_sensitive','instruction':'Use landing page and source terms. Do not bypass access controls or redistribution restrictions.'}); return []

ADAPTERS={'zenodo':zenodo,'era':era,'lbnl_index':lbnl,'eth_dspace':eth,'manual':manual}
