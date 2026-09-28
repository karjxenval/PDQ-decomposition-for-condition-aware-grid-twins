from __future__ import annotations
import argparse, json, math, os, re, shutil, tempfile, zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import wilcoxon
import scipy.io as sio

# ------------------------------- utilities -------------------------------

def jdump(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def orth(B):
    Q, R = np.linalg.qr(np.asarray(B, float), mode="reduced")
    d = np.sign(np.diag(R)); d[d == 0] = 1
    return Q * d.reshape(1, -1)


def svd_basis(X, r):
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :min(int(r), U.shape[1])]


def chronological_split(X):
    m = X.shape[1]
    a = max(1, int(.60*m)); b = max(a+1, int(.80*m)); b = min(b, m-1)
    return X[:, :a], X[:, a:b], X[:, b:]


class Scale:
    def __init__(self, mode="feature"): self.mode = mode
    def fit(self, X):
        self.mean = np.nanmean(X, axis=1, keepdims=True)
        C = X - self.mean
        if self.mode == "global":
            s = float(np.nanstd(C)); self.scale = np.full((X.shape[0],1), max(s,1e-12))
        else:
            self.scale = np.nanstd(C, axis=1, keepdims=True)
            self.scale[(~np.isfinite(self.scale)) | (self.scale < 1e-12)] = 1.0
        return self
    def transform(self, X): return (X-self.mean)/self.scale
    def inverse(self, X): return X*self.scale+self.mean


def finite_columns(X): return np.all(np.isfinite(X), axis=0)


def thin(X, max_snapshots):
    if X.shape[1] <= max_snapshots: return X
    idx = np.linspace(0, X.shape[1]-1, max_snapshots, dtype=int)
    return X[:, idx]


def qr_rows(B, s):
    _, _, piv = qr(orth(B).T, pivoting=True, mode="economic")
    return np.asarray(piv[:min(int(s), B.shape[0])], int)


def greedy_dopt_rows(B, s, ridge=1e-10):
    """Nested greedy D-optimal row selection for an orthonormal basis.

    Unlike pivoted QR, this remains meaningful after s exceeds the basis rank.
    Matrix-determinant-lemma gains are b_i^T M^{-1} b_i with
    M = ridge I + sum_{j in S} b_j b_j^T.
    """
    B=orth(B); n,r=B.shape; s=max(1,min(int(s),n))
    scale=max(float(np.mean(np.sum(B*B,axis=1))),1e-12)
    lam=max(float(ridge)*scale,1e-14)
    Minv=np.eye(r)/lam
    chosen=[]; available=np.ones(n,dtype=bool)
    for _ in range(s):
        # diag(B Minv B^T), computed without forming n x n matrix.
        gains=np.einsum('ij,jk,ik->i',B,Minv,B,optimize=True)
        gains[~available]=-np.inf
        i=int(np.argmax(gains)); chosen.append(i); available[i]=False
        b=B[i,:].reshape(-1,1)
        den=float(1.0+(b.T@Minv@b)[0,0])
        Minv=Minv-(Minv@b@b.T@Minv)/den
    return np.asarray(chosen,dtype=int)


def ridge_map(B, idx, gamma):
    B = orth(B); idx = np.asarray(idx, int)
    G = B[idx, :]
    H = G.T@G + float(gamma)*np.eye(B.shape[1])
    C = np.linalg.solve(H, G.T)
    T = B@C
    return T, G


def recover(B, idx, X, gamma):
    T, G = ridge_map(B, idx, gamma)
    return T@X[idx,:], T, G


def relerr(Xh, X):
    return np.linalg.norm(Xh-X, axis=0) / np.maximum(np.linalg.norm(X,axis=0), 1e-12)


def projection_error(B, X):
    B = orth(B); R = X - B@(B.T@X)
    return float(np.linalg.norm(R,"fro")/max(np.linalg.norm(X,"fro"),1e-12))


def diagnostics(B, idx, gamma):
    B = orth(B); T, G = ridge_map(B, idx, gamma)
    s = np.linalg.svd(G, compute_uv=False)
    rank = int(np.linalg.matrix_rank(G)); r = B.shape[1]
    smax = float(s[0]) if len(s) else 0.0
    # For a rectangular rank-deficient G, numpy returns only min(m,r)
    # singular values.  The full latent map still has r-rank zero singular
    # directions, so sigma_min(G) is exactly zero whenever rank < r.
    smin_observed = float(s[-1]) if len(s) else 0.0
    smin = smin_observed if rank == r else 0.0
    return {
        "rank_G": rank, "r": r, "nullity": int(r-rank),
        "sigma_min": smin, "sigma_min_observed": smin_observed,
        "sigma_max": smax,
        "kappa": float(smax/smin) if rank == r and smin > 0 else None,
        "exact_amplification": float(np.linalg.norm(T, 2)),
        "ridge_bias_factor": float(gamma/(smin*smin+gamma)) if gamma > 0 else (0.0 if rank == r else 1.0)
    }


def bootstrap_mean_ci(x, reps, seed):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if not len(x): return [None,None]
    rng=np.random.default_rng(seed); vals=np.empty(reps)
    for k in range(reps): vals[k]=np.mean(rng.choice(x,len(x),replace=True))
    return [float(np.quantile(vals,.025)), float(np.quantile(vals,.975))]


def paired(a,b,reps,seed):
    a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b)
    a=a[m]; b=b[m]; d=a-b
    try:
        st,p=wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        st=float(st); p=float(p)
    except Exception: st=p=None
    return {
        "n": int(len(d)), "mean_svd": float(np.mean(a)), "mean_rasd": float(np.mean(b)),
        "mean_difference_svd_minus_rasd": float(np.mean(d)),
        "difference_95ci": bootstrap_mean_ci(d,reps,seed),
        "relative_change_percent": float(100*np.mean(d)/np.mean(a)) if np.mean(a)>0 else None,
        "wilcoxon_stat": st, "wilcoxon_p": p
    }


def tune_gamma(B, idx, Xv, grid):
    G=orth(B)[np.asarray(idx),:]; s=np.linalg.svd(G,compute_uv=False)
    base=float(s[0]**2) if len(s) and s[0]>0 else 1.0
    best=None
    for rel in grid:
        gam=float(rel)*base; Xh,_,_=recover(B,idx,Xv,gam)
        loss=float(np.mean(relerr(Xh,Xv)))
        if best is None or loss<best[0]: best=(loss,gam,float(rel))
    return {"val_error":best[0],"gamma":best[1],"gamma_rel":best[2]}


# ---------------------- recovery-aware basis optimizer --------------------

def rasd(X, idx, r, gamma, max_iter=100, tol=1e-7):
    """Minimize empirical sparse-recovery loss on B^T B=I, initialized at POD."""
    import autograd.numpy as anp
    from autograd import grad
    X=np.asarray(X,float); n=X.shape[0]; r=min(int(r),n,X.shape[1]); idx=np.asarray(idx,int)
    B=svd_basis(X,r); Xa=anp.array(X); denom=float(np.sum(X*X))+1e-30
    def fflat(v):
        Bb=anp.reshape(v,(n,r)); G=Bb[idx,:]
        H=anp.dot(G.T,G)+gamma*anp.eye(r)
        C=anp.linalg.solve(H,G.T)
        E=anp.dot(Bb,anp.dot(C,Xa[idx,:]))-Xa
        return anp.sum(E*E)/denom
    gf=grad(fflat); f=float(fflat(B.ravel())); step=1.0; hist=[]
    for it in range(int(max_iter)):
        ge=np.asarray(gf(B.ravel())).reshape(B.shape)
        sym=.5*(B.T@ge+ge.T@B); gt=ge-B@sym; ng=float(np.linalg.norm(gt,"fro"))
        hist.append({"iter":it,"objective":f,"grad_norm":ng,"step":step})
        if ng<tol: break
        accepted=False; local=step
        for _ in range(20):
            Bt=orth(B-local*gt); ft=float(fflat(Bt.ravel()))
            if np.isfinite(ft) and ft <= f-1e-4*local*ng*ng:
                B,f=Bt,ft; step=min(2.0,1.5*local); accepted=True; break
            local*=.5
        if not accepted: break
    return orth(B), hist


# ------------------------------- Ponte -----------------------------------

def load_ponte(data_root):
    zpath=Path(data_root)/"raw"/"ponte_moesa_2025"/"Strain Data.zip"
    if not zpath.exists(): raise FileNotFoundError(zpath)
    with zipfile.ZipFile(zpath) as z:
        member=next(n for n in z.namelist() if n.lower().endswith(".mat"))
        with tempfile.NamedTemporaryFile(suffix=".mat",delete=False) as t:
            shutil.copyfileobj(z.open(member),t); tmp=t.name
    try: M=sio.loadmat(tmp,struct_as_record=False,squeeze_me=False)["M2"]
    finally:
        try: os.unlink(tmp)
        except OSError: pass
    rec=[]
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            cell=M[i,j]
            if not isinstance(cell,np.ndarray) or cell.size!=1: continue
            st=cell.flat[0]
            if not getattr(st,"_fieldnames",None): continue
            x=np.asarray(st.x).squeeze().astype(float).ravel()
            A=np.asarray(st.strain).squeeze().astype(float)
            if A.ndim==1:
                if len(A)!=len(x): raise ValueError(f"cell {i},{j}: x={len(x)} strain={A.shape}")
                A=A[:,None]
            elif A.ndim==2 and A.shape[0]==len(x): pass
            elif A.ndim==2 and A.shape[1]==len(x): A=A.T
            else: raise ValueError(f"cell {i},{j}: no strain axis matches len(x)={len(x)}; shape={A.shape}")
            rec.append({"name":f"cell_r{i}_c{j}","row":i,"col":j,"x":x,"X":A})
    return rec


def evaluate_field(X, positions, cfg, seed, scale_mode="global"):
    """Evaluate an already quality-controlled finite spatial field.

    Publication sensor budgets are absolute counts (not large percentages of a
    dense fibre field).  Sensors are selected from the TRAINING POD basis using
    a nested greedy D-optimal criterion and then held fixed for SVD-vs-RASD.
    """
    X=np.asarray(X,float); positions=np.asarray(positions,float)
    if X.ndim!=2 or X.shape[0]!=len(positions):
        raise ValueError(f"Field/position mismatch: X={X.shape}, positions={len(positions)}")
    if not np.all(np.isfinite(X)):
        raise ValueError("evaluate_field requires a finite field; quality-control missing values first")
    if X.shape[1] < 10 or X.shape[0] < 2:
        raise ValueError(f"Too little finite data after quality control: {X.shape}")

    X=thin(X,cfg["max_snapshots"]); Xtr0,Xv0,Xte0=chronological_split(X)
    tr_std=np.std(Xtr0,axis=1); good=np.isfinite(tr_std)&(tr_std>1e-12)
    if np.count_nonzero(good)<2: raise ValueError("Fewer than two nonconstant spatial channels in training data")
    Xtr0,Xv0,Xte0=Xtr0[good],Xv0[good],Xte0[good]; pos=positions[good]
    sc=Scale(scale_mode).fit(Xtr0); Xtr,Xv,Xte=[sc.transform(Z) for Z in (Xtr0,Xv0,Xte0)]
    n=Xtr.shape[0]; out=[]

    if "ponte_sensor_counts" in cfg:
        budgets=sorted(set(max(1,min(n,int(s))) for s in cfg["ponte_sensor_counts"]))
    else:
        budgets=sorted(set(max(1,min(n,int(round(float(f)*n)))) for f in cfg["sensor_fractions"]))

    for s in budgets:
        best=None; gridrows=[]
        for r in cfg["ranks"]:
            if r>min(Xtr.shape): continue
            B=svd_basis(Xtr,r); idx=greedy_dopt_rows(B,s)
            tg=tune_gamma(B,idx,Xv,cfg["gamma_rel_grid"])
            row={"r":r,"idx":idx,"gamma":tg["gamma"],"val":tg["val_error"],"gamma_rel":tg["gamma_rel"]}
            gridrows.append({"r":r,"val_error":tg["val_error"],"gamma":tg["gamma"],"gamma_rel":tg["gamma_rel"],
                             "rank_G":diagnostics(B,idx,tg["gamma"])["rank_G"]})
            if best is None or row["val"]<best["val"]: best=row
        if best is None: raise ValueError(f"No admissible rank for field {Xtr.shape}")
        r=best["r"]; idx=best["idx"]; B0=svd_basis(Xtr,r)
        Xs,_,_=recover(B0,idx,Xte,best["gamma"]); es=relerr(sc.inverse(Xs),Xte0)
        Br,hist=rasd(Xtr,idx,r,best["gamma"],cfg["rasd_max_iter"])
        gr=tune_gamma(Br,idx,Xv,cfg["gamma_rel_grid"])
        Xr,_,_=recover(Br,idx,Xte,gr["gamma"]); er=relerr(sc.inverse(Xr),Xte0)
        out.append({
            "sensor_count":int(s),"sensor_fraction":float(s/n),"n_channels":int(n),"rank":int(r),
            "sensor_indices":idx.tolist(),"sensor_positions":[float(pos[k]) for k in idx],
            "sensor_design":"training-only nested greedy D-optimal rows on POD basis; fixed for SVD/RASD comparison",
            "svd":{"gamma":best["gamma"],"validation_relerr":best["val"],
                   "mean_relerr":float(np.mean(es)),"median_relerr":float(np.median(es)),
                   "error_95ci":bootstrap_mean_ci(es,cfg["bootstrap_reps"],seed),
                   "projection_error_normalized":projection_error(B0,Xte),"diagnostics":diagnostics(B0,idx,best["gamma"])},
            "rasd":{"gamma":gr["gamma"],"validation_relerr":gr["val_error"],
                    "mean_relerr":float(np.mean(er)),"median_relerr":float(np.median(er)),
                    "error_95ci":bootstrap_mean_ci(er,cfg["bootstrap_reps"],seed+1),
                    "projection_error_normalized":projection_error(Br,Xte),"diagnostics":diagnostics(Br,idx,gr["gamma"]),
                    "iterations":len(hist),"final_train_objective":hist[-1]["objective"] if hist else None},
            "paired":paired(es,er,cfg["bootstrap_reps"],seed),"rank_gamma_grid":gridrows
        })
    return out


def _ponte_complete_domain(X, x, max_channels=None):
    """Construct an imputation-free spatial benchmark domain.

    We keep only fibre coordinates observed at every snapshot in the record.
    This uses only the observation mask, never held-out strain values.  If the
    domain is very dense, it is uniformly thinned in physical x so the quick
    experiment remains computationally tractable while retaining full span.
    """
    X=np.asarray(X,float); x=np.asarray(x,float).ravel()
    complete=np.all(np.isfinite(X),axis=1)
    idx=np.flatnonzero(complete)
    if len(idx)<2:
        raise ValueError("Fewer than two continuously observed fibre coordinates")
    # Spatially order the complete channels before any deterministic thinning.
    idx=idx[np.argsort(x[idx],kind="mergesort")]
    if max_channels is not None and len(idx)>int(max_channels):
        take=np.linspace(0,len(idx)-1,int(max_channels),dtype=int)
        idx=idx[take]
    return X[idx,:],x[idx],complete,idx


def run_ponte(data_root,cfg,outdir,seed):
    records=load_ponte(data_root); manifest=[]; allres=[]
    print(f"Ponte: {len(records)} non-empty records")
    max_channels=int(cfg.get("ponte_max_channels",2500))
    for k,r in enumerate(records):
        Xraw=r["X"]
        finite_pct=100.0*float(np.isfinite(Xraw).mean())
        full_rows=int(np.all(np.isfinite(Xraw),axis=1).sum())
        print(r["name"],"x=",len(r["x"]),"strain=",Xraw.shape,
              "finite=",round(finite_pct,2),"%","complete_spatial_rows=",full_rows)
        base={"name":r["name"],"row":r["row"],"col":r["col"],
              "x_len":len(r["x"]),"strain_shape":list(Xraw.shape),
              "finite_percent":finite_pct,"complete_spatial_rows":full_rows}
        try:
            X,x,mask,idx=_ponte_complete_domain(Xraw,r["x"],max_channels=max_channels)
            base.update({"benchmark_channels":int(X.shape[0]),
                         "benchmark_snapshots":int(X.shape[1]),
                         "benchmark_fraction_of_original_space":float(X.shape[0]/Xraw.shape[0]),
                         "quality_control":"coordinates finite at every snapshot; no value imputation; deterministic uniform thinning in x if needed"})
            print("  -> benchmark field",X.shape,"(imputation-free)")
            rr=evaluate_field(X,x,cfg,seed+k,"global")
            allres.append({"name":r["name"],"status":"ok","quality_control":base,"results":rr})
        except Exception as e:
            allres.append({"name":r["name"],"status":"failed","quality_control":base,"error":repr(e)})
        manifest.append(base)
    jdump(manifest,outdir/"ponte_manifest.json"); jdump(allres,outdir/"ponte_results.json")


# ------------------------------ GridEye ----------------------------------
PAIR=re.compile(r"^(Mag|Angle)_([VI][ABC])_(\d+_\d+)$")

def zip_csv(zpath,member,stride,max_rows):
    parts=[]; n=0
    with zipfile.ZipFile(zpath) as z, z.open(member) as f:
        for ch in pd.read_csv(f,chunksize=50000,low_memory=False):
            if stride>1: ch=ch.iloc[::stride,:]
            if max_rows:
                ch=ch.iloc[:max(0,max_rows-n),:]
            if len(ch): parts.append(ch); n+=len(ch)
            if max_rows and n>=max_rows: break
    return pd.concat(parts,ignore_index=True)


def phasor_matrix(df):
    cols=set(map(str,df.columns)); feats=[]; names=[]; site_rows={}
    for c in map(str,df.columns):
        m=PAIR.match(c)
        if not m or m.group(1)!="Mag": continue
        q=m.group(2); site=m.group(3); ac=f"Angle_{q}_{site}"
        if ac not in cols: continue
        mag=pd.to_numeric(df[c],errors="coerce").to_numpy(float); ang=pd.to_numeric(df[ac],errors="coerce").to_numpy(float)
        rad=np.deg2rad(ang)
        for nm,a in [(f"Re_{q}_{site}",mag*np.cos(rad)),(f"Im_{q}_{site}",mag*np.sin(rad))]:
            row=len(feats); feats.append(a); names.append(nm); site_rows.setdefault(site,[]).append(row)
    X=np.vstack(feats)
    groups=[np.asarray(site_rows[s],int) for s in sorted(site_rows)]; sites=sorted(site_rows)
    return X,names,groups,sites


def exact_site_set(Xtr,groups,k,r):
    B=svd_basis(Xtr,r); best=None
    for comb in combinations(range(len(groups)),k):
        idx=np.concatenate([groups[j] for j in comb]); G=B[idx,:]; s=np.linalg.svd(G,compute_uv=False)
        gam=1e-6*((s[0]**2) if len(s) and s[0]>0 else 1.0)
        Xh,_,_=recover(B,idx,Xtr,gam); loss=float(np.mean(relerr(Xh,Xtr)))
        if best is None or loss<best[0]: best=(loss,comb,idx)
    return best[1],best[2]


def grideye_site_drift(Xtr,Xv,Xte,groups,sites):
    """Standardized first/second-moment drift by physical PMU site."""
    out=[]; eps=1e-12
    for g,s in zip(groups,sites):
        mt=np.mean(Xtr[g,:],axis=1); mv=np.mean(Xv[g,:],axis=1); me=np.mean(Xte[g,:],axis=1)
        st=np.std(Xtr[g,:],axis=1)+eps; sv=np.std(Xv[g,:],axis=1)+eps; se=np.std(Xte[g,:],axis=1)+eps
        out.append({
            "site":s,
            "noon_mean_shift_rms":float(np.sqrt(np.mean((mv-mt)**2))),
            "evening_mean_shift_rms":float(np.sqrt(np.mean((me-mt)**2))),
            "noon_log_std_ratio_rms":float(np.sqrt(np.mean(np.log(sv/st)**2))),
            "evening_log_std_ratio_rms":float(np.sqrt(np.mean(np.log(se/st)**2)))
        })
    return out


def compatible_grideye_events(data_root,target_names,cfg):
    """Load only event files having the complete steady-state phasor feature set."""
    root=Path(data_root)/"raw"/"grideye_real_pmu_2025"; out={}
    for fn in ["Frequency events.zip","Voltage events.zip"]:
        zp=root/fn
        if not zp.exists(): continue
        with zipfile.ZipFile(zp) as z:
            members=[n for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/') and '/._' not in n]
        for mem in members:
            try:
                df=zip_csv(zp,mem,max(1,int(cfg["grideye_stride"])),cfg["max_snapshots"])
                X,names,_,_=phasor_matrix(df); pos={n:i for i,n in enumerate(names)}
                if not all(n in pos for n in target_names): continue
                X=X[[pos[n] for n in target_names],:]; X=X[:,finite_columns(X)]
                if X.shape[1]>=10: out[f"{fn}::{Path(mem).name}"]=X
            except Exception:
                continue
    return out


def nested_site_path(Xtr,groups,sites,r,max_k):
    """Training-only nested site path minimizing POD training recovery loss."""
    B=svd_basis(Xtr,r); selected=[]; order=[]
    for _ in range(min(max_k,len(groups))):
        best=None
        for j in range(len(groups)):
            if j in selected: continue
            cand=selected+[j]; idx=np.concatenate([groups[q] for q in cand])
            G=B[idx,:]; ss=np.linalg.svd(G,compute_uv=False); base=(ss[0]**2 if len(ss) and ss[0]>0 else 1.0)
            gam=1e-6*base; Xh,_,_=recover(B,idx,Xtr,gam); loss=float(np.mean(relerr(Xh,Xtr)))
            if best is None or loss<best[0]: best=(loss,j)
        selected.append(best[1]); order.append(best[1])
    return order


def run_grideye(data_root,cfg,outdir,seed):
    zpath=Path(data_root)/"raw"/"grideye_real_pmu_2025"/"Steady state data.zip"
    with zipfile.ZipFile(zpath) as z:
        members=[n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/") and "/._" not in n]
    def ok(n):
        s=n.lower(); return 0 if "morning" in s else 1 if "noon" in s else 2 if "evening" in s else 3
    members=sorted(members,key=lambda n:(ok(n),n)); regs=[]
    for mbr in members:
        df=zip_csv(zpath,mbr,cfg["grideye_stride"],cfg["max_snapshots"])
        X,names,groups,sites=phasor_matrix(df); X=X[:,finite_columns(X)]
        regs.append((mbr,X,names,groups,sites)); print("GridEye",mbr,"field",X.shape,"sites",sites)
    if len(regs)<3: raise RuntimeError("Need morning/noon/evening steady-state files")
    tr,va,te=regs[:3]
    if tr[2]!=va[2] or tr[2]!=te[2]: raise RuntimeError("GridEye feature headers differ across regimes")
    Xtr0,Xv0,Xte0=tr[1],va[1],te[1]; sc=Scale("feature").fit(Xtr0)
    Xtr,Xv,Xte=[sc.transform(Z) for Z in (Xtr0,Xv0,Xte0)]
    groups,sites=tr[3],tr[4]; target_names=tr[2]
    drift=grideye_site_drift(Xtr,Xv,Xte,groups,sites)
    raw_events=compatible_grideye_events(data_root,target_names,cfg)
    events={name:(Xe,sc.transform(Xe)) for name,Xe in raw_events.items()}

    site_counts=cfg.get("grideye_site_counts")
    if site_counts is None:
        site_counts=sorted(set(max(1,min(len(groups),int(round(f*len(groups))))) for f in cfg["sensor_fractions"]))
    else:
        site_counts=sorted(set(max(1,min(len(groups)-1,int(k))) for k in site_counts))

    results=[]
    for k in site_counts:
        best=None
        for r in cfg["ranks"]:
            if r>min(Xtr.shape): continue
            comb,idx=exact_site_set(Xtr,groups,k,r); B=svd_basis(Xtr,r); tg=tune_gamma(B,idx,Xv,cfg["gamma_rel_grid"])
            row=(tg["val_error"],r,comb,idx,tg)
            if best is None or row[0]<best[0]: best=row
        _,r,comb,idx,tg=best; B0=svd_basis(Xtr,r)
        Xs,_,_=recover(B0,idx,Xte,tg["gamma"]); es=relerr(sc.inverse(Xs),Xte0)
        Br,hist=rasd(Xtr,idx,r,tg["gamma"],cfg["rasd_max_iter"]); gr=tune_gamma(Br,idx,Xv,cfg["gamma_rel_grid"])
        Xr,_,_=recover(Br,idx,Xte,gr["gamma"]); er=relerr(sc.inverse(Xr),Xte0)
        event_res={}
        for ename,(Xe0,Xe) in events.items():
            Xes,_,_=recover(B0,idx,Xe,tg["gamma"]); ee_s=relerr(sc.inverse(Xes),Xe0)
            Xer,_,_=recover(Br,idx,Xe,gr["gamma"]); ee_r=relerr(sc.inverse(Xer),Xe0)
            event_res[ename]={"n":int(Xe0.shape[1]),"svd_mean_relerr":float(np.mean(ee_s)),
                              "rasd_mean_relerr":float(np.mean(ee_r)),
                              "paired":paired(ee_s,ee_r,min(cfg["bootstrap_reps"],500),seed)}
        results.append({
            "selected_site_count":int(k),"selected_sites":[sites[j] for j in comb],"rank":int(r),
            "svd":{"gamma":tg["gamma"],"validation_relerr":tg["val_error"],"mean_relerr":float(np.mean(es)),
                   "median_relerr":float(np.median(es)),"diagnostics":diagnostics(B0,idx,tg["gamma"])},
            "rasd":{"gamma":gr["gamma"],"validation_relerr":gr["val_error"],"mean_relerr":float(np.mean(er)),
                    "median_relerr":float(np.median(er)),"diagnostics":diagnostics(Br,idx,gr["gamma"]),"iterations":len(hist)},
            "paired":paired(es,er,cfg["bootstrap_reps"],seed),"event_ood":event_res
        })

    # A separate nested, fixed-rank path isolates the effect of adding sites.
    nested_rank=int(cfg.get("grideye_nested_rank",6)); nested_rank=min(nested_rank,min(Xtr.shape))
    order=nested_site_path(Xtr,groups,sites,nested_rank,max(site_counts))
    Bn=svd_basis(Xtr,nested_rank); nested=[]
    for k in site_counts:
        comb=order[:k]; idx=np.concatenate([groups[j] for j in comb]); tg=tune_gamma(Bn,idx,Xv,cfg["gamma_rel_grid"])
        Xh,_,_=recover(Bn,idx,Xte,tg["gamma"]); e=relerr(sc.inverse(Xh),Xte0)
        nested.append({"site_count":int(k),"sites":[sites[j] for j in comb],"rank":int(nested_rank),
                       "validation_relerr":tg["val_error"],"test_mean_relerr":float(np.mean(e)),
                       "diagnostics":diagnostics(Bn,idx,tg["gamma"])})

    jdump({"protocol":"morning_train_noon_validation_evening_test",
           "members":{"train":tr[0],"validation":va[0],"test":te[0]},"sites":sites,
           "site_drift":drift,"compatible_event_files":list(events.keys()),
           "results":results,"nested_fixed_rank_sensor_path":nested},outdir/"grideye_results.json")


# -------------------------------- CLI ------------------------------------
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("dataset",choices=["ponte","grideye"]); ap.add_argument("--data-root",type=Path,default=Path("data"))
    g=ap.add_mutually_exclusive_group(); g.add_argument("--quick",action="store_true"); g.add_argument("--publication",action="store_true")
    ap.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"configs"/"validation.json"); a=ap.parse_args()
    C=json.loads(a.config.read_text(encoding="utf-8")); mode="publication" if a.publication else "quick"; cfg=C[mode]; seed=int(C["seed"])
    out=Path("results")/a.dataset/mode; out.mkdir(parents=True,exist_ok=True)
    print("Mode:",mode," Data root:",a.data_root.resolve()," Results:",out.resolve())
    if a.dataset=="ponte": run_ponte(a.data_root,cfg,out,seed)
    else: run_grideye(a.data_root,cfg,out,seed)
    print("Finished.")

if __name__=="__main__": main()
