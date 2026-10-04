"""Shared-feature, cache-aware LOSO execution helpers.

The module is deliberately separate from the audited runner: the legacy
run_single_fold API remains available for regression tests, while the CLI can
use this implementation to fit/transform MiniROCKET once per fold.
"""
import os, json, time, pickle, hashlib, gc
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score, f1_score

import config
from data_loader import load_ninapro_stream, validate_data_path, get_n_classes
from feature_extractors import get_feature_extractor
from subject_split import split_by_repetition
from domain_adaptation import (BatchedStandardScaler, covariance_frobenius,
    mmd_squared_linear, feature_centering, coral_adaptation, tca_transform,
    sa_transform)
from statistical_utils import (per_class_f1, confusion_matrix,
    descriptive_stats, random_baseline_from_labels, ratio_vs_random)

N_RIDGE = 8000


def stratified_subsample(X, y, n_max, seed):
    if len(y) <= n_max:
        return X, y
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        idx.extend(rng.choice(ci, max(1, min(len(ci), int(len(ci)/len(y)*n_max))), replace=False))
    idx = np.asarray(idx); rng.shuffle(idx)
    return X[idx], y[idx]


def atomic_pickle(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def manifest(db, sid, seed, extractor, window_ms, overlap, kernels, calib,
             coral_reg, tca_components, sa_components, fit_mode):
    m = {'database': db, 'test_subject': int(sid), 'seed': int(seed),
         'extractor': extractor, 'window_ms': int(window_ms),
         'overlap': float(overlap), 'num_kernels': int(kernels),
         'calibration_fraction': float(calib), 'coral_reg': float(coral_reg),
         'tca_components': int(tca_components), 'sa_components': int(sa_components),
         'fit_mode': fit_mode, 'pipeline_version': 'shared-v1'}
    m['configuration_hash'] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    return m


def save_features(path, m, arrays):
    tmp = path + '.tmp.npz'
    np.savez(tmp, **{k: np.asarray(v, dtype=np.float32 if k.startswith('X') else np.int32)
                     for k, v in arrays.items()}, manifest=np.array(json.dumps(m)))
    os.replace(tmp, path)


def load_features(path, m):
    if not os.path.exists(path): return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if json.loads(str(z['manifest']))['configuration_hash'] != m['configuration_hash']:
                return None
            return {k: z[k] for k in z.files if k != 'manifest'}
    except Exception:
        return None


def load_subject_window_cache(db_key, window_ms, overlap, group_classes):
    """Load all windowed subjects from a validated Level-1 disk cache."""
    cache_dir = os.path.join(config.OUTPUT_DIR, 'cache', 'windows')
    os.makedirs(cache_dir, exist_ok=True)
    key = json.dumps({'db': db_key, 'window_ms': int(window_ms),
                      'overlap': float(overlap), 'group_classes': bool(group_classes),
                      'cache_version': 'windows-v1'}, sort_keys=True)
    stem = hashlib.sha256(key.encode()).hexdigest()
    manifest_path = os.path.join(cache_dir, stem + '.json')
    expected = {'key': key, 'database': db_key, 'window_ms': int(window_ms),
                'overlap': float(overlap), 'group_classes': bool(group_classes)}
    if os.path.exists(manifest_path):
        try:
            got = json.load(open(manifest_path))
            files = got['files']
            if got.get('metadata') == expected and all(os.path.exists(x) for x in files):
                return _lazy_subjects_from_cache(files)
        except Exception:
            pass
    subjects = list(load_ninapro_stream(db_key, config, window_ms, overlap, group_classes))
    files = []
    for s in subjects:
        fp = os.path.join(cache_dir, f'{stem}_subject{int(s["subject_id"])}.npz')
        tmp = fp + '.tmp.npz'
        np.savez(tmp, subject_id=np.array(s['subject_id']), X=s['X'].astype(np.float32),
                 y=s['y'].astype(np.int32), rep_id=s['rep_id'].astype(np.int32),
                 n_classes=np.array(s['n_classes']), sampling_rate=np.array(s['sampling_rate']))
        os.replace(tmp, fp); files.append(fp)
    with open(manifest_path + '.tmp', 'w') as f:
        json.dump({'metadata': expected, 'files': files}, f, indent=2)
    os.replace(manifest_path + '.tmp', manifest_path)
    return subjects


class _LazyNpzArray:
    """Read one uncompressed NPZ member only when the array is accessed.

    The window cache is intentionally disk-backed. Loading all DB2 subjects
    eagerly duplicates roughly 10 GB of arrays in RAM before LOSO starts.
    """
    def __init__(self, path, key):
        self.path, self.key = path, key
        with np.load(path, allow_pickle=False) as z:
            self.shape, self.dtype = z[key].shape, z[key].dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        with np.load(self.path, allow_pickle=False) as z:
            return z[self.key][item]

    def astype(self, dtype):
        return np.asarray(self, dtype=dtype)

    def __array__(self, dtype=None):
        with np.load(self.path, allow_pickle=False) as z:
            a = z[self.key]
            return np.asarray(a, dtype=dtype) if dtype is not None else np.asarray(a)


def _lazy_subjects_from_cache(files):
    out = []
    for fp in files:
        with np.load(fp, allow_pickle=False) as z:
            sid = int(z['subject_id']); ncls = int(z['n_classes'])
            fs = int(z['sampling_rate'])
        out.append({'subject_id': sid, 'X': _LazyNpzArray(fp, 'X'),
                    'y': _LazyNpzArray(fp, 'y'), 'rep_id': _LazyNpzArray(fp, 'rep_id'),
                    'n_classes': ncls, 'sampling_rate': fs})
    return out


def adapt(Xs, Xc, Xt, method, seed, coral_reg, tca_components, sa_components):
    if method == 'raw': return Xs, Xt, 0
    if method == 'centering': return (*feature_centering(Xs, Xt), 0)
    if method == 'coral':
        a, _, b = coral_adaptation(Xs, Xc, Xt, reg=coral_reg); return a, b, len(Xc)
    if method == 'tca':
        a, _, b = tca_transform(Xs, Xc, Xt, n_components=tca_components, random_state=seed); return a, b, len(Xc)
    if method == 'sa':
        a, _, b = sa_transform(Xs, Xc, Xt, n_components=sa_components, random_state=seed); return a, b, len(Xc)
    raise ValueError(method)


def evaluate(Xs, ys, Xc, Xt, yt, method, sid, n_classes, seed, extractor,
             n_subjects, coral_reg, tca_components, sa_components):
    t = time.time()
    Xa, Xta, ncal = adapt(Xs, Xc, Xt, method, seed, coral_reg, tca_components, sa_components)
    t_adapt = time.time() - t
    t = time.time(); scaler = BatchedStandardScaler(batch_size=5000)
    Xsc = scaler.fit_transform(Xa); Xtsc = scaler.transform(Xta); t_scale = time.time()-t
    # FIX-DIAGNOSTIC-01: measure covariance before/after on the same
    # z-scored scale the classifier actually sees, not on raw pre-
    # standardization numbers (which were numerically unstable, esp.
    # for TCA where cov_fro_after reached 1e9-1e10).
    raw_scaler = BatchedStandardScaler(batch_size=5000)
    Xs_std = raw_scaler.fit_transform(Xs); Xt_std = raw_scaler.transform(Xt)
    before_cov = covariance_frobenius(Xs_std, Xt_std); before_mmd = mmd_squared_linear(Xs_std, Xt_std)
    after_cov = covariance_frobenius(Xsc, Xtsc); after_mmd = mmd_squared_linear(Xsc, Xtsc)
    del Xa, Xta, Xs_std, Xt_std, raw_scaler, scaler; gc.collect()
    n = min(N_RIDGE, len(ys)); idx = np.random.RandomState(seed).choice(len(ys), n, replace=False)
    Xsub, ysub = Xsc[idx].copy(), ys[idx]; del Xsc; gc.collect()
    clf = None
    for kw in (dict(alphas=np.logspace(-3,6,15),gcv_mode='svd',store_cv_results=False),
               dict(alphas=np.logspace(-3,6,15),gcv_mode='svd',store_cv_values=False),
               dict(alphas=np.logspace(-3,6,15),store_cv_results=False),
               dict(alphas=np.logspace(-3,6,15))):
        try: clf = RidgeClassifierCV(**kw); clf.fit(Xsub, ysub); break
        except TypeError: pass
    if clf is None: raise RuntimeError('RidgeClassifierCV unsupported by installed sklearn')
    ypred = clf.predict(Xtsc); acc=float(np.mean(ypred==yt)); bal=float(balanced_accuracy_score(yt,ypred)); mf=float(f1_score(yt,ypred,average='macro',zero_division=0))
    rb=random_baseline_from_labels(yt,len(np.unique(yt)))
    out={'subject_id':int(sid),'accuracy':round(acc,6),'balanced_accuracy':round(bal,6),'macro_f1':round(mf,6),'best_alpha':float(clf.alpha_),'f1_per_class':[round(v,4) for v in per_class_f1(yt,ypred,n_classes)],'confusion_matrix':confusion_matrix(yt,ypred,n_classes).tolist(),'n_train_subjects':n_subjects,'n_train_samples':len(ys),'n_test_samples':len(yt),'n_calibration_samples_available':len(Xc),'n_calibration_samples_used':ncal if method in config.TRANSDUCTIVE_METHODS else 0,'evaluation_paradigm':config.EVALUATION_PARADIGM[method],'feature_extractor':extractor,'seed':int(seed),'random_baseline_pct':round(rb,2),'ratio_vs_random':round(ratio_vs_random(acc,rb),2),'domain_shift':{'cov_before':before_cov,'cov_after':after_cov,'mmd_before':before_mmd,'mmd_after':after_mmd},'timing':{'adaptation':round(t_adapt,2),'scaling':round(t_scale,2)}}
    del Xsub, ysub, Xtsc, clf; gc.collect(); return out


def run_loso_shared(db_key, methods, num_kernels, output_dir, window_ms=200, overlap=0.5,
                    group_classes=False, resume=False, max_train_samples=50000,
                    per_subject_limit=5000, seed=42, feature_extractor='canonical',
                    calibration_fraction=0.5, max_subjects=None, fit_mode='full_train',
                    rocket_n_jobs=1, **kwargs):
    valid,msg=validate_data_path(db_key,config)
    if not valid: raise FileNotFoundError(msg)
    os.makedirs(output_dir,exist_ok=True); cache_dir=os.path.join(config.OUTPUT_DIR,'cache','features'); os.makedirs(cache_dir,exist_ok=True); os.makedirs(config.CHECKPOINT_DIR,exist_ok=True)
    subjects=load_subject_window_cache(db_key,window_ms,overlap,group_classes); ids=[s['subject_id'] for s in subjects]
    if max_subjects and max_subjects<len(ids): ids=sorted(np.random.RandomState(seed).choice(ids,max_subjects,replace=False).tolist())
    ncls=get_n_classes(db_key,config,group_classes); tag=f'{db_key}__{feature_extractor}__seed{seed}__shared'; ck=os.path.join(config.CHECKPOINT_DIR,'ckpt_'+tag+'.pkl'); state={}
    if resume and os.path.exists(ck):
        try:
            with open(ck,'rb') as f: state=pickle.load(f)
        except Exception: state={}
    for sid_idx, sid in enumerate(ids):
        sidkey=str(sid); state.setdefault(sidkey,{})
        # FIX-P08-01: قبل أي شيء آخر (بما فيها تجميع بيانات التدريب أو
        # إعادة استخراج MiniROCKET)، تحقق إن كانت كل الطرق المطلوبة
        # لهذا subject منجزة فعلاً في checkpoint. سابقاً كان هذا التحقق
        # يحدث فقط داخل حلقة method بعد استخراج الخصائص بالكامل — فإن
        # كان ملف الخصائص المخبّأ محذوفاً (DELETE_FEATURE_CACHE_AFTER_FOLD=1،
        # كما في التشغيل الأصلي)، كان يُعاد الاستخراج المكلف (600-1200s)
        # لكل subject فقط ليُكتشَف بعدها أنه غير ضروري إطلاقاً ويُتخطى.
        # هذا التحقق المبكر يجعل استئناف تجربة مكتملة بالفعل فوريًا.
        if all(m in state[sidkey] and 'error' not in state[sidkey][m]
              for m in methods):
            print(f"  [{db_key} seed={seed}] subject {sid} "
                  f"({sid_idx+1}/{len(ids)}): SKIP (all "
                  f"{len(methods)} methods already in checkpoint)",
                  flush=True)
            continue
        print(f"  [{db_key} seed={seed}] subject {sid} "
              f"({sid_idx+1}/{len(ids)}): processing...", flush=True)
        target=next(s for s in subjects if s['subject_id']==sid)
        target_y = np.asarray(target['y'])
        target_rep = np.asarray(target['rep_id'])
        mf,mc,_=split_by_repetition(target_y,target_rep,held_out_fraction=calibration_fraction,seed=seed,min_reps_per_class_each_side=1)
        xs=[]; ys=[]; total=0; ns=0
        for s in subjects:
            if s['subject_id']==sid: continue
            x,y=s['X'].astype(np.float32),np.asarray(s['y'])
            if len(y)>per_subject_limit: x,y=stratified_subsample(x,y,per_subject_limit,seed)
            xs.append(x); ys.append(y); total+=len(y); ns+=1
            if total>=max_train_samples: break
        Xs=np.vstack(xs); Y=np.concatenate(ys)
        if len(Y)>max_train_samples: Xs,Y=stratified_subsample(Xs,Y,max_train_samples,seed)
        m=manifest(db_key,sid,seed,feature_extractor,window_ms,overlap,num_kernels,calibration_fraction,kwargs.get('coral_reg',config.CORAL_REG),kwargs.get('tca_components',config.TCA_COMPONENTS),kwargs.get('sa_components',config.SA_COMPONENTS),fit_mode)
        fp=os.path.join(cache_dir,m['configuration_hash']+'.npz'); a=load_features(fp,m); fit_time=0.0
        if a is None:
            t=time.time(); rocket=get_feature_extractor(feature_extractor,num_kernels=num_kernels,random_state=seed,n_jobs=rocket_n_jobs,fit_mode=fit_mode); Xtr=rocket.fit_transform(Xs); Xraw=np.asarray(target['X']); Xca=rocket.transform(Xraw[mc]) if mc.sum() else np.zeros((0,Xtr.shape[1]),np.float32); Xte=rocket.transform(Xraw[mf]); a={'X_train':Xtr,'y_train':Y,'X_calib':Xca,'X_final':Xte,'y_test':target_y[mf]}; del Xraw; save_features(fp,m,a); fit_time=time.time()-t
            print(f"    extracted in {fit_time:.1f}s", flush=True)
        for method in methods:
            if method in state[sidkey] and 'error' not in state[sidkey][method]: continue
            try:
                r=evaluate(a['X_train'],a['y_train'],a['X_calib'],a['X_final'],a['y_test'],method,sid,ncls,seed,feature_extractor,ns,kwargs.get('coral_reg',config.CORAL_REG),kwargs.get('tca_components',config.TCA_COMPONENTS),kwargs.get('sa_components',config.SA_COMPONENTS)); r['timing']['feature_fit']=round(fit_time,2); state[sidkey][method]=r
            except Exception as e: state[sidkey][method]={'subject_id':int(sid),'error':str(e)}
            atomic_pickle(state,ck)
        del Xs,Y,xs,ys,a; gc.collect()
        if os.environ.get('DELETE_FEATURE_CACHE_AFTER_FOLD', '0') == '1':
            try:
                os.remove(fp)
            except FileNotFoundError:
                pass
    outputs={}
    for method in methods:
        folds=[state[str(sid)][method] for sid in ids if method in state.get(str(sid),{}) and 'error' not in state[str(sid)][method]]; acc=[r['accuracy'] for r in folds]; bal=[r['balanced_accuracy'] for r in folds]; f1=[r['macro_f1'] for r in folds]; d=descriptive_stats(acc) if acc else {}; db=descriptive_stats(bal) if bal else {}; df=descriptive_stats(f1) if f1 else {}
        out={'experiment':f'{db_key}_{method}__{feature_extractor}__seed{seed}__shared','database':db_key,'method':method,'evaluation_paradigm':config.EVALUATION_PARADIGM[method],'feature_extractor':feature_extractor,'seed':seed,'num_kernels':num_kernels,'window_ms':window_ms,'overlap':overlap,'fit_mode':fit_mode,'n_subjects':len(ids),'n_folds_ok':len(folds),'summary':{'mean':d.get('mean',0),'std':d.get('std',0),'median':d.get('median',0),'min':d.get('min',0),'max':d.get('max',0),'ci95_lower':d.get('ci95_lower',0),'ci95_upper':d.get('ci95_upper',0),'n':len(folds),'balanced_accuracy_mean':db.get('mean',0),'balanced_accuracy_std':db.get('std',0),'macro_f1_mean':df.get('mean',0),'macro_f1_std':df.get('std',0)},'per_subject':folds,'provenance':{'pipeline_version':'shared-v1','feature_cache_dir':cache_dir}}
        path=os.path.join(output_dir,f'results_{db_key}_{method}__{feature_extractor}__seed{seed}__shared.json'); open(path,'w').write(json.dumps(out,indent=2,default=str)); outputs[method]=out
    return outputs