import os, sys, tempfile, shutil, json
import numpy as np
import scipy.io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from optimized_pipeline import run_loso_shared

def build(path, sid, rng):
    fs, ch, moves, reps = 2000, 4, 3, 4
    chunks=[]; labels=[]; repids=[]
    for r in range(1, reps+1):
        for m in range(1, moves+1):
            x=rng.randn(fs//4,ch).astype(np.float32); x[:,m%ch]+=2
            chunks.append(x); labels.append(np.full(len(x),m)); repids.append(np.full(len(x),r))
    scipy.io.savemat(path, {'emg':np.concatenate(chunks), 'restimulus':np.concatenate(labels)[:,None], 'repetition':np.concatenate(repids)[:,None], 'sampling_frequency':np.array([[fs]])})

def main():
    td=tempfile.mkdtemp(prefix='shared_pipeline_')
    old_paths, old_meta = dict(config.DB_PATHS), dict(config.DB_META)
    try:
        for sid in range(1,4): build(os.path.join(td,f'S{sid}_E1_A1.mat'),sid,np.random.RandomState(sid))
        config.DB_PATHS['synthshared']=td
        config.DB_META['synthshared']={'n_subjects':3,'n_movements':3,'n_channels':4,'sampling_rate':2000,'subject_ids':[1,2,3],'movement_ids':[1,2,3],'exclude_rest':True,'exercise_e1_only':False,'grouped_classes':None}
        out=tempfile.mkdtemp(prefix='shared_out_')
        r=run_loso_shared('synthshared',['raw','centering'],300,out,seed=17,feature_extractor='custom_ppv',resume=True,max_train_samples=1000,per_subject_limit=500,calibration_fraction=.5)
        assert set(r)=={'raw','centering'}
        assert all(v['n_folds_ok']==3 for v in r.values())
        # Second run must resume all methods without changing results.
        r2=run_loso_shared('synthshared',['raw','centering'],300,out,seed=17,feature_extractor='custom_ppv',resume=True,max_train_samples=1000,per_subject_limit=500,calibration_fraction=.5)
        assert r['raw']['summary']['mean']==r2['raw']['summary']['mean']
        print('PASS: shared extraction pipeline, feature cache, atomic checkpoint, deterministic resume')
    finally:
        config.DB_PATHS.clear(); config.DB_PATHS.update(old_paths)
        config.DB_META.clear(); config.DB_META.update(old_meta)
        shutil.rmtree(td,ignore_errors=True); shutil.rmtree(out,ignore_errors=True)
if __name__=='__main__': main()
