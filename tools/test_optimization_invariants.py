import os, sys, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import StandardScaler
from domain_adaptation import BatchedStandardScaler
from optimized_pipeline import adapt

def main():
    rng=np.random.RandomState(42)
    X=rng.randn(127,31).astype(np.float32); T=rng.randn(29,31).astype(np.float32)
    a=BatchedStandardScaler(batch_size=17).fit_transform(X); b=StandardScaler().fit_transform(X)
    assert np.max(np.abs(a-b)) <= 1e-6, np.max(np.abs(a-b))
    # Inductive methods must be invariant to arbitrary calibration values.
    Xs=rng.randn(40,8).astype(np.float32); Xt=rng.randn(12,8).astype(np.float32)
    c1=rng.randn(10,8).astype(np.float32); c2=c1+1000
    for method in ('raw','centering'):
        p1=adapt(Xs,c1,Xt,method,42,1e-3,5,5)
        p2=adapt(Xs,c2,Xt,method,42,1e-3,5,5)
        assert np.array_equal(p1[0],p2[0]) and np.array_equal(p1[1],p2[1])
    # Repetition sets are disjoint by construction in the audited splitter.
    from subject_split import split_by_repetition
    y=np.repeat(np.arange(3),12); reps=np.tile(np.arange(1,5),9)
    mf,mc,_=split_by_repetition(y,reps,.5,42,1)
    final_pairs = set(zip(y[mf].tolist(), reps[mf].tolist()))
    calib_pairs = set(zip(y[mc].tolist(), reps[mc].tolist()))
    assert not final_pairs.intersection(calib_pairs)
    print('PASS: BatchedStandardScaler equivalence, inductive calibration isolation, repetition disjointness')
if __name__=='__main__': main()
