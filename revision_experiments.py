#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revision_experiments.py
=======================
Extra experiments requested by the reviewers of
"A Cost-Aware Evaluation of Ensemble Intrusion Detection for Drones".

Run in Google Colab (or any machine) with the processed CSV:

    !pip install -q imbalanced-learn
    !python revision_experiments.py --csv "/content/drive/MyDrive/Drone IDS Dataset Processed/updated_drone_ids_dataset.csv" \
                                     --out /content/drive/MyDrive/revision_out --stages all

Stages (each one saves its result to --out and is skipped on re-run unless --force):
  prep      data audit, duplicate removal, grouped split, temporal split checks
  main      17 detectors, train-only scaling, one single-row latency harness
  stats     exact McNemar + Holm/BH, paired CIs, equivalence (TOST), Wilson CIs
  ablation  identifier ablation (62/60/58 features) x (grouped, temporal) split
  calib     post hoc calibration (temperature, isotonic) + cascade threshold sweep
  prior     reweighting of the test set to the raw class proportions
  repeat    5 grouped seeds + 5 blocked time folds for DT, RF, HGB, cascade
  figs      figures for the revised manuscript

Total run time on a 2 vCPU machine is roughly 3 hours for the full data set.
The final step is to zip the --out folder and send it back.
"""
import os, sys, json, time, argparse, platform, warnings, math
from collections import OrderedDict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, VotingClassifier,
                              BaggingClassifier, StackingClassifier, AdaBoostClassifier)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, log_loss, recall_score)
from scipy.optimize import minimize, minimize_scalar
from scipy import stats as sps
from threadpoolctl import threadpool_limits

SEED = 42
C = 10
CLASS_NAMES = {0: "Benign", 1: "DoS", 2: "Injection", 3: "IP Spoofing", 4: "MITM",
               5: "Password Cracking", 6: "Payload Manipulation", 7: "Replay Attack",
               8: "Unauthorized UDP", 9: "Video Interception"}
RAW_COUNTS = {0: 1280814, 1: 724702, 2: 4743, 3: 9733, 4: 8219,
              5: 231218, 6: 6099, 7: 5860, 8: 5012, 9: 77235}
ID_COLS = ["ts", "Sequence number"]
DEVICE_COLS = ["Drone_port", "DS status"]

CFG = dict(smoke=False, trees=200, hgb_iter=200, bag_trees=100, ada_trees=200,
           n_lat=500, n_warm=20, B=1000, n_seeds=5, n_folds=5, pareto_B=200)

# ----------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------
class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, cls=NpEnc)

def load(path):
    with open(path) as f:
        return json.load(f)

def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)

def align(proba, model_classes, classes):
    out = np.zeros((proba.shape[0], len(classes)))
    idx = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(model_classes):
        out[:, idx[c]] = proba[:, j]
    return out

def force_single_thread(obj, seen=None, depth=0):
    """Set n_jobs=1 on every nested scikit-learn estimator (inference only)."""
    if seen is None: seen = set()
    if id(obj) in seen or depth > 6: return
    seen.add(id(obj))
    if isinstance(obj, (list, tuple)):
        for o in obj: force_single_thread(o, seen, depth + 1)
        return
    if isinstance(obj, dict):
        for o in obj.values(): force_single_thread(o, seen, depth + 1)
        return
    if isinstance(obj, BaseEstimator) or hasattr(obj, "__dict__"):
        if hasattr(obj, "n_jobs"):
            try: obj.n_jobs = 1
            except Exception: pass
        if "sklearn.tree._classes" in type(obj).__module__: return
        for v in list(getattr(obj, "__dict__", {}).values()):
            if isinstance(v, (BaseEstimator, list, tuple, dict)) or hasattr(v, "__dict__"):
                if not isinstance(v, (np.ndarray, pd.DataFrame)):
                    force_single_thread(v, seen, depth + 1)

# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def quality(y, p):
    return dict(accuracy=float(accuracy_score(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, p)),
                macro_f1=float(f1_score(y, p, average="macro", zero_division=0)),
                weighted_f1=float(f1_score(y, p, average="weighted", zero_division=0)),
                mcc=float(matthews_corrcoef(y, p)),
                n_errors=int((y != p).sum()), n=int(len(y)))

def macro_f1_idx(y, p, idx):
    yy, pp = y[idx], p[idx]
    cm = np.bincount(yy * C + pp, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return float(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).mean())

def per_sample_brier(y, P):
    Y = np.zeros_like(P); Y[np.arange(len(y)), y] = 1
    return ((P - Y) ** 2).sum(1)

def per_sample_logloss(y, P):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))

def ece_parts(conf, correct, nb=15):
    b = np.clip(np.ceil(conf * nb).astype(int) - 1, 0, nb - 1)
    return b

def ece_from(b, conf, correct, w=None, nb=15):
    if w is None:
        n = np.bincount(b, minlength=nb).astype(float)
        a = np.bincount(b, weights=correct, minlength=nb)
        c = np.bincount(b, weights=conf, minlength=nb)
        tot = n.sum()
    else:
        n = np.bincount(b, weights=w, minlength=nb)
        a = np.bincount(b, weights=w * correct, minlength=nb)
        c = np.bincount(b, weights=w * conf, minlength=nb)
        tot = n.sum()
    return float(np.abs(a - c).sum() / tot), n

def calib_summary(y, P, w=None, nb=15, boot=0, seed=0):
    """Brier, log loss, ECE (equal width bins), signed gap, accuracy (+ bootstrap CIs)."""
    y = np.asarray(y); conf = P.max(1); pred = P.argmax(1)
    correct = (pred == y).astype(float)
    br = per_sample_brier(y, P); ll = per_sample_logloss(y, P)
    b = ece_parts(conf, correct, nb)
    ece, counts = ece_from(b, conf, correct, w, nb)
    ww = np.ones(len(y)) if w is None else w
    out = dict(brier=float((ww * br).sum() / ww.sum()), logloss=float((ww * ll).sum() / ww.sum()),
               ece=ece, accuracy=float((ww * correct).sum() / ww.sum()),
               mean_conf=float((ww * conf).sum() / ww.sum()),
               gap_conf_minus_acc=float((ww * (conf - correct)).sum() / ww.sum()),
               bin_counts=counts)
    if boot:
        rng = np.random.default_rng(seed); n = len(y)
        bb, ll_, ee = [], [], []
        for _ in range(boot):
            idx = rng.integers(0, n, n)
            wi = None if w is None else w[idx]
            wsum = n if w is None else w[idx].sum()
            bb.append(((np.ones(n) if w is None else w[idx]) * br[idx]).sum() / wsum)
            ll_.append(((np.ones(n) if w is None else w[idx]) * ll[idx]).sum() / wsum)
            ee.append(ece_from(b[idx], conf[idx], correct[idx], wi, nb)[0])
        for k, v in (("brier", bb), ("logloss", ll_), ("ece", ee)):
            out[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out

def nondominated(lat, scores):
    """lat: minimise. scores: list of arrays, maximise. Strict Pareto dominance."""
    n = len(lat); keep = []
    for i in range(n):
        dom = False
        for j in range(n):
            if i == j: continue
            ge = lat[j] <= lat[i] and all(s[j] >= s[i] for s in scores)
            st = lat[j] < lat[i] or any(s[j] > s[i] for s in scores)
            if ge and st: dom = True; break
        if not dom: keep.append(i)
    return keep

# ----------------------------------------------------------------------------
# detectors (numpy versions of the notebook classes)
# ----------------------------------------------------------------------------
def mk_dt(seed=SEED):
    return DecisionTreeClassifier(max_depth=25, min_samples_leaf=2, class_weight="balanced", random_state=seed)
def mk_rf(seed=SEED):
    return RandomForestClassifier(n_estimators=CFG["trees"], max_depth=25, min_samples_leaf=2, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
def mk_et(seed=SEED):
    return ExtraTreesClassifier(n_estimators=CFG["trees"], max_depth=25, min_samples_leaf=2, max_features="sqrt",
                                class_weight="balanced", n_jobs=-1, random_state=seed)
def mk_hgb(seed=SEED):
    return HistGradientBoostingClassifier(max_iter=CFG["hgb_iter"], learning_rate=0.1, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          random_state=seed)
def base_list(seed=SEED):
    return [("GNB", GaussianNB(var_smoothing=1e-9)), ("DT", mk_dt(seed)), ("RF", mk_rf(seed)),
            ("ET", mk_et(seed)), ("HGB", mk_hgb(seed))]


class ConfidenceCascade(BaseEstimator, ClassifierMixin):
    """DT -> HGB -> RF. Exit when top-1 confidence >= conf_thr and margin >= margin_thr."""
    def __init__(self, conf_thr=0.995, margin_thr=0.20):
        self.conf_thr = conf_thr; self.margin_thr = margin_thr
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.stage1_ = mk_dt().fit(X, y); self.stage2_ = mk_hgb().fit(X, y); self.stage3_ = mk_rf().fit(X, y)
        return self
    @classmethod
    def from_fitted(cls, dt, hgb, rf, classes, conf_thr=0.995, margin_thr=0.20):
        o = cls(conf_thr, margin_thr); o.classes_ = classes
        o.stage1_, o.stage2_, o.stage3_ = dt, hgb, rf
        return o
    @staticmethod
    def _cm(P):
        s = np.sort(P, axis=1); return s[:, -1], s[:, -1] - s[:, -2]
    def predict_proba_with_stage(self, X):
        n = X.shape[0]; K = len(self.classes_)
        final = np.zeros((n, K)); stage = np.zeros(n, dtype=int); rem = np.arange(n)
        P1 = align(self.stage1_.predict_proba(X[rem]), self.stage1_.classes_, self.classes_)
        c1, m1 = self._cm(P1); e1 = (c1 >= self.conf_thr) & (m1 >= self.margin_thr)
        final[rem[e1]] = P1[e1]; stage[rem[e1]] = 1; rem = rem[~e1]
        if len(rem):
            P2 = align(self.stage2_.predict_proba(X[rem]), self.stage2_.classes_, self.classes_)
            c2, m2 = self._cm(P2); e2 = (c2 >= self.conf_thr) & (m2 >= self.margin_thr)
            final[rem[e2]] = P2[e2]; stage[rem[e2]] = 2; rem2 = rem[~e2]
            if len(rem2):
                P3 = align(self.stage3_.predict_proba(X[rem2]), self.stage3_.classes_, self.classes_)
                final[rem2] = P3; stage[rem2] = 3
        return final, stage
    def predict_proba(self, X): return self.predict_proba_with_stage(X)[0]
    def predict(self, X): return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class HierarchicalIDS(BaseEstimator, ClassifierMixin):
    """RF benign vs attack, HGB attack type; both stages are combined for every flow."""
    def __init__(self, benign_label=0): self.benign_label = benign_label
    def fit(self, X, y):
        y = np.asarray(y); self.classes_ = np.unique(y)
        yb = (y != self.benign_label).astype(int)
        self.stage1_ = mk_rf().fit(X, yb)
        m = y != self.benign_label
        self.stage2_ = mk_hgb().fit(X[m], y[m])
        return self
    def predict_proba(self, X):
        n = X.shape[0]; out = np.zeros((n, len(self.classes_))); idx = {c: i for i, c in enumerate(self.classes_)}
        pb = self.stage1_.predict_proba(X); b = list(self.stage1_.classes_).index(0)
        p_ben = pb[:, b]; p_att = 1 - p_ben
        out[:, idx[self.benign_label]] = p_ben
        pt = self.stage2_.predict_proba(X)
        for j, c in enumerate(self.stage2_.classes_): out[:, idx[c]] = p_att * pt[:, j]
        return out / np.clip(out.sum(1, keepdims=True), 1e-12, None)
    def predict(self, X): return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class BlendingClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        y = np.asarray(y); self.classes_ = np.unique(y)
        Xb, Xh, yb, yh = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        models, feats = [], []
        for name, est in base_list():
            est.fit(Xb, yb); models.append((name, est))
            feats.append(align(est.predict_proba(Xh), est.classes_, self.classes_))
        self.meta_ = LogisticRegression(max_iter=1000, random_state=SEED).fit(np.hstack(feats), yh)
        self.models_ = [(n, e.fit(X, y)) for n, e in models]
        return self
    def predict_proba(self, X):
        M = np.hstack([align(e.predict_proba(X), e.classes_, self.classes_) for _, e in self.models_])
        return align(self.meta_.predict_proba(M), self.meta_.classes_, self.classes_)
    def predict(self, X): return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class SpecialistEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, top_pairs=3, min_confusions=5, ambiguity=0.15):
        self.top_pairs = top_pairs; self.min_confusions = min_confusions; self.ambiguity = ambiguity
    def fit(self, X, y):
        from itertools import combinations
        from sklearn.metrics import confusion_matrix
        y = np.asarray(y); self.classes_ = np.unique(y)
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        g = mk_hgb().fit(Xt, yt)
        cm = confusion_matrix(yv, g.predict(Xv), labels=self.classes_)
        sc = []
        for i, j in combinations(range(len(self.classes_)), 2):
            c = cm[i, j] + cm[j, i]
            if c >= self.min_confusions: sc.append((c, self.classes_[i], self.classes_[j]))
        sc.sort(reverse=True)
        self.pairs_ = [(a, b) for _, a, b in sc[:self.top_pairs]]
        self.specialists_ = {}
        for a, b in self.pairs_:
            m = (y == a) | (y == b)
            self.specialists_[(a, b)] = mk_rf().fit(X[m], y[m])
        self.general_ = mk_hgb().fit(X, y)
        return self
    def predict(self, X):
        proba = self.general_.predict_proba(X); pred = self.general_.classes_[proba.argmax(1)]
        for (a, b), sp in self.specialists_.items():
            ia = list(self.general_.classes_).index(a); ib = list(self.general_.classes_).index(b)
            amb = (np.abs(proba[:, ia] - proba[:, ib]) < self.ambiguity) & np.isin(pred, [a, b])
            if amb.any(): pred[amb] = sp.predict(X[amb])
        return pred
    def predict_proba(self, X): return self.general_.predict_proba(X)


def feature_views(cols):
    timing = [c for c in cols if any(k in c.lower() for k in ["time", "ts", "iat", "duration", "interval", "rate"])]
    payload = [c for c in cols if any(k in c.lower() for k in
               ["len", "size", "byte", "payload", "mean", "std", "min", "max", "count"])]
    used = set(timing) | set(payload)
    protocol = [c for c in cols if c not in used]
    return {"timing": timing, "payload": payload, "protocol": protocol}

class FeatureViewEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, columns=None): self.columns = columns
    def fit(self, X, y):
        y = np.asarray(y); self.classes_ = np.unique(y)
        v = feature_views(list(self.columns)); pos = {c: i for i, c in enumerate(self.columns)}
        self.views_ = {k: [pos[c] for c in cs] for k, cs in v.items()}
        self.view_names_ = v
        self.experts_ = {"timing": mk_rf(), "payload": mk_hgb(), "protocol": mk_et()}
        for k, e in self.experts_.items(): e.fit(X[:, self.views_[k]], y)
        return self
    def predict_proba(self, X):
        return np.mean([align(e.predict_proba(X[:, self.views_[k]]), e.classes_, self.classes_)
                        for k, e in self.experts_.items()], axis=0)
    def predict(self, X): return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class DCS(BaseEstimator, ClassifierMixin):
    def __init__(self, k=15): self.k = k
    def fit(self, X, y):
        y = np.asarray(y); self.classes_ = np.unique(y)
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        pool = [m.fit(Xt, yt) for _, m in base_list()]
        self.val_profiles_ = np.hstack([align(m.predict_proba(Xv), m.classes_, self.classes_) for m in pool])
        self.val_correct_ = np.vstack([(m.predict(Xv) == yv) for m in pool]).T
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(self.val_profiles_)
        self.pool_ = [m.fit(X, y) for _, m in base_list()]
        return self
    def _profiles(self, X):
        return np.hstack([align(m.predict_proba(X), m.classes_, self.classes_) for m in self.pool_])
    def predict_proba(self, X):
        _, nbr = self.nn_.kneighbors(self._profiles(X))
        comp = self.val_correct_[nbr].mean(1); best = comp.argmax(1)
        allp = np.stack([align(m.predict_proba(X), m.classes_, self.classes_) for m in self.pool_], axis=1)
        return allp[np.arange(len(X)), best]
    def predict(self, X):
        _, nbr = self.nn_.kneighbors(self._profiles(X))
        comp = self.val_correct_[nbr].mean(1); best = comp.argmax(1)
        allp = np.vstack([m.predict(X) for m in self.pool_]).T
        return allp[np.arange(len(X)), best]


class DES(DCS):
    def __init__(self, k=15, top=2):
        self.k = k; self.top = top
    @classmethod
    def from_dcs(cls, d, top=2):
        o = cls(d.k, top)
        for a in ("classes_", "val_profiles_", "val_correct_", "nn_", "pool_"): setattr(o, a, getattr(d, a))
        return o
    def predict_proba(self, X):
        _, nbr = self.nn_.kneighbors(self._profiles(X))
        comp = self.val_correct_[nbr].mean(1); topk = np.argsort(-comp, axis=1)[:, :self.top]
        allp = np.stack([align(m.predict_proba(X), m.classes_, self.classes_) for m in self.pool_], axis=1)
        out = np.zeros((len(X), len(self.classes_)))
        for r in range(len(X)): out[r] = allp[r, topk[r]].mean(0)
        return out
    def predict(self, X): return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class MixtureOfExperts(BaseEstimator, ClassifierMixin):
    def _experts(self): return {"RF": mk_rf(), "ET": mk_et(), "HGB": mk_hgb()}
    def fit(self, X, y):
        y = np.asarray(y); self.classes_ = np.unique(y)
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        V = []
        for n, e in self._experts().items():
            e.fit(Xt, yt); V.append(align(e.predict_proba(Xv), e.classes_, self.classes_))
        V = np.stack(V, 0)
        def nll(w):
            g = np.exp(w - w.max()); g /= g.sum()
            return log_loss(yv, np.tensordot(g, V, axes=(0, 0)), labels=self.classes_)
        r = minimize(nll, np.zeros(V.shape[0]), method="L-BFGS-B")
        g = np.exp(r.x - r.x.max()); self.gate_ = g / g.sum(); self.names_ = list(self._experts().keys())
        self.experts_ = {n: e.fit(X, y) for n, e in self._experts().items()}
        return self
    def predict_proba(self, X):
        mix = np.zeros((X.shape[0], len(self.classes_)))
        for g, n in zip(self.gate_, self.names_):
            mix += g * align(self.experts_[n].predict_proba(X), self.experts_[n].classes_, self.classes_)
        return mix
    def predict(self, X): return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def mk_adaboost():
    kw = dict(estimator=DecisionTreeClassifier(max_depth=3, random_state=SEED),
              n_estimators=CFG["ada_trees"], learning_rate=0.5, random_state=SEED)
    try:
        return AdaBoostClassifier(algorithm="SAMME", **kw)
    except TypeError:       # scikit-learn >= 1.8 removed the argument (SAMME is the only option)
        return AdaBoostClassifier(**kw)

def mk_balanced_bagging():
    from imblearn.ensemble import BalancedBaggingClassifier
    return BalancedBaggingClassifier(estimator=DecisionTreeClassifier(max_depth=25, min_samples_leaf=2, random_state=SEED),
                                     n_estimators=CFG["bag_trees"], sampling_strategy="auto", replacement=False,
                                     n_jobs=-1, random_state=SEED)

# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
class Data: pass

def prepare(csv, nrows=None):
    df0 = pd.read_csv(csv, nrows=nrows)
    feat_all = [c for c in df0.columns if c != "Label"]
    feat60 = [c for c in feat_all if c not in ID_COLS]
    d60 = df0[feat60 + ["Label"]]
    keep = ~d60.duplicated(subset=feat60, keep="first")
    dd = df0.loc[keep].reset_index(drop=True)
    D = Data(); D.dd = dd; D.feat_all = feat_all; D.feat60 = feat60
    D.feat58 = [c for c in feat60 if c not in DEVICE_COLS]
    D.y = dd["Label"].values.astype(np.int64)
    D.groups = pd.util.hash_pandas_object(dd[feat60].round(3), index=False).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    D.tr, D.te = next(gss.split(dd[feat60], D.y, groups=D.groups))
    D.info = dict(n_raw=int(len(df0)), n_after_dedup=int(len(dd)), n_removed=int((~keep).sum()),
                  n_features_with_ids=len(feat_all), n_features_paper=len(feat60), n_features_no_device=len(D.feat58),
                  n_pseudo_groups=int(len(np.unique(D.groups))), n_train=int(len(D.tr)), n_test=int(len(D.te)),
                  class_counts=np.bincount(D.y, minlength=C), test_class_counts=np.bincount(D.y[D.te], minlength=C),
                  train_class_counts=np.bincount(D.y[D.tr], minlength=C))
    return D

def xy(D, feats, tr, te, scale=True):
    X = D.dd[feats].values.astype(np.float64)
    if scale:
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    else:
        Xtr, Xte = X[tr], X[te]
    return Xtr, Xte, D.y[tr], D.y[te]

def temporal_split(D, test_frac=0.2):
    ts = D.dd["ts"].values; te_parts = []
    for c in range(C):
        ii = np.where(D.y == c)[0]; ii = ii[np.argsort(ts[ii], kind="stable")]
        k = int(round(len(ii) * (1 - test_frac))); te_parts.append(ii[k:])
    te = np.concatenate(te_parts); m = np.ones(len(D.y), bool); m[te] = False; tr = np.where(m)[0]
    keep = ~np.isin(D.groups[te], D.groups[tr]); purged = int((~keep).sum())
    return tr, te[keep], purged

def blocked_folds(D, k):
    ts = D.dd["ts"].values; chunks = [[] for _ in range(k)]
    for c in range(C):
        ii = np.where(D.y == c)[0]; ii = ii[np.argsort(ts[ii], kind="stable")]
        for f, part in enumerate(np.array_split(ii, k)): chunks[f].append(part)
    out = []
    for f in range(k):
        te = np.concatenate(chunks[f]); m = np.ones(len(D.y), bool); m[te] = False; tr = np.where(m)[0]
        keep = ~np.isin(D.groups[te], D.groups[tr]); out.append((tr, te[keep], int((~keep).sum())))
    return out

# ----------------------------------------------------------------------------
# stage: prep
# ----------------------------------------------------------------------------
def stage_prep(D, out):
    info = dict(D.info)
    # does row order follow time inside each class?
    rho = []
    ts = D.dd["ts"].values
    for c in range(C):
        ii = np.where(D.y == c)[0]
        if len(ii) > 10:
            rho.append(float(sps.spearmanr(np.arange(len(ii)), ts[ii]).correlation))
    info["spearman_roworder_vs_ts_per_class"] = rho
    tr, te, purged = temporal_split(D)
    info["temporal_split"] = dict(n_train=int(len(tr)), n_test=int(len(te)), purged_test_rows=purged,
                                  test_class_counts=np.bincount(D.y[te], minlength=C))
    # how well do identity fields alone predict the label ?
    info["majority_baseline_accuracy"] = float(np.bincount(D.y[D.te], minlength=C).max() / len(D.te))
    info["python"] = sys.version; info["platform"] = platform.platform(); info["cpu_count"] = os.cpu_count()
    import sklearn, scipy
    info["versions"] = dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__)
    try:
        import imblearn; info["versions"]["imblearn"] = imblearn.__version__
    except Exception: pass
    info["feature_views"] = feature_views(D.feat60)
    info["features_paper"] = D.feat60
    info["expected_from_paper_notebook"] = dict(n_after_dedup=301553, n_train=241324, n_test=60229)
    dump(info, os.path.join(out, "prep.json")); log("prep done:", {k: info[k] for k in
         ("n_raw", "n_after_dedup", "n_removed", "n_train", "n_test", "n_pseudo_groups")})
    return info

# ----------------------------------------------------------------------------
# stage: main benchmark
# ----------------------------------------------------------------------------
def latency_harness(model, Xte, idx):
    force_single_thread(model)
    rows = [np.ascontiguousarray(Xte[i:i + 1]) for i in idx]
    with threadpool_limits(limits=1):
        for r in rows[:CFG["n_warm"]]: model.predict(r)
        t = np.empty(len(rows))
        for k, r in enumerate(rows):
            t0 = time.perf_counter_ns(); model.predict(r); t[k] = (time.perf_counter_ns() - t0) / 1e6
    return dict(mean_ms=float(t.mean()), median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)),
                p99_ms=float(np.percentile(t, 99)), std_ms=float(t.std()), n_rows=int(len(t)))

def stage_main(D, out):
    Xtr, Xte, ytr, yte = xy(D, D.feat60, D.tr, D.te, scale=True)
    rng = np.random.default_rng(0)
    lat_idx = rng.choice(len(Xte), size=min(CFG["n_lat"] + CFG["n_warm"], len(Xte)), replace=False)
    classes = np.unique(ytr)
    res = OrderedDict(); preds = {}; probas = {}
    extra = {}

    def measure(name, model, fit_s, has_proba=True):
        t0 = time.perf_counter()
        if has_proba:
            P = align(model.predict_proba(Xte), model.classes_, classes); pred = classes[P.argmax(1)]
        else:
            pred = model.predict(Xte); P = None
        batch_s = time.perf_counter() - t0
        row = dict(fit_s=fit_s, batch_s=batch_s, throughput=len(yte) / max(batch_s, 1e-12))
        row.update(quality(yte, pred))
        if P is not None:
            cs = calib_summary(yte, P); row.update(brier=cs["brier"], logloss=cs["logloss"], ece=cs["ece"],
                                                   mean_conf=cs["mean_conf"], gap_conf_minus_acc=cs["gap_conf_minus_acc"],
                                                   bin_counts=cs["bin_counts"])
            probas[name] = P.astype(np.float32)
        row["latency"] = latency_harness(model, Xte, lat_idx)
        res[name] = row; preds[name] = pred.astype(np.int8)
        log(f"{name:24s} acc={row['accuracy']:.5f} mF1={row['macro_f1']:.5f} fit={fit_s:7.1f}s "
            f"lat(med)={row['latency']['median_ms']:.3f}ms p95={row['latency']['p95_ms']:.3f}ms")

    def timed_fit(model):
        t0 = time.perf_counter(); model.fit(Xtr, ytr); return time.perf_counter() - t0

    # --- single learners, then the cascade built from the SAME fitted stages
    dt = mk_dt(); f_dt = timed_fit(dt)
    hgb = mk_hgb(); f_hgb = timed_fit(hgb)
    rf = mk_rf(); f_rf = timed_fit(rf)
    # latency of each stage in the same harness (needed for the cascade cost model)
    cascade = ConfidenceCascade.from_fitted(dt, hgb, rf, classes)
    # NB: the cascade shares the stage objects, so measure the cascade FIRST and the singles afterwards
    measure("Confidence Cascade", cascade, f_dt + f_hgb + f_rf)
    P, stage = cascade.predict_proba_with_stage(Xte)
    ex = dict(stage_counts=np.bincount(stage, minlength=4)[1:], n_test=int(len(yte)))
    c1, m1 = cascade._cm(dt.predict_proba(Xte))
    ex["margin_binding_stage1"] = int(((c1 >= cascade.conf_thr) & (m1 < cascade.margin_thr)).sum())
    ex["min_margin_when_conf_ge_thr"] = float(m1[c1 >= cascade.conf_thr].min()) if (c1 >= cascade.conf_thr).any() else None
    ex["stage_accuracy"] = {int(s): (float((classes[P.argmax(1)][stage == s] == yte[stage == s]).mean())
                                     if (stage == s).any() else None) for s in (1, 2, 3)}
    esc = np.where(stage > 1)[0]
    if len(esc):
        lat = latency_harness(cascade, Xte, esc[:max(CFG["n_warm"], 1)] if len(esc) < CFG["n_warm"] else esc)
        ex["latency_on_escalated_flows"] = lat; ex["n_escalated"] = int(len(esc))
    extra["cascade_exit"] = ex
    rf.set_params(n_jobs=-1)          # the harness forced n_jobs=1 on the shared RF; restore for its batch timing
    measure("Decision Tree (single)", dt, f_dt)
    measure("HGB (single)", hgb, f_hgb)
    measure("Random Forest (single)", rf, f_rf)
    # modeled cascade cost from the harness medians / means
    lt = {k: res[k]["latency"]["mean_ms"] for k in ("Decision Tree (single)", "HGB (single)", "Random Forest (single)")}
    fr = ex["stage_counts"] / ex["n_test"]
    extra["cascade_modeled_mean_ms"] = float(fr[0] * lt["Decision Tree (single)"]
        + fr[1] * (lt["Decision Tree (single)"] + lt["HGB (single)"])
        + fr[2] * (lt["Decision Tree (single)"] + lt["HGB (single)"] + lt["Random Forest (single)"]))
    extra["full_three_stage_mean_ms"] = float(sum(lt.values()))
    del dt, hgb, rf, cascade

    def run(name, factory, has_proba=True):
        m = factory(); f = timed_fit(m); measure(name, m, f, has_proba)
        if name == "Mixture of Experts": extra["moe_gate"] = dict(zip(m.names_, m.gate_))
        if name == "Specialist Ensemble": extra["specialist_pairs"] = [list(map(int, p)) for p in m.pairs_]
        if name == "Feature-View Ensemble": extra["feature_view_members"] = m.view_names_
        return m

    run("Specialist Ensemble", SpecialistEnsemble)
    run("AdaBoost (SAMME)", mk_adaboost)
    run("Bagging (DT x100)", lambda: BaggingClassifier(estimator=mk_dt(), n_estimators=CFG["bag_trees"],
                                                      n_jobs=2, random_state=SEED))
    run("Hierarchical IDS", HierarchicalIDS)
    run("Mixture of Experts", MixtureOfExperts)
    run("Hard Voting", lambda: VotingClassifier(base_list(), voting="hard", n_jobs=1), has_proba=False)
    run("Soft Voting", lambda: VotingClassifier(base_list(), voting="soft", n_jobs=1))
    run("Blending (LR meta)", BlendingClassifier)
    run("Stacking (LR meta)", lambda: StackingClassifier(
        estimators=base_list(), final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        stack_method="predict_proba", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1))
    run("Feature-View Ensemble", lambda: FeatureViewEnsemble(columns=D.feat60))
    run("Balanced Bagging", mk_balanced_bagging)
    d = DCS(); f = timed_fit(d); measure("DCS", d, f)
    for m_ in d.pool_:
        if hasattr(m_, "n_jobs"): m_.n_jobs = -1
    de = DES.from_dcs(d); measure("DES", de, f)           # same pool, same fit time
    del d, de

    names = list(res.keys())
    lat = np.array([res[n]["latency"]["median_ms"] for n in names])
    f1 = np.array([res[n]["macro_f1"] for n in names]); acc = np.array([res[n]["accuracy"] for n in names])
    p95 = np.array([res[n]["latency"]["p95_ms"] for n in names])
    pareto = dict(f1_vs_median_latency=[names[i] for i in nondominated(lat, [f1])],
                  f1_vs_p95_latency=[names[i] for i in nondominated(p95, [f1])],
                  acc_vs_median_latency=[names[i] for i in nondominated(lat, [acc])])
    has_b = [i for i, n in enumerate(names) if "brier" in res[n]]
    br = np.array([-res[names[i]]["brier"] for i in has_b])
    pareto["f1_latency_brier_3d"] = [names[has_b[i]] for i in nondominated(lat[has_b], [f1[has_b], br])]
    # bootstrap stability of Pareto membership (F1 resampled, latency fixed)
    rng = np.random.default_rng(1); cnt = np.zeros(len(names)); n = len(yte)
    P_ = np.stack([preds[k].astype(np.int64) for k in names])
    for _ in range(CFG["pareto_B"]):
        idx = rng.integers(0, n, n)
        fb = np.array([macro_f1_idx(yte, P_[i], idx) for i in range(len(names))])
        for i in nondominated(lat, [fb]): cnt[i] += 1
    pareto["bootstrap_membership_freq"] = {nm: float(c / CFG["pareto_B"]) for nm, c in zip(names, cnt)}
    summary = dict(results=res, pareto=pareto, extra=extra,
                   test_size=int(len(yte)), train_size=int(len(ytr)), n_features=int(Xtr.shape[1]),
                   harness=dict(rows=int(CFG["n_lat"]), warmup=int(CFG["n_warm"]), input="numpy float64 (1,d)",
                                call="model.predict", threads="1 (threadpool_limits + n_jobs=1)",
                                clock="time.perf_counter_ns", cpu_count=os.cpu_count(), platform=platform.platform(),
                                processor=platform.processor()))
    dump(summary, os.path.join(out, "main.json"))
    np.savez_compressed(os.path.join(out, "main_preds.npz"), y=yte.astype(np.int8),
                        names=np.array(names), **{"pred__" + k: v for k, v in preds.items()},
                        **{"proba__" + k: v for k, v in probas.items()})
    log("main done. Pareto (F1 vs median latency):", pareto["f1_vs_median_latency"])
    return summary

# ----------------------------------------------------------------------------
# stage: statistics on main predictions
# ----------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]

def holm(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def bh(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p)[::-1]; adj = np.empty(m); run = 1.0
    for r, i in enumerate(o):
        k = m - r; run = min(run, p[i] * m / k); adj[i] = run
    return adj

def stage_stats(out):
    z = np.load(os.path.join(out, "main_preds.npz"), allow_pickle=True)
    y = z["y"].astype(np.int64); names = [str(n) for n in z["names"]]
    pr = {n: z["pred__" + n].astype(np.int64) for n in names}; N = len(y)
    corr = {n: pr[n] == y for n in names}
    rng = np.random.default_rng(3)
    pairs = []; Bq = 20000
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n10 = int((corr[a] & ~corr[b]).sum()); n01 = int((~corr[a] & corr[b]).sum())
            p = 1.0 if n10 + n01 == 0 else float(sps.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
            draw = rng.multinomial(N, [n10 / N, n01 / N, 1 - (n10 + n01) / N], size=Bq)
            d = (draw[:, 0] - draw[:, 1]) / N
            pairs.append(dict(a=a, b=b, a_only_right=n10, b_only_right=n01, p_exact=p,
                              diff=(n10 - n01) / N, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                              ci90=[float(np.percentile(d, 5)), float(np.percentile(d, 95))]))
    ps = np.array([q["p_exact"] for q in pairs]); hh = holm(ps); bb = bh(ps)
    for q, h, b_ in zip(pairs, hh, bb): q["p_holm"] = float(h); q["p_bh"] = float(b_)
    def count(sel):
        s = [q for q in pairs if sel(q)]
        return dict(n_pairs=len(s), raw_sig=sum(q["p_exact"] < .05 for q in s), holm_sig=sum(q["p_holm"] < .05 for q in s),
                    bh_sig=sum(q["p_bh"] < .05 for q in s), degenerate=sum((q["a_only_right"] + q["b_only_right"]) == 0 for q in s))
    allc = count(lambda q: True)
    nobb = count(lambda q: "Balanced Bagging" not in (q["a"], q["b"]))
    equiv = {}
    for delta in (0.0002, 0.0005):
        for tag, sel in (("all", lambda q: True), ("without_balanced_bagging", lambda q: "Balanced Bagging" not in (q["a"], q["b"]))):
            s = [q for q in pairs if sel(q)]
            eq = [q for q in s if q["ci90"][0] > -delta and q["ci90"][1] < delta]
            equiv[f"delta_{delta}_{tag}"] = dict(n_pairs=len(s), n_equivalent=len(eq),
                                                 not_equivalent=[[q["a"], q["b"]] for q in s if q not in eq])
    # per model intervals
    per = {}; B = CFG["B"]
    for n in names:
        k = int((~corr[n]).sum()); fb = []
        for _ in range(B):
            idx = rng.integers(0, N, N); fb.append(macro_f1_idx(y, pr[n], idx))
        per[n] = dict(n_errors=k, accuracy=1 - k / N, accuracy_wilson95=[1 - wilson(k, N)[1], 1 - wilson(k, N)[0]],
                      macro_f1=macro_f1_idx(y, pr[n], np.arange(N)), macro_f1_boot95=[float(np.percentile(fb, 2.5)), float(np.percentile(fb, 97.5))])
    # per-class recall for all models
    rec = {n: recall_score(y, pr[n], labels=list(range(C)), average=None, zero_division=0) for n in names}
    spread = {CLASS_NAMES[c]: float(max(rec[n][c] for n in names) - min(rec[n][c] for n in names)) for c in range(C)}
    sup = np.bincount(y, minlength=C)
    spread_wo = {CLASS_NAMES[c]: float(max(rec[n][c] for n in names if n != "Balanced Bagging") -
                                        min(rec[n][c] for n in names if n != "Balanced Bagging")) for c in range(C)}
    # spans
    mains = [n for n in names if n != "Balanced Bagging"]
    A = [per[n]["accuracy"] for n in mains]; F = [per[n]["macro_f1"] for n in mains]
    spans = dict(acc_all=max(per[n]["accuracy"] for n in names) - min(per[n]["accuracy"] for n in names),
                 f1_all=max(per[n]["macro_f1"] for n in names) - min(per[n]["macro_f1"] for n in names),
                 acc_wo_bb=max(A) - min(A), f1_wo_bb=max(F) - min(F),
                 flows_wo_bb=(max(A) - min(A)) * N, max_errors=max(per[n]["n_errors"] for n in names),
                 min_errors=min(per[n]["n_errors"] for n in names))
    dump(dict(pairs=pairs, all=allc, without_balanced_bagging=nobb, equivalence=equiv, per_model=per,
              recall={n: rec[n] for n in names}, recall_spread=spread, recall_spread_without_bb=spread_wo,
              support=sup, spans=spans), os.path.join(out, "stats.json"))
    log("stats done:", allc, nobb, spans)

# ----------------------------------------------------------------------------
# stage: identifier ablation
# ----------------------------------------------------------------------------
def stage_ablation(D, out):
    t_tr, t_te, purged = temporal_split(D)
    splits = {"grouped": (D.tr, D.te), "temporal": (t_tr, t_te)}
    fsets = {"62 (with ts and sequence number)": D.feat_all, "60 (paper)": D.feat60,
             "58 (also no drone port, DS status)": D.feat58}
    rows = []
    for sname, (tr, te) in splits.items():
        for fname, feats in fsets.items():
            Xtr, Xte, ytr, yte = xy(D, feats, tr, te, scale=True); cl = np.unique(ytr)
            dt = mk_dt(); t0 = time.perf_counter(); dt.fit(Xtr, ytr); f_dt = time.perf_counter() - t0
            hgb = mk_hgb(); t0 = time.perf_counter(); hgb.fit(Xtr, ytr); f_hgb = time.perf_counter() - t0
            rf = mk_rf(); t0 = time.perf_counter(); rf.fit(Xtr, ytr); f_rf = time.perf_counter() - t0
            casc = ConfidenceCascade.from_fitted(dt, hgb, rf, cl)
            Pc, stg = casc.predict_proba_with_stage(Xte)
            for mname, m, P in (("Decision Tree", dt, None), ("Random Forest", rf, None), ("HGB", hgb, None), ("Confidence Cascade", casc, Pc)):
                pred = m.predict(Xte); q = quality(yte, pred)
                rc = recall_score(yte, pred, labels=list(range(C)), average=None, zero_division=0)
                rows.append(dict(split=sname, features=fname, n_features=len(feats), model=mname, n_train=int(len(ytr)),
                                 n_test=int(len(yte)), min_class_recall=float(rc.min()),
                                 worst_class=CLASS_NAMES[int(rc.argmin())], **q,
                                 cascade_stage1_share=(float((stg == 1).mean()) if mname == "Confidence Cascade" else None)))
            log(f"ablation {sname:8s} {fname:38s} " + " ".join(
                f"{r['model'][:6]}={r['accuracy']:.5f}" for r in rows[-4:]))
    # identity-only probes
    probes = []
    for sname, (tr, te) in splits.items():
        for pname, cols in (("ts only", ["ts"]), ("sequence number only", ["Sequence number"]),
                            ("ts and sequence number", ["ts", "Sequence number"]),
                            ("drone port and DS status", DEVICE_COLS)):
            Xtr, Xte, ytr, yte = xy(D, cols, tr, te, scale=False)
            m = mk_dt().fit(Xtr, ytr); q = quality(yte, m.predict(Xte))
            probes.append(dict(split=sname, features=pname, **q))
        probes.append(dict(split=sname, features="majority class", accuracy=float(np.bincount(D.y[te], minlength=C).max() / len(te))))
    dump(dict(rows=rows, probes=probes, temporal=dict(n_train=int(len(t_tr)), n_test=int(len(t_te)), purged=purged)),
         os.path.join(out, "ablation.json"))

# ----------------------------------------------------------------------------
# stage: post hoc calibration and cascade threshold sweep
# ----------------------------------------------------------------------------
def softmax_T(P, T):
    z = np.log(np.clip(P, 1e-12, 1)) / T; z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit_temperature(Pv, yv):
    f = lambda lt: log_loss(yv, softmax_T(Pv, math.exp(lt)), labels=list(range(C)))
    r = minimize_scalar(f, bounds=(-6, 6), method="bounded"); return math.exp(r.x)

def fit_isotonic(Pv, yv):
    irs = []
    for c in range(C):
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip"); ir.fit(Pv[:, c], (yv == c).astype(float)); irs.append(ir)
    return irs

def apply_isotonic(irs, P):
    Q = np.stack([irs[c].predict(P[:, c]) for c in range(C)], 1) + 1e-6
    return Q / Q.sum(1, keepdims=True)

def stage_calib(D, out, with_stacking=False):
    # inner validation split taken from the training part (group aware)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    sub_rel, val_rel = next(gss.split(D.tr, D.y[D.tr], groups=D.groups[D.tr]))
    sub, val = D.tr[sub_rel], D.tr[val_rel]
    Xsub, Xval, ysub, yval = xy(D, D.feat60, sub, val, scale=True)
    sc = StandardScaler().fit(D.dd[D.feat60].values[sub])
    Xte = sc.transform(D.dd[D.feat60].values[D.te]); yte = D.y[D.te]
    classes = np.unique(ysub)
    factories = OrderedDict([("Decision Tree", mk_dt), ("Random Forest", mk_rf), ("HGB", mk_hgb), ("AdaBoost (SAMME)", mk_adaboost),
        ("Soft Voting", lambda: VotingClassifier(base_list(), voting="soft", n_jobs=1)),
        ("Feature-View Ensemble", lambda: FeatureViewEnsemble(columns=D.feat60)), ("Hierarchical IDS", HierarchicalIDS)])
    if with_stacking:
        factories["Stacking (LR meta)"] = lambda: StackingClassifier(estimators=base_list(),
            final_estimator=LogisticRegression(max_iter=1000, random_state=SEED), stack_method="predict_proba",
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1)
    raw = {}; cal = OrderedDict(); store = {}
    stage_models = {}
    for name, fac in factories.items():
        m = fac(); t0 = time.perf_counter(); m.fit(Xsub, ysub); log(f"calib fit {name} {time.perf_counter() - t0:.0f}s")
        Pv = align(m.predict_proba(Xval), m.classes_, classes); Pt = align(m.predict_proba(Xte), m.classes_, classes)
        raw[name] = (Pv, Pt)
        if name in ("Decision Tree", "HGB", "Random Forest"): stage_models[name] = (m, Pv, Pt)
        del m
    # cascade from the three sub-models: P1,P2,P3
    P1v, P1t = stage_models["Decision Tree"][1:]; P2v, P2t = stage_models["HGB"][1:]; P3v, P3t = stage_models["Random Forest"][1:]
    def casc(P1, P2, P3, tau):
        e1 = P1.max(1) >= tau; e2 = (~e1) & (P2.max(1) >= tau)
        P = np.where(e1[:, None], P1, np.where(e2[:, None], P2, P3)); stg = np.where(e1, 1, np.where(e2, 2, 3)); return P, stg
    Pcv, _ = casc(P1v, P2v, P3v, 0.995); Pct, _ = casc(P1t, P2t, P3t, 0.995)
    raw["Confidence Cascade"] = (Pcv, Pct)
    for name, (Pv, Pt) in raw.items():
        T = fit_temperature(Pv, yval); irs = fit_isotonic(Pv, yval)
        variants = OrderedDict([("raw", Pt), ("temperature", softmax_T(Pt, T)), ("isotonic", apply_isotonic(irs, Pt))])
        cal[name] = dict(T=T)
        for vname, Pq in variants.items():
            s = calib_summary(yte, Pq, boot=CFG["B"], seed=11); s["accuracy_unchanged"] = bool((Pq.argmax(1) == Pt.argmax(1)).all())
            cal[name][vname] = s
        store[name] = {k: v.astype(np.float32) for k, v in variants.items()}
        log(f"calib {name:24s} T={T:7.3f} Brier raw={cal[name]['raw']['brier']:.5f} T={cal[name]['temperature']['brier']:.5f} "
            f"iso={cal[name]['isotonic']['brier']:.5f}")
    # cascade threshold sweep (validation and test)
    sweep = []
    lat = {}
    mp = os.path.join(out, "main.json")
    if os.path.exists(mp):
        r = load(mp)["results"]; lat = dict(dt=r["Decision Tree (single)"]["latency"]["mean_ms"],
                                           hgb=r["HGB (single)"]["latency"]["mean_ms"], rf=r["Random Forest (single)"]["latency"]["mean_ms"])
    for tau in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0):
        row = dict(tau=tau)
        for tag, (P1, P2, P3, yy) in (("val", (P1v, P2v, P3v, yval)), ("test", (P1t, P2t, P3t, yte))):
            P, stg = casc(P1, P2, P3, tau); pred = classes[P.argmax(1)]
            row[tag] = dict(accuracy=float((pred == yy).mean()), macro_f1=float(f1_score(yy, pred, average="macro", zero_division=0)),
                            n_errors=int((pred != yy).sum()), stage_share=[float((stg == s).mean()) for s in (1, 2, 3)])
            if lat:
                sh = row[tag]["stage_share"]
                row[tag]["modeled_mean_ms"] = sh[0] * lat["dt"] + sh[1] * (lat["dt"] + lat["hgb"]) + sh[2] * (lat["dt"] + lat["hgb"] + lat["rf"])
        sweep.append(row)
    dump(dict(calibration=cal, cascade_sweep=sweep, n_sub=int(len(sub)), n_val=int(len(val)), n_test=int(len(yte)),
              bins=15, bootstrap=CFG["B"]), os.path.join(out, "calib.json"))
    np.savez_compressed(os.path.join(out, "calib_probas.npz"), y=yte.astype(np.int8),
                        **{f"{k}__{v}": a for k, d in store.items() for v, a in d.items()})

# ----------------------------------------------------------------------------
# stage: prior shift (raw class proportions)
# ----------------------------------------------------------------------------
def stage_prior(out):
    z = np.load(os.path.join(out, "main_preds.npz"), allow_pickle=True)
    y = z["y"].astype(np.int64); names = [str(n) for n in z["names"]]
    pi_raw = np.array([RAW_COUNTS[c] for c in range(C)], float); pi_raw /= pi_raw.sum()
    pi_te = np.bincount(y, minlength=C) / len(y); wc = pi_raw / pi_te; w = wc[y]
    res = {}
    for n in names:
        k = "proba__" + n
        if k not in z.files: continue
        P = z[k].astype(np.float64); pred = P.argmax(1); conf = P.max(1); ok = pred == y
        row = dict(balanced=calib_summary(y, P, boot=0), reweighted=calib_summary(y, P, w=w, boot=0))
        for key in ("bin_counts",): row["balanced"].pop(key, None); row["reweighted"].pop(key, None)
        thr = []
        for t in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995):
            acc_m = conf >= t
            cov = float((w * acc_m).sum() / w.sum())
            err_acc = float((w * acc_m * (~ok)).sum() / max((w * acc_m).sum(), 1e-12))
            ben = y == 0
            fpr = float(((pred != 0) & ben & acc_m).sum() / ben.sum())          # benign flagged as attack (class-conditional)
            miss = float(((pred == 0) & ~ben & acc_m).sum() / (~ben).sum())      # attack passed as benign
            att_alert = (pred != 0) & acc_m
            prec = float((w * att_alert * ~ben).sum() / max((w * att_alert).sum(), 1e-12))   # alert precision at raw prevalence
            thr.append(dict(threshold=t, coverage=cov, error_among_accepted=err_acc, benign_false_alarm_rate=fpr,
                            attack_miss_rate=miss, alert_precision_raw_prior=prec,
                            abstain_share=float(1 - cov)))
        row["thresholds"] = thr; res[n] = row
    dump(dict(weights_per_class={CLASS_NAMES[c]: float(wc[c]) for c in range(C)},
              raw_share={CLASS_NAMES[c]: float(pi_raw[c]) for c in range(C)},
              test_share={CLASS_NAMES[c]: float(pi_te[c]) for c in range(C)}, models=res,
              note="Importance weights pi_raw(c)/pi_working(c). Valid if the downsampling was random within class."),
         os.path.join(out, "prior.json"))
    log("prior done")

# ----------------------------------------------------------------------------
# stage: repeated and blocked validation
# ----------------------------------------------------------------------------
def fit_four(Xtr, ytr, Xte, yte, seed):
    cl = np.unique(ytr)
    dt = mk_dt(seed); dt.fit(Xtr, ytr); hgb = mk_hgb(seed); hgb.fit(Xtr, ytr); rf = mk_rf(seed); rf.fit(Xtr, ytr)
    casc = ConfidenceCascade.from_fitted(dt, hgb, rf, cl)
    out = {}
    for n, m in (("Decision Tree", dt), ("Random Forest", rf), ("HGB", hgb), ("Confidence Cascade", casc)):
        p = m.predict(Xte); q = quality(yte, p); out[n] = q
    return out

def stage_repeat(D, out):
    rows = []
    for s in range(1, CFG["n_seeds"] + 1):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s)
        tr, te = next(gss.split(D.dd[D.feat60], D.y, groups=D.groups))
        Xtr, Xte, ytr, yte = xy(D, D.feat60, tr, te, scale=True)
        r = fit_four(Xtr, ytr, Xte, yte, s)
        for n, q in r.items(): rows.append(dict(design="grouped_seed", rep=s, model=n, **q))
        log(f"repeat grouped seed {s}: " + " ".join(f"{n[:6]}={q['accuracy']:.5f}" for n, q in r.items()))
    for f, (tr, te, purged) in enumerate(blocked_folds(D, CFG["n_folds"]), 1):
        Xtr, Xte, ytr, yte = xy(D, D.feat60, tr, te, scale=True)
        r = fit_four(Xtr, ytr, Xte, yte, SEED)
        for n, q in r.items(): rows.append(dict(design="blocked_time_fold", rep=f, model=n, purged=purged, **q))
        log(f"repeat blocked fold {f}: " + " ".join(f"{n[:6]}={q['accuracy']:.5f}" for n, q in r.items()))
    df = pd.DataFrame(rows); summ = {}
    for design, g in df.groupby("design"):
        summ[design] = {}
        for model, gm in g.groupby("model"):
            summ[design][model] = {k: dict(mean=float(gm[k].mean()), sd=float(gm[k].std(ddof=1)) if len(gm) > 1 else 0.0)
                                   for k in ("accuracy", "macro_f1", "balanced_accuracy", "n_errors")}
        ref = g[g.model == "Decision Tree"].set_index("rep")
        for model, gm in g.groupby("model"):
            if model == "Decision Tree": continue
            d = gm.set_index("rep")["accuracy"] - ref["accuracy"]
            n = len(d); se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
            tcrit = sps.t.ppf(0.975, n - 1) if n > 1 else float("nan")
            summ[design][model]["paired_accuracy_diff_vs_DT"] = dict(mean=float(d.mean()), ci95=[float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)])
    dump(dict(rows=rows, summary=summ), os.path.join(out, "repeat.json"))

# ----------------------------------------------------------------------------
# stage: figures
# ----------------------------------------------------------------------------
def stage_figs(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 9, "font.family": "serif"})
    M = load(os.path.join(out, "main.json")); res = M["results"]; names = list(res)
    lat = np.array([res[n]["latency"]["median_ms"] for n in names]); f1 = np.array([res[n]["macro_f1"] for n in names])
    nd = set(M["pareto"]["f1_vs_median_latency"])
    # Fig: Pareto, numbered points + key
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for i, n in enumerate(names):
        on = n in nd
        ax[0].scatter(lat[i], f1[i], s=80 if on else 45, c="#c0392b" if on else "#9aa0a6", edgecolor="k", lw=.5, zorder=3)
        ax[0].annotate(str(i + 1), (lat[i], f1[i]), ha="center", va="center", fontsize=6, color="white" if on else "black", zorder=4)
    pts = sorted([(lat[i], f1[i]) for i, n in enumerate(names) if n in nd])
    ax[0].step([p[0] for p in pts], [p[1] for p in pts], where="post", c="#c0392b", ls="--", lw=1)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Median single-row latency (ms, log scale)"); ax[0].set_ylabel("Macro-F1")
    ax[0].grid(alpha=.3, which="both")
    ax[1].axis("off")
    ax[1].text(0, 1, "\n".join(f"{i + 1:2d}  {n}{'  *' if n in nd else ''}" for i, n in enumerate(names)), va="top", fontsize=7.5, family="monospace")
    ax[1].text(0, 0.02, "* non-dominated (macro-F1 vs median latency)", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_pareto.png"), dpi=300); fig.savefig(os.path.join(out, "fig_pareto.pdf")); plt.close(fig)
    # Fig: accuracy vs Brier (log)
    fig, ax = plt.subplots(figsize=(5.2, 3.6)); k = 0
    for i, n in enumerate(names):
        if "brier" not in res[n]: continue
        ax.scatter(res[n]["accuracy"], res[n]["brier"], s=40, c="#2c7fb8", edgecolor="k", lw=.5)
        ax.annotate(str(i + 1), (res[n]["accuracy"], res[n]["brier"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_yscale("log"); ax.set_xlabel("Accuracy"); ax.set_ylabel("Brier score (log scale)"); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_acc_vs_brier.png"), dpi=300); plt.close(fig)
    # Fig: reliability panels (raw)
    z = np.load(os.path.join(out, "main_preds.npz"), allow_pickle=True); y = z["y"].astype(np.int64)
    show = ["Decision Tree (single)", "Random Forest (single)", "Confidence Cascade", "Soft Voting", "AdaBoost (SAMME)", "Stacking (LR meta)"]
    fig, axs = plt.subplots(2, 3, figsize=(8.6, 5.4), sharex=True, sharey=True)
    for a, n in zip(axs.ravel(), show):
        P = z["proba__" + n].astype(np.float64); conf = P.max(1); ok = (P.argmax(1) == y).astype(float)
        b = ece_parts(conf, ok); cnt = np.bincount(b, minlength=15); acc = np.bincount(b, weights=ok, minlength=15) / np.maximum(cnt, 1)
        cf = np.bincount(b, weights=conf, minlength=15) / np.maximum(cnt, 1); m = cnt > 0
        a.plot([0, 1], [0, 1], "k--", lw=.8); a.scatter(cf[m], acc[m], s=10 + 90 * np.sqrt(cnt[m] / cnt.max()), c="#2c7fb8", edgecolor="k", lw=.4)
        a.plot(cf[m], acc[m], c="#2c7fb8", lw=.8)
        a.set_title(f"{n}\nECE={res[n]['ece']:.4f}, conf minus acc={res[n]['gap_conf_minus_acc']:+.3f}", fontsize=7.5); a.grid(alpha=.3)
    for a in axs[1]: a.set_xlabel("Mean confidence")
    for a in axs[:, 0]: a.set_ylabel("Accuracy")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_reliability.png"), dpi=300); plt.close(fig)
    # Fig: calibration before/after
    cp = os.path.join(out, "calib_probas.npz")
    if os.path.exists(cp):
        zc = np.load(cp); yc = zc["y"].astype(np.int64); Cj = load(os.path.join(out, "calib.json"))["calibration"]
        mods = [m for m in ("AdaBoost (SAMME)", "Soft Voting", "Feature-View Ensemble") if m in Cj]
        fig, axs = plt.subplots(1, len(mods), figsize=(3.2 * len(mods), 3.3), sharey=True)
        axs = np.atleast_1d(axs)
        for a, n in zip(axs, mods):
            a.plot([0, 1], [0, 1], "k--", lw=.8)
            for v, col in (("raw", "#c0392b"), ("temperature", "#2c7fb8"), ("isotonic", "#27ae60")):
                P = zc[f"{n}__{v}"].astype(np.float64); conf = P.max(1); ok = (P.argmax(1) == yc).astype(float)
                b = ece_parts(conf, ok); cnt = np.bincount(b, minlength=15); acc = np.bincount(b, weights=ok, minlength=15) / np.maximum(cnt, 1)
                cf = np.bincount(b, weights=conf, minlength=15) / np.maximum(cnt, 1); m = cnt > 0
                a.plot(cf[m], acc[m], "o-", ms=3, lw=.9, c=col, label=f"{v} (ECE {Cj[n][v]['ece']:.4f})")
            a.set_title(n, fontsize=8); a.set_xlabel("Mean confidence"); a.grid(alpha=.3); a.legend(fontsize=6.5, loc="lower right")
        axs[0].set_ylabel("Accuracy"); fig.tight_layout(); fig.savefig(os.path.join(out, "fig_calibration_after.png"), dpi=300); plt.close(fig)
    log("figures done")

# ----------------------------------------------------------------------------
def make_synthetic(path, n_per=1500, seed=0):
    from sklearn.datasets import make_classification
    cols = ['ts', 'Payload_Length', 'Var_Payload', 'Protocol Type', 'Duration', 'Entropy', 'Drone_port', 'Rate', 'Srate', 'Drate',
            'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 'urg_flag_number',
            'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count', 'max_duration',
            'min_duration', 'sum_duration', 'average_duration', 'std_duration', 'CoAP', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
            'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT',
            'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight', 'DS status', 'Fragments', 'Sequence number',
            'flow_idle_time', 'flow_active_time']
    X, y = make_classification(n_samples=n_per * 10, n_features=len(cols), n_informative=20, n_redundant=5, n_classes=10,
                               n_clusters_per_class=1, class_sep=2.0, flip_y=0.002, random_state=seed)
    df = pd.DataFrame(X, columns=cols); df["Label"] = y
    rng = np.random.default_rng(seed)
    df["ts"] = df.groupby("Label").cumcount() * 1e-4 + df["Label"] * 0.5
    df["Sequence number"] = df["ts"] * 100 + rng.normal(size=len(df)) * 0.01
    df = df.round(4); df = pd.concat([df, df.iloc[:3]]).reset_index(drop=True)
    df.to_csv(path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--out", default="revision_out")
    ap.add_argument("--stages", default="all"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny settings to test the pipeline")
    ap.add_argument("--synthetic", action="store_true", help="create a synthetic CSV at --csv first (smoke test only)")
    ap.add_argument("--calib-stacking", action="store_true"); ap.add_argument("--nrows", type=int)
    a = ap.parse_args()
    if a.smoke:
        CFG.update(smoke=True, trees=15, hgb_iter=20, bag_trees=8, ada_trees=15, n_lat=25, n_warm=3, B=60, n_seeds=2, n_folds=3, pareto_B=20)
    os.makedirs(a.out, exist_ok=True)
    if a.synthetic: make_synthetic(a.csv)
    stages = ["prep", "main", "stats", "ablation", "calib", "prior", "repeat", "figs"] if a.stages == "all" else a.stages.split(",")
    need_data = any(s in stages for s in ("prep", "main", "ablation", "calib", "repeat"))
    D = prepare(a.csv, a.nrows) if need_data else None
    def todo(name, fn):
        path = os.path.join(a.out, name + ".json")
        if os.path.exists(path) and not a.force: log(f"skip {name} (exists)"); return
        t0 = time.time(); fn(); log(f"stage {name} finished in {(time.time() - t0) / 60:.1f} min")
    if "prep" in stages: todo("prep", lambda: stage_prep(D, a.out))
    if "main" in stages: todo("main", lambda: stage_main(D, a.out))
    if "stats" in stages: todo("stats", lambda: stage_stats(a.out))
    if "ablation" in stages: todo("ablation", lambda: stage_ablation(D, a.out))
    if "calib" in stages: todo("calib", lambda: stage_calib(D, a.out, a.calib_stacking))
    if "prior" in stages: todo("prior", lambda: stage_prior(a.out))
    if "repeat" in stages: todo("repeat", lambda: stage_repeat(D, a.out))
    if "figs" in stages: stage_figs(a.out)
    log("ALL DONE. Zip the folder", a.out, "and send it back.")

if __name__ == "__main__":
    main()
