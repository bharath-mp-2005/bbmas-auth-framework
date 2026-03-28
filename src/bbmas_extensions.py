"""
BB-MAS Extension Experiments
==============================
Two independent experiments that extend the main evaluation:

  Experiment A — Drift Simulation
  --------------------------------
  Validates the EMA adaptation claim in the paper (Section VII-D / Table V).
  Genuine chunks are split CHRONOLOGICALLY into early (first 30%) and late
  (final 70%), simulating temporal behavioral drift.  A model is trained on
  early chunks then evaluated twice:
    1. No adaptation  : late chunks normalised with static training-time stats
    2. EMA adaptation : late chunks normalised with running EMA that updates
                        after each prediction (online, as in the browser)

  Expected outcome: EMA-adapted EER ≈ 1-3 pp lower than no-adapt EER.
  This directly validates the 0.49% vs 1.23% single-user claim in the paper.

  Experiment B — Adversarial Mimicry
  ------------------------------------
  Tests whether a "dangerous" impostor who deliberately imitates the target
  can fool the model.  For each target user:
    1. Find the closest natural impostor (minimum L1 distance between mean
       feature vectors — the "most similar" user in the dataset).
    2. Create synthetic mimic samples: linear blend of closest-impostor
       features (80%) and target-genuine features (20%).
    3. Evaluate the trained model with:
       a. Natural impostors only  (standard LOUO — baseline)
       b. Mixed impostors: 50% natural + 50% mimic           (partial threat)
       c. Mimic impostors only                               (worst case)

  Expected outcome: EER rises 5-15 pp under full mimicry.  Reporting
  "robust EER" (mimic-only) alongside standard EER is required by TIFS/USENIX.

Run AFTER the main bbmas_auth_final.py loop (needs key_df already loaded),
OR run standalone — it reloads the data itself.


"""

import os
import random

# ── Determinism block — must come before numpy/tensorflow imports ──
os.environ['TF_DETERMINISTIC_OPS']   = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED']         = '42'
random.seed(42)

import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from scipy.special import entr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# CONFIG  — must match bbmas_auth_final.py
# ==============================================================================
BBMAS_ROOT         = r"C:\Users\mpbmp\Desktop\privacy-preserving-ca\bbmas-ca-framework\BB-MAS_Dataset"
CHUNK_KEY          = 50
MIN_GENUINE_CHUNKS = 15

# Drift experiment
EARLY_FRAC  = 0.30   # first 30% of user's chunks  -> training
# remaining 70% -> test (simulates late-session / new-keyboard drift)

# Adversarial experiment
MIMIC_BLEND = 0.80   # 80% closest-impostor + 20% target genuine
MIMIC_FRAC  = 0.50   # fraction of test impostors replaced with mimic samples


# ==============================================================================
# HELPERS & FEATURE EXTRACTOR  (identical to main script)
# ==============================================================================

def _safe_entropy(arr):
    counts, _ = np.histogram(arr, bins=10, density=False)
    p = counts / (counts.sum() + 1e-10)
    return float(np.sum(entr(p + 1e-10)))

def _bigram_stats(arr):
    if len(arr) < 2:
        return 0.0, 0.0, 0.0
    pairs = arr[:-1] + arr[1:]
    return float(np.mean(pairs)), float(np.std(pairs)), _safe_entropy(pairs)

def _fft_dominant(arr, top_k=3):
    if len(arr) < 4:
        return [0.0] * top_k
    mag = np.abs(fft(arr))[:len(arr) // 2]
    mag = mag / (mag.sum() + 1e-10)
    top = sorted(mag, reverse=True)[:top_k]
    while len(top) < top_k:
        top.append(0.0)
    return [float(v) for v in top]

def get_keystroke_features(press, release, user_id):
    """20-dim keystroke feature vector (identical to main script)."""
    if len(press) < 5 or len(release) < 5:
        return None
    dwell_times = []
    j = 0
    for i in range(len(press)):
        p_time = press['time'].iloc[i]
        while j < len(release) and release['time'].iloc[j] <= p_time:
            j += 1
        if j < len(release):
            d = (release['time'].iloc[j] - p_time).total_seconds() * 1000
            if 0 < d < 5000:
                dwell_times.append(d)
    if len(dwell_times) < 5:
        return None
    dwell    = np.array(dwell_times)
    press_ms = press['time'].values.astype(np.int64) / 1e6
    flight   = np.diff(press_ms)
    flight   = flight[(flight > 0) & (flight < 5000)]
    duration    = (press['time'].iloc[-1] - press['time'].iloc[0]).total_seconds()
    typing_rate = len(dwell) / (duration + 1e-5)
    bg_mean, bg_std, bg_ent = _bigram_stats(list(dwell))
    fft_d = _fft_dominant(dwell)
    return {
        'user': user_id,
        'key_dwell_mean':  float(np.mean(dwell)),
        'key_dwell_std':   float(np.std(dwell)),
        'key_dwell_cv':    float(np.std(dwell) / (np.mean(dwell) + 1e-5)),
        'key_dwell_skew':  float(skew(dwell)),
        'key_dwell_kurt':  float(kurtosis(dwell)),
        'key_dwell_ent':   _safe_entropy(dwell),
        'key_bigram_mean': bg_mean,
        'key_bigram_std':  bg_std,
        'key_bigram_ent':  bg_ent,
        'key_fft1':        fft_d[0],
        'key_fft2':        fft_d[1],
        'key_fft3':        fft_d[2],
        'key_flight_mean': float(np.mean(flight))   if len(flight) > 0 else 0.0,
        'key_flight_std':  float(np.std(flight))    if len(flight) > 0 else 0.0,
        'key_flight_cv':   float(np.std(flight) / (np.mean(flight) + 1e-5)) if len(flight) > 0 else 0.0,
        'key_flight_skew': float(skew(flight))      if len(flight) > 2 else 0.0,
        'key_flight_kurt': float(kurtosis(flight))  if len(flight) > 2 else 0.0,
        'key_flight_ent':  _safe_entropy(flight)    if len(flight) > 2 else 0.0,
        'key_typing_rate': typing_rate,
        'key_n_keys':      float(len(dwell)),
    }


# ==============================================================================
# DATA LOADING  (keystroke only — both experiments use only keystroke)
# ==============================================================================

def load_keystroke_chunked_ordered():
    """
    Load keystroke features preserving TEMPORAL ORDER within each user.
    Each row carries a chunk_index so chronological splitting is exact.
    """
    rows = []
    user_dirs = sorted([d for d in os.listdir(BBMAS_ROOT) if d.isdigit()], key=int)
    for uid_str in user_dirs:
        uid  = int(uid_str)
        udir = os.path.join(BBMAS_ROOT, uid_str)
        kfile = os.path.join(udir, f"{uid}_Desktop_Keyboard.csv")
        if not os.path.exists(kfile):
            continue
        try:
            df = pd.read_csv(kfile)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
            chunk_idx = 0
            for i in range(0, len(df), CHUNK_KEY):
                chunk   = df.iloc[i:i + CHUNK_KEY]
                press   = chunk[chunk['direction'] == 0].copy()
                release = chunk[chunk['direction'] == 1].copy()
                feat    = get_keystroke_features(press, release, uid)
                if feat:
                    feat['chunk_order'] = chunk_idx   # <-- temporal position
                    rows.append(feat)
                    chunk_idx += 1
        except Exception:
            pass
    return pd.DataFrame(rows)


# ==============================================================================
# SHARED UTILITIES
# ==============================================================================

def apply_smote(X, y, random_state=42):
    n_genuine = int(np.sum(y == 1))
    if n_genuine < 2:
        return X, y
    k = min(5, n_genuine - 1)
    try:
        sm = SMOTE(k_neighbors=k, random_state=random_state)
        return sm.fit_resample(X, y)
    except Exception:
        repeat  = max(1, int(np.sum(y == 0)) // n_genuine)
        gen_idx = np.where(y == 1)[0]
        X2 = np.vstack([X, np.tile(X[gen_idx], (repeat, 1))])
        y2 = np.concatenate([y, np.ones(len(gen_idx) * repeat)])
        return X2, y2

def build_mlp(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x   = tf.keras.layers.Dense(128, activation='relu')(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(64, activation='relu')(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    mdl = tf.keras.Model(inp, out)
    mdl.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss='binary_crossentropy', metrics=['accuracy'])
    return mdl

def train_mlp(Xtr, ytr):
    """Train and return fitted model + scaler (scaler fitted on Xtr)."""
    cw  = dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
    ))
    mdl = build_mlp(Xtr.shape[1])
    es  = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )
    mdl.fit(Xtr, ytr, epochs=40, batch_size=64, class_weight=cw,
            validation_split=0.15, callbacks=[es], verbose=0)
    return mdl

def compute_eer(scores, labels):
    """Returns EER as a fraction (0–1)."""
    if len(np.unique(labels)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx])

def compute_auc(scores, labels):
    if len(np.unique(labels)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return float(sk_auc(fpr, tpr))


# ==============================================================================
# EXPERIMENT A — DRIFT SIMULATION
# ==============================================================================

def run_drift_experiment(key_df_ordered, feature_cols):
    """
    For each user:
      - Sort genuine chunks by chunk_order (chronological)
      - Train on first EARLY_FRAC genuine chunks + all training impostors
      - Test late genuine chunks against held-out impostor with:
          (1) Static normalisation (training-time mean/std, no update)
          (2) Online EMA normalisation (alpha=0.9, updated chunk-by-chunk)

    Returns DataFrame with per-user results.
    """
    print(f"\n{'='*65}")
    print("  EXPERIMENT A — DRIFT SIMULATION")
    print(f"  Early split: {EARLY_FRAC:.0%} | Late test: {1-EARLY_FRAC:.0%}")
    print(f"{'='*65}\n")

    results = []
    users   = key_df_ordered['user'].unique()
    users   = [u for u in sorted(users)
               if (key_df_ordered['user'] == u).sum() >= MIN_GENUINE_CHUNKS]

    for u_idx, u in enumerate(users, 1):
        # ── collect indices ───────────────────────────────────────────────────
        user_mask    = key_df_ordered['user'] == u
        genuine_rows = key_df_ordered[user_mask].sort_values('chunk_order')
        n_genuine    = len(genuine_rows)

        if n_genuine < MIN_GENUINE_CHUNKS:
            continue

        early_n   = max(2, int(EARLY_FRAC * n_genuine))
        early_idx = genuine_rows.index[:early_n]
        late_idx  = genuine_rows.index[early_n:]

        if len(late_idx) < 3:
            continue

        # ── impostor pool (all other users) ──────────────────────────────────
        impostor_rows = key_df_ordered[~user_mask]
        imp_users = sorted([x for x in impostor_rows['user'].unique() if x != u])

        rng      = np.random.RandomState(42)
        held_out = rng.choice(imp_users)
        imp_train = impostor_rows[impostor_rows['user'] != held_out]
        imp_test  = impostor_rows[impostor_rows['user'] == held_out]

        if len(imp_test) == 0:
            continue

        # ── build raw arrays ──────────────────────────────────────────────────
        X_early  = key_df_ordered.loc[early_idx, feature_cols].values
        y_early  = np.ones(len(X_early))

        X_imp_tr = imp_train[feature_cols].values
        y_imp_tr = np.zeros(len(X_imp_tr))

        X_late   = key_df_ordered.loc[late_idx, feature_cols].values
        y_late   = np.ones(len(X_late))

        X_imp_te = imp_test[feature_cols].values
        y_imp_te = np.zeros(len(X_imp_te))

        X_train_raw = np.vstack([X_early, X_imp_tr])
        y_train     = np.concatenate([y_early, y_imp_tr])

        if len(np.unique(y_train)) < 2:
            continue

        # ── fit scaler on TRAINING data ───────────────────────────────────────
        sc = StandardScaler()
        X_train_sc = sc.fit_transform(X_train_raw)

        # Apply SMOTE to training set
        X_train_sc, y_train = apply_smote(X_train_sc, y_train)
        if len(np.unique(y_train)) < 2:
            continue

        # ── train model ───────────────────────────────────────────────────────
        mdl = train_mlp(X_train_sc, y_train)

        # ── TEST 1: No adaptation — static scaler ─────────────────────────────
        X_test_combined = np.vstack([X_late, X_imp_te])
        y_test_combined = np.concatenate([y_late, y_imp_te])

        X_test_sc    = sc.transform(X_test_combined)
        scores_static = mdl.predict(X_test_sc, verbose=0).ravel()
        eer_no_adapt  = compute_eer(scores_static, y_test_combined)
        auc_no_adapt  = compute_auc(scores_static, y_test_combined)

        # ── TEST 2: EMA adaptation — online normalisation ─────────────────────
        # Initialise EMA from training statistics (raw feature space)
        ema_mu  = sc.mean_.copy()        # training mean per feature
        ema_std = sc.scale_.copy()       # training std per feature
        alpha   = 0.9

        scores_ema = []
        for feat_raw in X_test_combined:
            # Normalise with current EMA
            feat_norm = (feat_raw - ema_mu) / (ema_std + 1e-5)
            score     = float(mdl.predict(feat_norm.reshape(1, -1), verbose=0))
            scores_ema.append(score)
            # CRITICAL: Only update EMA when model is confident the window is
            # genuine (score > 0.5 = model says "this looks like the enrolled user").
            # Updating on impostor windows would shift the baseline toward impostor
            # feature distributions and corrupt future normalisation — exactly the
            # bug that caused EMA to hurt 85% of users in the uncorrected version.
            # In a real browser deployment this is the only sensible strategy:
            # you can't trust unlabelled windows that the model flags as suspicious.
            if score > 0.5:
                ema_mu  = alpha * ema_mu  + (1 - alpha) * feat_raw
                ema_std = alpha * ema_std + (1 - alpha) * np.abs(feat_raw - ema_mu)

        scores_ema   = np.array(scores_ema)
        eer_ema      = compute_eer(scores_ema, y_test_combined)
        auc_ema      = compute_auc(scores_ema, y_test_combined)
        improvement  = eer_no_adapt - eer_ema   # positive = EMA helped

        results.append({
            'user':           u,
            'n_early_chunks': early_n,
            'n_late_chunks':  len(late_idx),
            'EER_no_adapt':   eer_no_adapt * 100,
            'AUC_no_adapt':   auc_no_adapt,
            'EER_EMA':        eer_ema       * 100,
            'AUC_EMA':        auc_ema,
            'EER_improvement_pp': improvement * 100,
            'EMA_helped':     improvement > 0,
        })

        if u_idx % 10 == 0 or u_idx <= 3:
            print(f"  [{u_idx:2d}/{len(users)}] User {u:3d} | "
                  f"No-adapt: {eer_no_adapt*100:5.2f}%  "
                  f"EMA: {eer_ema*100:5.2f}%  "
                  f"Δ: {improvement*100:+.2f}pp", flush=True)

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  ERROR: No results collected.")
        return df

    n = len(df)
    helped = df['EMA_helped'].sum()
    mean_no = df['EER_no_adapt'].mean()
    std_no  = df['EER_no_adapt'].std()
    mean_em = df['EER_EMA'].mean()
    std_em  = df['EER_EMA'].std()
    mean_imp = df['EER_improvement_pp'].mean()

    print(f"\n{'─'*55}")
    print(f"  DRIFT RESULTS  ({n} users)")
    print(f"{'─'*55}")
    print(f"  No-adapt EER   : {mean_no:.2f}% ± {std_no:.2f}%")
    print(f"  EMA-adapt EER  : {mean_em:.2f}% ± {std_em:.2f}%")
    print(f"  Mean Δ (pp)    : {mean_imp:+.2f}pp")
    print(f"  EMA helped     : {helped}/{n} users ({helped/n*100:.1f}%)")

    # ── LaTeX snippet ─────────────────────────────────────────────────────────
    print(f"\n  LATEX — Drift rows for Table V:")
    print(f"  No drift adapt. & {mean_no:.2f} & {std_no:.2f} & "
          f"{df['AUC_no_adapt'].mean():.3f} \\\\")
    print(f"  \\textbf{{+ EMA adaptation}} & \\textbf{{{mean_em:.2f}}} & "
          f"\\textbf{{{std_em:.2f}}} & \\textbf{{{df['AUC_EMA'].mean():.3f}}} \\\\")

    df.to_csv("paper_drift_results.csv", index=False)
    print(f"\n  Saved: paper_drift_results.csv")
    return df


# ==============================================================================
# EXPERIMENT B — ADVERSARIAL MIMICRY
# ==============================================================================

def run_adversarial_experiment(key_df, feature_cols):
    """
    For each target user:
      1. Find 'closest' impostor (min mean L1 distance between feature vectors)
      2. Generate synthetic mimic samples: MIMIC_BLEND*closest + (1-BLEND)*target
      3. Evaluate the trained model under three test conditions:
           (a) Natural LOUO impostors only           — baseline
           (b) Mixed: 50% natural + 50% mimic        — partial threat
           (c) Mimic impostors only                  — worst-case "robust EER"

    Returns DataFrame with per-user results.
    """
    print(f"\n{'='*65}")
    print("  EXPERIMENT B — ADVERSARIAL MIMICRY")
    print(f"  Blend: {MIMIC_BLEND:.0%} closest-impostor + "
          f"{1-MIMIC_BLEND:.0%} target genuine")
    print(f"{'='*65}\n")

    results = []
    users   = key_df['user'].unique()
    users   = [u for u in sorted(users)
               if (key_df['user'] == u).sum() >= MIN_GENUINE_CHUNKS]

    for u_idx, u in enumerate(users, 1):
        # ── standard LOUO split ───────────────────────────────────────────────
        y_all     = (key_df['user'] == u).astype(int).values
        X_all     = key_df[feature_cols].values
        users_all = key_df['user'].values

        genuine_idx  = np.where(y_all == 1)[0]
        impostor_idx = np.where(y_all == 0)[0]

        if len(genuine_idx) < MIN_GENUINE_CHUNKS:
            continue

        rng = np.random.RandomState(42)
        rng.shuffle(genuine_idx)
        split     = int(0.7 * len(genuine_idx))
        gen_train = genuine_idx[:split]
        gen_test  = genuine_idx[split:]

        imp_users = sorted([x for x in key_df['user'].unique() if x != u])
        held_out  = rng.choice(imp_users)
        imp_train_idx = impostor_idx[users_all[impostor_idx] != held_out]
        imp_test_idx  = impostor_idx[users_all[impostor_idx] == held_out]

        if len(gen_test) == 0 or len(imp_test_idx) == 0:
            continue

        # ── train model (standard pipeline) ───────────────────────────────────
        Xtr_raw = np.vstack([X_all[gen_train], X_all[imp_train_idx]])
        ytr     = np.concatenate([np.ones(len(gen_train)),
                                  np.zeros(len(imp_train_idx))])

        sc       = StandardScaler()
        Xtr      = sc.fit_transform(Xtr_raw)
        Xtr, ytr = apply_smote(Xtr, ytr)

        if len(np.unique(ytr)) < 2:
            continue

        mdl = train_mlp(Xtr, ytr)

        # ── build genuine test set ────────────────────────────────────────────
        X_gen_test = X_all[gen_test]
        y_gen_test = np.ones(len(gen_test))

        # ── natural impostor test set ─────────────────────────────────────────
        X_nat_imp  = X_all[imp_test_idx]
        y_nat_imp  = np.zeros(len(imp_test_idx))

        # ── find closest impostor by mean L1 distance ─────────────────────────
        X_genuine_mean = X_all[genuine_idx].mean(axis=0)
        closest_dist   = np.inf
        closest_user   = None

        for imp_u in imp_users:
            imp_mask  = users_all == imp_u
            imp_feats = X_all[imp_mask]
            if len(imp_feats) == 0:
                continue
            dist = np.mean(np.abs(X_genuine_mean - imp_feats.mean(axis=0)))
            if dist < closest_dist:
                closest_dist = dist
                closest_user = imp_u

        if closest_user is None:
            continue

        # ── generate mimic samples ────────────────────────────────────────────
        X_closest = X_all[users_all == closest_user]
        n_mimic   = max(len(X_nat_imp), 20)   # at least as many as natural test

        # Sample with replacement from closest-impostor pool
        rng2       = np.random.RandomState(0)
        close_samp = X_closest[rng2.choice(len(X_closest), n_mimic, replace=True)]
        # Sample from genuine pool (target's own chunks)
        gen_samp   = X_all[genuine_idx][
            rng2.choice(len(genuine_idx), n_mimic, replace=True)
        ]

        # Blended mimic: MIMIC_BLEND fraction from closest-impostor
        X_mimic  = MIMIC_BLEND * close_samp + (1 - MIMIC_BLEND) * gen_samp
        y_mimic  = np.zeros(n_mimic)   # label as impostor

        # ── CONDITION A: Natural impostors only (LOUO baseline) ───────────────
        X_test_a = np.vstack([X_gen_test, X_nat_imp])
        y_test_a = np.concatenate([y_gen_test, y_nat_imp])
        sc_a     = sc.transform(X_test_a)
        eer_a    = compute_eer(mdl.predict(sc_a, verbose=0).ravel(), y_test_a)
        auc_a    = compute_auc(mdl.predict(sc_a, verbose=0).ravel(), y_test_a)

        # ── CONDITION B: Mixed — 50% natural + 50% mimic ──────────────────────
        n_mix      = min(len(X_nat_imp), len(X_mimic))
        X_mixed_imp = np.vstack([X_nat_imp[:n_mix], X_mimic[:n_mix]])
        y_mixed_imp = np.zeros(len(X_mixed_imp))
        X_test_b    = np.vstack([X_gen_test, X_mixed_imp])
        y_test_b    = np.concatenate([y_gen_test, y_mixed_imp])
        sc_b        = sc.transform(X_test_b)
        eer_b       = compute_eer(mdl.predict(sc_b, verbose=0).ravel(), y_test_b)
        auc_b       = compute_auc(mdl.predict(sc_b, verbose=0).ravel(), y_test_b)

        # ── CONDITION C: Mimic impostors only (worst-case / "robust EER") ─────
        X_test_c = np.vstack([X_gen_test, X_mimic])
        y_test_c = np.concatenate([y_gen_test, y_mimic])
        sc_c     = sc.transform(X_test_c)
        eer_c    = compute_eer(mdl.predict(sc_c, verbose=0).ravel(), y_test_c)
        auc_c    = compute_auc(mdl.predict(sc_c, verbose=0).ravel(), y_test_c)

        degradation = eer_c - eer_a  # positive = mimic is harder to detect

        results.append({
            'user':               u,
            'closest_impostor':   closest_user,
            'closest_L1_dist':    closest_dist,
            'EER_natural':        eer_a * 100,
            'AUC_natural':        auc_a,
            'EER_mixed':          eer_b * 100,
            'AUC_mixed':          auc_b,
            'EER_mimic':          eer_c * 100,   # <-- "robust EER" for paper
            'AUC_mimic':          auc_c,
            'EER_degradation_pp': degradation * 100,
            'model_fooled':       degradation > 0.05,  # >5pp rise = meaningful
        })

        if u_idx % 10 == 0 or u_idx <= 3:
            print(f"  [{u_idx:2d}/{len(users)}] User {u:3d} | "
                  f"Natural: {eer_a*100:5.2f}%  "
                  f"Mixed: {eer_b*100:5.2f}%  "
                  f"Mimic: {eer_c*100:5.2f}%  "
                  f"Δ: {degradation*100:+.2f}pp", flush=True)

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  ERROR: No results collected.")
        return df

    n         = len(df)
    fooled    = df['model_fooled'].sum()
    mean_nat  = df['EER_natural'].mean()
    std_nat   = df['EER_natural'].std()
    mean_mix  = df['EER_mixed'].mean()
    std_mix   = df['EER_mixed'].std()
    mean_mim  = df['EER_mimic'].mean()
    std_mim   = df['EER_mimic'].std()
    mean_deg  = df['EER_degradation_pp'].mean()

    print(f"\n{'─'*55}")
    print(f"  ADVERSARIAL RESULTS  ({n} users)")
    print(f"{'─'*55}")
    print(f"  Natural LOUO EER   : {mean_nat:.2f}% ± {std_nat:.2f}%")
    print(f"  Mixed (50/50) EER  : {mean_mix:.2f}% ± {std_mix:.2f}%")
    print(f"  Mimic-only EER     : {mean_mim:.2f}% ± {std_mim:.2f}%  <- Robust EER")
    print(f"  Mean degradation   : {mean_deg:+.2f}pp")
    print(f"  Users fooled (>5pp): {fooled}/{n} ({fooled/n*100:.1f}%)")

    if mean_deg < 5:
        verdict = "ROBUST: Mimicry provides negligible improvement for the attacker."
    elif mean_deg < 15:
        verdict = "MODERATE: Mimicry degrades performance but system remains partially effective."
    else:
        verdict = ("VULNERABLE: EER nearly doubles under closest-neighbor mimicry. "
                   "Paper must acknowledge this as a limitation and scope the threat model "
                   "to exclude adversaries with behavioral observation capability.")
    print(f"\n  Adversarial verdict: {verdict}")

    # ── LaTeX Table V ─────────────────────────────────────────────────────────
    print(f"\n  LATEX — Table V: Adversarial Robustness Evaluation")
    print(r"  \begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Adversarial Mimicry Robustness (80 Users, Keystroke MLP)}")
    print(r"  \label{tab:adversarial}")
    print(r"  \renewcommand{\arraystretch}{1.15}")
    print(r"  \begin{tabular}{|l|c|c|c|}")
    print(r"  \hline")
    print(r"  \textbf{Impostor Type} & \textbf{Mean EER (\%)} & \textbf{Std (\%)} & \textbf{AUC} \\")
    print(r"  \hline")
    print(f"  Natural (LOUO baseline) & {mean_nat:.2f} & {std_nat:.2f} & "
          f"{df['AUC_natural'].mean():.3f} \\\\")
    print(f"  Mixed (50\\% mimic) & {mean_mix:.2f} & {std_mix:.2f} & "
          f"{df['AUC_mixed'].mean():.3f} \\\\")
    print(r"  \hline")
    print(f"  \\textbf{{Mimic-only (robust EER)}} & \\textbf{{{mean_mim:.2f}}} & "
          f"\\textbf{{{std_mim:.2f}}} & \\textbf{{{df['AUC_mimic'].mean():.3f}}} \\\\")
    print(r"  \hline")
    print(r"  \end{tabular}")
    print(r"  \end{table}")

    df.to_csv("paper_adversarial_results.csv", index=False)
    print(f"\n  Saved: paper_adversarial_results.csv")
    return df


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__" or True:

    print("=" * 65)
    print("  BB-MAS Extension Experiments")
    print("  A: Drift Simulation  |  B: Adversarial Mimicry")
    print("=" * 65)
    print(f"\nLoading data from: {BBMAS_ROOT}\n")

    # Load data with temporal ordering (for drift experiment)
    key_df_ordered = load_keystroke_chunked_ordered()

    # key_df without chunk_order column (for adversarial experiment,
    # which doesn't need temporal order)
    feature_cols = [c for c in key_df_ordered.columns
                    if c not in ('user', 'chunk_order')]
    key_df_plain = key_df_ordered.drop(columns=['chunk_order'])

    assert len(feature_cols) == 20, \
        f"Expected 20 features, got {len(feature_cols)}: {feature_cols}"
    print(f"Features: {len(feature_cols)}-dim  ✓")

    valid_users = [
        u for u in key_df_ordered['user'].unique()
        if (key_df_ordered['user'] == u).sum() >= MIN_GENUINE_CHUNKS
    ]
    print(f"Eligible users: {len(valid_users)}\n")

    # ── Run Experiment A ──────────────────────────────────────────────────────
    drift_df = run_drift_experiment(key_df_ordered, feature_cols)

    # ── Run Experiment B ──────────────────────────────────────────────────────
    adv_df = run_adversarial_experiment(key_df_plain, feature_cols)

    # ── Combined summary for paper ────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  COMBINED SUMMARY FOR PAPER")
    print(f"{'='*65}")

    if len(drift_df) > 0:
        print(f"\n  [A] DRIFT (Table V addition)")
        print(f"      Without EMA : {drift_df['EER_no_adapt'].mean():.2f}% "
              f"± {drift_df['EER_no_adapt'].std():.2f}%")
        print(f"      With EMA    : {drift_df['EER_EMA'].mean():.2f}% "
              f"± {drift_df['EER_EMA'].std():.2f}%")
        print(f"      Improvement : {drift_df['EER_improvement_pp'].mean():+.2f}pp mean  "
              f"({drift_df['EMA_helped'].mean()*100:.1f}% of users benefited)")

    if len(adv_df) > 0:
        print(f"\n  [B] ADVERSARIAL (Table V / new table)")
        print(f"      Standard EER : {adv_df['EER_natural'].mean():.2f}%")
        print(f"      Robust EER   : {adv_df['EER_mimic'].mean():.2f}%")
        print(f"      Degradation  : {adv_df['EER_degradation_pp'].mean():+.2f}pp")

    print(f"\n  Output files:")
    print(f"    paper_drift_results.csv")
    print(f"    paper_adversarial_results.csv")
