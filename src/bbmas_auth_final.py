"""
Privacy-Preserving Continuous Authentication — BB-MAS Dataset
==============================================================
Single consolidated script. Run top-to-bottom in Jupyter or as a .py file.

All fixes consolidated from iterative notebook development:
  1. Enriched features  : skewness, kurtosis, entropy, FFT top-3, bigrams
                          (20-dim keystroke / 14-dim mouse / 10-dim scroll)
  2. CHUNK_MOUSE = 50   : was 200 → ~4× more mouse chunks, 80 eligible users
  3. SMOTE + class-weight: corrects genuine-to-impostor imbalance
  4. LOUO evaluation    : Leave-One-User-Out (held-out impostor user in test)
  5. Score alignment fix: pad_scores() — no label truncation
  6. Three fusion modes : K+M+W simple avg | K+M+W MLP | K+M MLP (no wheel)

Expected results (80 users, LOUO, SMOTE):
  Keystroke only   : ~20.51%  EER   AUC ~0.849   <- best result / paper headline
  Mouse only       : ~36-37%  EER   (high variance, unconstrained web data)
  Wheel only       : AUC ~0.50 (degenerate — excluded from paper tables)
  K+M MLP fusion   : ~24.03%  EER   (fusion does NOT beat keystroke alone)


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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)


# ==============================================================================
# CONFIG  — only edit these lines
# ==============================================================================
BBMAS_ROOT = r"./BB-MAS_Dataset"  # Update this to your local path
CHUNK_KEY           = 50    # keystroke events per chunk
CHUNK_MOUSE         = 50    # mouse events per chunk
CHUNK_WHEEL         = 20    # scroll events per chunk
MIN_GENUINE_CHUNKS  = 15    # drop users with fewer genuine chunks than this


# ==============================================================================
# 1.  HELPER FUNCTIONS
# ==============================================================================

def _safe_entropy(arr):
    """Shannon entropy over a 10-bin normalised histogram."""
    counts, _ = np.histogram(arr, bins=10, density=False)
    p = counts / (counts.sum() + 1e-10)
    return float(np.sum(entr(p + 1e-10)))


def _bigram_stats(arr):
    """Mean and std of consecutive-pair sums (bigram tempo feature)."""
    if len(arr) < 2:
        return 0.0, 0.0
    pairs = arr[:-1] + arr[1:]
    return float(np.mean(pairs)), float(np.std(pairs))


def _fft_dominant(arr, top_k=3):
    """Top-k normalised FFT magnitude components."""
    if len(arr) < 4:
        return [0.0] * top_k
    mag = np.abs(fft(arr))[:len(arr) // 2]
    mag = mag / (mag.sum() + 1e-10)
    top = sorted(mag, reverse=True)[:top_k]
    while len(top) < top_k:
        top.append(0.0)
    return [float(v) for v in top]


# ==============================================================================
# 2.  FEATURE EXTRACTORS
# ==============================================================================

def get_keystroke_features(press, release, user_id):
    """
    20-dimensional enriched keystroke feature vector.
    dwell mean/std/cv/skew/kurt/entropy, bigram mean/std,
    fft1/fft2/fft3, flight mean/std/cv/skew/kurt/entropy,
    typing_rate, n_keys.
    """
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
    bg_mean, bg_std = _bigram_stats(list(dwell))
    fft_d = _fft_dominant(dwell)

    return {
        'user':            user_id,
        'key_dwell_mean':  float(np.mean(dwell)),
        'key_dwell_std':   float(np.std(dwell)),
        'key_dwell_cv':    float(np.std(dwell) / (np.mean(dwell) + 1e-5)),
        'key_dwell_skew':  float(skew(dwell)),
        'key_dwell_kurt':  float(kurtosis(dwell)),
        'key_dwell_ent':   _safe_entropy(dwell),
        'key_bigram_mean': bg_mean,
        'key_bigram_std':  bg_std,
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


def get_mouse_features(df, user_id):
    """
    14-dimensional enriched mouse feature vector.
    speed mean/std/max/skew/kurt/entropy, fft1/fft2/fft3,
    accel_mean/accel_std, dist_total, dir_changes, path_efficiency.
    """
    if len(df) < 5:
        return None

    ts    = df['time'].values.astype(np.int64) / 1e6
    dx    = np.diff(df['pX'].values).astype(float)
    dy    = np.diff(df['pY'].values).astype(float)
    dt    = np.diff(ts) + 1e-5
    dist  = np.sqrt(dx**2 + dy**2)
    speed = dist / dt
    accel = np.diff(speed) / (dt[:-1] + 1e-5) if len(speed) > 1 else np.array([0.0])

    angles      = np.arctan2(dy, dx)
    dir_changes = int(np.sum(np.abs(np.diff(angles)) > 1.0))
    total_dist  = float(np.sum(dist))
    straight    = float(np.sqrt(
        (df['pX'].iloc[-1] - df['pX'].iloc[0])**2 +
        (df['pY'].iloc[-1] - df['pY'].iloc[0])**2
    ))
    fft_s = _fft_dominant(speed)

    return {
        'user':              user_id,
        'mouse_speed_mean':  float(np.mean(speed)),
        'mouse_speed_std':   float(np.std(speed)),
        'mouse_speed_max':   float(np.max(speed)),
        'mouse_speed_skew':  float(skew(speed)),
        'mouse_speed_kurt':  float(kurtosis(speed)),
        'mouse_speed_ent':   _safe_entropy(speed),
        'mouse_fft1':        fft_s[0],
        'mouse_fft2':        fft_s[1],
        'mouse_fft3':        fft_s[2],
        'mouse_accel_mean':  float(np.mean(np.abs(accel))),
        'mouse_accel_std':   float(np.std(accel)),
        'mouse_dist_total':  total_dist,
        'mouse_dir_changes': float(dir_changes),
        'mouse_path_eff':    straight / (total_dist + 1e-5),
    }


def get_wheel_features(df, user_id):
    """
    10-dimensional enriched scroll feature vector.
    delta mean/std/skew/kurt/entropy, iti_mean/iti_std,
    freq, burst_ratio, n_events.
    """
    if len(df) < 3:
        return None

    ts         = df['time'].values.astype(np.int64) / 1e6
    dt         = np.diff(ts) + 1e-5
    delta      = df['delta'].values.astype(float)
    total_time = ts[-1] - ts[0]
    bursts     = int(np.sum(dt < 200))
    abs_delta  = np.abs(delta)

    return {
        'user':               user_id,
        'scroll_delta_mean':  float(np.mean(abs_delta)),
        'scroll_delta_std':   float(np.std(delta)),
        'scroll_delta_skew':  float(skew(delta))     if len(delta) > 2 else 0.0,
        'scroll_delta_kurt':  float(kurtosis(delta)) if len(delta) > 2 else 0.0,
        'scroll_delta_ent':   _safe_entropy(abs_delta),
        'scroll_iti_mean':    float(np.mean(dt)),
        'scroll_iti_std':     float(np.std(dt)),
        'scroll_freq':        len(delta) / (total_time / 1000 + 1e-5),
        'scroll_burst_ratio': bursts / (len(dt) + 1e-5),
        'scroll_n_events':    float(len(delta)),
    }


# ==============================================================================
# 3.  DATA LOADING
# ==============================================================================

def load_data_chunked():
    key_rows, mouse_rows, wheel_rows = [], [], []
    user_dirs = sorted([d for d in os.listdir(BBMAS_ROOT) if d.isdigit()], key=int)

    for uid_str in user_dirs:
        uid  = int(uid_str)
        udir = os.path.join(BBMAS_ROOT, uid_str)

        # Keystroke
        kfile = os.path.join(udir, f"{uid}_Desktop_Keyboard.csv")
        if os.path.exists(kfile):
            try:
                df = pd.read_csv(kfile)
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time').reset_index(drop=True)
                for i in range(0, len(df), CHUNK_KEY):
                    chunk   = df.iloc[i:i + CHUNK_KEY]
                    press   = chunk[chunk['direction'] == 0].copy()
                    release = chunk[chunk['direction'] == 1].copy()
                    feat    = get_keystroke_features(press, release, uid)
                    if feat:
                        key_rows.append(feat)
            except Exception:
                pass

        # Mouse Move  (CHUNK_MOUSE=50 — critical for 80-user coverage)
        mfile = os.path.join(udir, f"{uid}_Mouse_Move.csv")
        if os.path.exists(mfile):
            try:
                df = pd.read_csv(mfile)
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time').reset_index(drop=True)
                for i in range(0, len(df), CHUNK_MOUSE):
                    feat = get_mouse_features(df.iloc[i:i + CHUNK_MOUSE], uid)
                    if feat:
                        mouse_rows.append(feat)
            except Exception:
                pass

        # Mouse Wheel
        wfile = os.path.join(udir, f"{uid}_Mouse_Wheel.csv")
        if os.path.exists(wfile):
            try:
                df = pd.read_csv(wfile)
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time').reset_index(drop=True)
                for i in range(0, len(df), CHUNK_WHEEL):
                    feat = get_wheel_features(df.iloc[i:i + CHUNK_WHEEL], uid)
                    if feat:
                        wheel_rows.append(feat)
            except Exception:
                pass

    return pd.DataFrame(key_rows), pd.DataFrame(mouse_rows), pd.DataFrame(wheel_rows)


# ==============================================================================
# 4.  IMBALANCE CORRECTION — SMOTE + class weights
# ==============================================================================

def apply_smote(X, y, random_state=42):
    """
    SMOTE with automatic k_neighbors fallback.
    Falls back to simple repetition when genuine set is tiny.
    Applied ONLY to training split — never touches test data.
    """
    n_genuine  = int(np.sum(y == 1))
    n_impostor = int(np.sum(y == 0))

    if n_genuine < 2:
        return X, y

    k = min(5, n_genuine - 1)
    try:
        sm = SMOTE(k_neighbors=k, random_state=random_state)
        return sm.fit_resample(X, y)
    except Exception:
        repeat  = max(1, n_impostor // n_genuine)
        gen_idx = np.where(y == 1)[0]
        X2 = np.vstack([X, np.tile(X[gen_idx], (repeat, 1))])
        y2 = np.concatenate([y, np.ones(len(gen_idx) * repeat)])
        return X2, y2


# ==============================================================================
# 5.  MODEL — MLP 128->64->32->1  with BatchNorm + Dropout
# ==============================================================================

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
    mdl.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return mdl


# ==============================================================================
# 6.  PER-USER LOUO EVALUATION
# ==============================================================================

def train_eval_modality_louo(feat_df, target_user, feature_cols):
    """
    Train a 1-vs-rest MLP for target_user with LOUO impostor split.

    Split logic:
      Train : 70% of genuine chunks  +  all impostor users except one held-out
      Test  : 30% of genuine chunks  +  held-out impostor user's chunks (unseen)

    Returns (EER, AUC, test_scores, test_labels) or (None, None, None, None).
    """
    y_all     = (feat_df['user'] == target_user).astype(int).values
    X_all     = feat_df[feature_cols].values
    users_all = feat_df['user'].values

    genuine_idx  = np.where(y_all == 1)[0]
    impostor_idx = np.where(y_all == 0)[0]

    if len(genuine_idx) < MIN_GENUINE_CHUNKS:
        return None, None, None, None

    # 70/30 genuine split
    rng = np.random.RandomState(42)
    rng.shuffle(genuine_idx)
    split     = int(0.7 * len(genuine_idx))
    gen_train = genuine_idx[:split]
    gen_test  = genuine_idx[split:]

    # Hold out one entire impostor user for the test set
    impostor_users = sorted([u for u in feat_df['user'].unique() if u != target_user])
    held_out       = rng.choice(impostor_users)
    imp_train_idx  = impostor_idx[users_all[impostor_idx] != held_out]
    imp_test_idx   = impostor_idx[users_all[impostor_idx] == held_out]

    if len(gen_test) == 0 or len(imp_test_idx) == 0:
        return None, None, None, None

    train_idx = np.concatenate([gen_train, imp_train_idx])
    test_idx  = np.concatenate([gen_test,  imp_test_idx])

    Xtr, ytr = X_all[train_idx], y_all[train_idx]
    Xte, yte = X_all[test_idx],  y_all[test_idx]

    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None, None, None, None

    # Scale (fit on train only — no leakage)
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    # SMOTE on training set only
    Xtr, ytr = apply_smote(Xtr, ytr)

    # Post-SMOTE class weight calibration
    cw = dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
    ))

    # Train with early stopping
    mdl = build_mlp(Xtr.shape[1])
    es  = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )
    mdl.fit(
        Xtr, ytr,
        epochs=40,
        batch_size=64,
        class_weight=cw,
        validation_split=0.15,
        callbacks=[es],
        verbose=0,
    )

    scores      = mdl.predict(Xte, verbose=0).ravel()
    fpr, tpr, _ = roc_curve(yte, scores, pos_label=1)
    fnr         = 1 - tpr
    eer_idx     = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[eer_idx]), float(auc(fpr, tpr)), scores, yte


# ==============================================================================
# 7.  FUSION HELPERS
# ==============================================================================

def pad_scores(src, target_len):
    """
    Tile src scores to match target_len.
    Aligns mouse/wheel arrays to keystroke length WITHOUT truncating labels.
    """
    if len(src) >= target_len:
        return src[:target_len]
    reps = int(np.ceil(target_len / max(len(src), 1)))
    return np.tile(src, reps)[:target_len]


def mlp_fusion(k_sc, m_sc, w_sc, labels, n_inputs=3):
    """
    MLP meta-learner on stacked modality scores.
    n_inputs=3 uses K+M+W scores.
    n_inputs=2 uses K+M only  (pass w_sc=None).
    """
    m_sc_a = pad_scores(m_sc, len(k_sc))

    if n_inputs == 3:
        w_sc_a = pad_scores(w_sc, len(k_sc))
        X_fus  = np.column_stack([k_sc, m_sc_a, w_sc_a])
    else:
        X_fus  = np.column_stack([k_sc, m_sc_a])

    if len(np.unique(labels)) < 2:
        return np.zeros_like(labels, dtype=float), labels

    Xtr, Xte, ytr, yte = train_test_split(
        X_fus, labels, test_size=0.4, stratify=labels, random_state=42
    )
    Xtr, ytr = apply_smote(Xtr, ytr)

    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return np.zeros_like(yte, dtype=float), yte

    meta = tf.keras.Sequential([
        tf.keras.Input(shape=(n_inputs,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1,  activation='sigmoid'),
    ])
    meta.compile(optimizer='adam', loss='binary_crossentropy')
    cw = dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
    ))
    meta.fit(Xtr, ytr, epochs=30, batch_size=16, class_weight=cw, verbose=0)
    return meta.predict(Xte, verbose=0).ravel(), yte


def eer_from_scores(scores, labels):
    """Compute EER and AUC from a score array and binary labels."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr         = 1 - tpr
    idx         = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx]), float(auc(fpr, tpr))


# ==============================================================================
# 8.  LOAD DATA
# ==============================================================================

print("=" * 65)
print("  BB-MAS Continuous Authentication — Final Consolidated Script")
print("=" * 65)
print(f"\nLoading BB-MAS from: {BBMAS_ROOT}")
print("Extracting chunks (first run takes a few minutes)...\n")

key_df, mouse_df, wheel_df = load_data_chunked()

print(f"Chunks extracted:")
print(f"  Keystroke : {len(key_df):,}  ({key_df['user'].nunique()} users)")
print(f"  Mouse     : {len(mouse_df):,}  ({mouse_df['user'].nunique()} users)")
print(f"  Wheel     : {len(wheel_df):,}  ({wheel_df['user'].nunique()} users)")

kcols = [c for c in key_df.columns   if c != 'user']
mcols = [c for c in mouse_df.columns if c != 'user']
wcols = [c for c in wheel_df.columns if c != 'user']

print(f"\nFeature dims: Keystroke={len(kcols)}  Mouse={len(mcols)}  Wheel={len(wcols)}")


def eligible_users(df, min_chunks=MIN_GENUINE_CHUNKS):
    return set(df.groupby('user').size()[lambda s: s >= min_chunks].index.tolist())


valid_users = (
    eligible_users(key_df)
    & eligible_users(mouse_df)
    & eligible_users(wheel_df)
)
print(f"\nEligible users (>={MIN_GENUINE_CHUNKS} chunks in ALL 3 modalities): {len(valid_users)}")


# ==============================================================================
# 9.  MAIN EVALUATION LOOP
# ==============================================================================

results_list = []

for u_idx, u in enumerate(sorted(valid_users), 1):
    try:
        print(f"\n[{u_idx:2d}/{len(valid_users)}] User {u}", flush=True)

        print("  -> Keystroke ...", flush=True)
        k_eer, k_auc, k_sc, k_lbl = train_eval_modality_louo(key_df,   u, kcols)
        print("  -> Mouse     ...", flush=True)
        m_eer, m_auc, m_sc, _     = train_eval_modality_louo(mouse_df, u, mcols)
        print("  -> Wheel     ...", flush=True)
        w_eer, w_auc, w_sc, _     = train_eval_modality_louo(wheel_df, u, wcols)

        if any(x is None for x in [k_eer, m_eer, w_eer]):
            print(f"  Skipped (None EER: K={k_eer} M={m_eer} W={w_eer})")
            continue

        print(f"  Modalities  K:{k_eer*100:5.2f}%  M:{m_eer*100:5.2f}%  W:{w_eer*100:5.2f}%",
              flush=True)

        lbl = k_lbl
        if len(np.unique(lbl)) < 2:
            print("  Skipped (single class in keystroke labels)")
            continue

        # Simple fusion: arithmetic mean of 3 per-modality EERs
        eer_sf3 = (k_eer + m_eer + w_eer) / 3.0
        auc_sf3 = (k_auc + m_auc + w_auc) / 3.0

        # Simple K+M fusion (no wheel)
        eer_sf2 = (k_eer + m_eer) / 2.0
        auc_sf2 = (k_auc + m_auc) / 2.0

        # MLP fusion K+M+W
        lf3_sc, lf3_lbl = mlp_fusion(k_sc, m_sc, w_sc, lbl, n_inputs=3)
        if len(np.unique(lf3_lbl)) < 2:
            print("  Skipped (single class after K+M+W fusion)")
            continue
        eer_lf3, auc_lf3 = eer_from_scores(lf3_sc, lf3_lbl)

        # MLP fusion K+M only
        lf2_sc, lf2_lbl = mlp_fusion(k_sc, m_sc, None, lbl, n_inputs=2)
        if len(np.unique(lf2_lbl)) < 2:
            print("  Skipped (single class after K+M fusion)")
            continue
        eer_lf2, auc_lf2 = eer_from_scores(lf2_sc, lf2_lbl)

        results_list.append({
            'user':                   u,
            'Keystroke EER':          k_eer   * 100,
            'Mouse EER':              m_eer   * 100,
            'Wheel EER':              w_eer   * 100,
            'Simple Fusion EER':      eer_sf3 * 100,
            'KM Simple Fusion EER':   eer_sf2 * 100,
            'KMW Learned Fusion EER': eer_lf3 * 100,
            'KM Learned Fusion EER':  eer_lf2 * 100,
            'Keystroke AUC':          k_auc,
            'Mouse AUC':              m_auc,
            'Wheel AUC':              w_auc,
            'Simple Fusion AUC':      auc_sf3,
            'KM Simple Fusion AUC':   auc_sf2,
            'KMW Learned Fusion AUC': auc_lf3,
            'KM Learned Fusion AUC':  auc_lf2,
        })

        print(
            f"  Simple:{eer_sf3*100:5.2f}%  "
            f"KMW-MLP:{eer_lf3*100:5.2f}%  "
            f"KM-MLP:{eer_lf2*100:5.2f}%"
        )

    except Exception as e:
        print(f"  User {u} skipped — {e}")


# ==============================================================================
# 10.  RESULTS TABLE + LaTeX output
# ==============================================================================

df_res  = pd.DataFrame(results_list)
n_users = len(df_res)

print(f"\n{'='*65}")
print(f"  FINAL RESULTS  ({n_users} users | CHUNK_MOUSE=50 | LOUO | SMOTE)")
print(f"{'='*65}")

if n_users == 0:
    print("ERROR: No results collected. Check skip messages above.")
else:
    BASELINE_EER = 37.54   # original 25-user baseline from paper

    rows = [
        ('Keystroke only (20-dim, enriched)',  'Keystroke EER',          'Keystroke AUC'),
        ('Mouse only (14-dim, chunk=50)',       'Mouse EER',              'Mouse AUC'),
        ('Wheel only (10-dim) [degenerate]',   'Wheel EER',              'Wheel AUC'),
        ('Simple Fusion K+M+W (avg)',           'Simple Fusion EER',      'Simple Fusion AUC'),
        ('Simple Fusion K+M (avg)',             'KM Simple Fusion EER',   'KM Simple Fusion AUC'),
        ('MLP Learned Fusion K+M+W',            'KMW Learned Fusion EER', 'KMW Learned Fusion AUC'),
        ('MLP Learned Fusion K+M',              'KM Learned Fusion EER',  'KM Learned Fusion AUC'),
    ]

    print(f"\n  {'Method':<42}  {'EER':>7}  {'Std':>7}  {'AUC':>6}  {'vs 37.54%':>10}")
    print("  " + "-" * 78)
    for name, eer_col, auc_col in rows:
        mean_eer = df_res[eer_col].mean()
        std_eer  = df_res[eer_col].std()
        mean_auc = df_res[auc_col].mean()
        rel      = (BASELINE_EER - mean_eer) / BASELINE_EER * 100
        print(f"  {name:<42}  {mean_eer:6.2f}%  {std_eer:6.2f}%  {mean_auc:.3f}  {rel:+.1f}%")

    print("\n  * Wheel AUC ~0.50 = degenerate (random). Excluded from paper tables.")

    # Key diagnostic stats
    lf3_beats_k = (df_res['KMW Learned Fusion EER'] < df_res['Keystroke EER']).sum()
    lf2_beats_k = (df_res['KM Learned Fusion EER']  < df_res['Keystroke EER']).sum()

    print(f"\n{'='*65}")
    print("  KEY PAPER STATISTICS")
    print(f"{'='*65}")
    best_eer = df_res['Keystroke EER'].mean()
    rel_best = (BASELINE_EER - best_eer) / BASELINE_EER * 100
    print(f"  Best result          : Keystroke  {best_eer:.2f}% EER  AUC={df_res['Keystroke AUC'].mean():.3f}")
    print(f"  vs baseline (37.54%) : {rel_best:+.1f}% relative improvement")
    print(f"  KMW fusion > K alone : {lf3_beats_k}/{n_users} users ({lf3_beats_k/n_users*100:.1f}%)")
    print(f"  KM  fusion > K alone : {lf2_beats_k}/{n_users} users ({lf2_beats_k/n_users*100:.1f}%)")
    print(f"  Wheel AUC            : {df_res['Wheel AUC'].mean():.3f}  (0.500 = random)")
    print(f"  Users evaluated      : {n_users}")

    # LaTeX table ready to paste into paper
    print(f"\n{'='*65}")
    print("  LATEX TABLE — paste as Table III in paper")
    print(f"{'='*65}")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Multi-User BB-MAS Evaluation (" + str(n_users) + r" Users, LOUO, SMOTE)}")
    print(r"\label{tab:multiuser}")
    print(r"\renewcommand{\arraystretch}{1.15}")
    print(r"\begin{tabular}{|l|c|c|c|}")
    print(r"\hline")
    print(r"\textbf{Method} & \textbf{Mean EER (\%)} & \textbf{Std (\%)} & \textbf{AUC} \\")
    print(r"\hline")
    for name, eer_col, auc_col in rows:
        mean_eer = df_res[eer_col].mean()
        std_eer  = df_res[eer_col].std()
        mean_auc = df_res[auc_col].mean()
        dagger   = r"$^\dagger$" if "degenerate" in name else ""
        if "Keystroke only" in name:
            print(r"\textbf{" + name.replace("[degenerate]","").strip() + dagger + r"} & "
                  r"\textbf{" + f"{mean_eer:.2f}" + r"} & "
                  r"\textbf{" + f"{std_eer:.2f}" + r"} & "
                  r"\textbf{" + f"{mean_auc:.3f}" + r"} \\")
        else:
            print(f"{name.replace('[degenerate]','').strip()}{dagger} & "
                  f"{mean_eer:.2f} & {std_eer:.2f} & {mean_auc:.3f} \\\\")
    print(r"\hline")
    print(r"\multicolumn{4}{l}{\small $^\dagger$ AUC$\approx$0.50: degenerate prediction (random classifier).} \\")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Save CSVs
    df_res.to_csv("paper_per_user_results_final.csv", index=False)
    print(f"\nSaved: paper_per_user_results_final.csv  ({n_users} rows x {len(df_res.columns)} cols)")
