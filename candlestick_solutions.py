"""
Candlestick Pattern Recognition and Price Movement Prediction for Nifty 50 Index
Group 16 - CCA1 Pattern Recognition

Task: predict next-day direction (1 = Close[t+1] > Close[t], else 0) from the last WINDOW candles.

Solution 1: Rule-based candlestick pattern detection + Feature Selection + k-Nearest Neighbours
Solution 2: HOG features on candlestick chart images + Feature Selection + SVM
Solution 3: Convolutional Neural Network on candlestick chart images

Usage:
    python candlestick_solutions.py --data nifty50.csv
Outputs go to ./results/ (metrics.csv, confusion matrices, sample chart images, training curve).
"""
import argparse, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)
from skimage.feature import hog
import tensorflow as tf
from keras import layers, models, callbacks

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
WINDOW = 10          # candles per chart image / feature window
IMG = 64             # image size (IMG x IMG)
OUT = "results"
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df[["Date", "Open", "High", "Low", "Close"]].dropna()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)   # next-day direction
    return df.iloc[:-1].reset_index(drop=True)                          # last row has no label


# ----------------------------------------------------------------------------------------
# Solution 1: rule-based pattern detection
# ----------------------------------------------------------------------------------------
def detect_patterns(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng_ = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    bull, bear = c > o, c < o
    avg_body = body.rolling(10).mean()
    downtrend = c.shift(1) < c.shift(5)
    uptrend = c.shift(1) > c.shift(5)

    p = pd.DataFrame(index=df.index)
    p["doji"] = (body <= 0.1 * rng_)
    p["hammer"] = (lower >= 2 * body) & (upper <= 0.3 * body.clip(lower=1e-9)) & (body > 0) & downtrend
    p["inv_hammer"] = (upper >= 2 * body) & (lower <= 0.3 * body.clip(lower=1e-9)) & (body > 0) & downtrend
    p["shooting_star"] = (upper >= 2 * body) & (lower <= 0.3 * body.clip(lower=1e-9)) & (body > 0) & uptrend
    p["hanging_man"] = (lower >= 2 * body) & (upper <= 0.3 * body.clip(lower=1e-9)) & (body > 0) & uptrend
    p["marubozu_bull"] = bull & (body >= 0.9 * rng_)
    p["marubozu_bear"] = bear & (body >= 0.9 * rng_)
    p["spinning_top"] = (body <= 0.3 * rng_) & (upper >= body) & (lower >= body) & ~p["doji"]
    p["bull_engulf"] = bear.shift(1) & bull & (c >= o.shift(1)) & (o <= c.shift(1))
    p["bear_engulf"] = bull.shift(1) & bear & (o >= c.shift(1)) & (c <= o.shift(1))
    p["bull_harami"] = bear.shift(1) & bull & (o > c.shift(1)) & (c < o.shift(1))
    p["bear_harami"] = bull.shift(1) & bear & (o < c.shift(1)) & (c > o.shift(1))
    p["piercing"] = bear.shift(1) & bull & (o < l.shift(1)) & (c > (o.shift(1) + c.shift(1)) / 2) & (c < o.shift(1))
    p["dark_cloud"] = bull.shift(1) & bear & (o > h.shift(1)) & (c < (o.shift(1) + c.shift(1)) / 2) & (c > o.shift(1))
    mid1 = (o.shift(2) + c.shift(2)) / 2
    small1 = body.shift(1) < 0.5 * avg_body.shift(1)
    p["morning_star"] = bear.shift(2) & small1 & bull & (c > mid1) & (body.shift(2) > avg_body.shift(2))
    p["evening_star"] = bull.shift(2) & small1 & bear & (c < mid1) & (body.shift(2) > avg_body.shift(2))
    p["three_white"] = bull & bull.shift(1) & bull.shift(2) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    p["three_black"] = bear & bear.shift(1) & bear.shift(2) & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    return p.fillna(False).astype(int)


def solution1_features(df, pat):
    """Per-candle: pattern flags + candle geometry; flattened over the last WINDOW candles."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    rng_ = (h - l).replace(0, np.nan)
    geo = pd.DataFrame({
        "body_ratio": (c - o) / rng_,
        "upper_ratio": (h - np.maximum(o, c)) / rng_,
        "lower_ratio": (np.minimum(o, c) - l) / rng_,
        "ret": c.pct_change(),
    }).fillna(0)
    per_candle = pd.concat([pat, geo], axis=1).values
    idx = np.arange(WINDOW + 10, len(df))              # skip warm-up rows for rolling indicators
    X = np.stack([per_candle[i - WINDOW + 1: i + 1].ravel() for i in idx])
    return X, idx


# ----------------------------------------------------------------------------------------
# Candlestick image rendering (Solutions 2 and 3)
# ----------------------------------------------------------------------------------------
def render_window(w):
    """Render WINDOW candles into a grayscale IMG x IMG array. Bull = filled white, bear = outline-only."""
    img = np.zeros((IMG, IMG), dtype=np.uint8)
    lo, hi = w[:, 2].min(), w[:, 1].max()
    scale = (IMG - 1) / (hi - lo if hi > lo else 1.0)
    y = lambda v: int(round((IMG - 1) - (v - lo) * scale))       # price -> row
    slot = IMG // WINDOW
    for k, (o, h, l, c) in enumerate(w):
        cx = k * slot + slot // 2
        img[y(h): y(l) + 1, cx] = 200                            # wick
        top, bot = sorted((y(o), y(c)))
        x0, x1 = cx - 2, cx + 2
        if c >= o:
            img[top: bot + 1, x0: x1 + 1] = 255                  # bullish: filled
        else:
            img[top: bot + 1, x0: x1 + 1] = 120                  # bearish: darker fill
    return img


def build_images(df, idx):
    arr = df[["Open", "High", "Low", "Close"]].values
    return np.stack([render_window(arr[i - WINDOW + 1: i + 1]) for i in idx])


def save_samples(images, y):
    fig, ax = plt.subplots(2, 6, figsize=(12, 4.4))
    for r, cls in enumerate([1, 0]):
        sel = np.where(y == cls)[0][:6]
        for a, j in zip(ax[r], sel):
            a.imshow(images[j], cmap="gray"); a.axis("off")
        ax[r][0].set_title("Next day UP" if cls else "Next day DOWN", loc="left", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/sample_chart_images.png", dpi=150); plt.close()


# ----------------------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------------------
results = []

def evaluate(name, y_true, y_pred, y_score=None):
    row = {"Model": name,
           "Accuracy": accuracy_score(y_true, y_pred),
           "Precision": precision_score(y_true, y_pred, zero_division=0),
           "Recall": recall_score(y_true, y_pred, zero_division=0),
           "F1": f1_score(y_true, y_pred, zero_division=0),
           "AUC": roc_auc_score(y_true, y_score) if y_score is not None else np.nan}
    results.append(row)
    print(f"\n[{name}] " + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "Model"))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["Down", "Up"]).plot(cmap="Blues")
    plt.title(name); plt.tight_layout()
    plt.savefig(f"{OUT}/cm_{name.replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')}.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main(args):
    df = load_data(args.data)
    print(f"Loaded {len(df)} rows: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print("Class balance (Up=1):", df["Target"].mean().round(3))

    pat = detect_patterns(df)
    pat.sum().sort_values(ascending=False).rename("count").to_csv(f"{OUT}/pattern_counts.csv")
    print("\nPattern frequency:\n", pat.sum().sort_values(ascending=False).to_string())

    X1, idx = solution1_features(df, pat)
    y = df["Target"].values[idx]
    images = build_images(df, idx)
    save_samples(images, y)

    # Chronological split (no shuffling -> avoids look-ahead leakage): 70 / 15 / 15
    n = len(idx); i_tr, i_va = int(0.70 * n), int(0.85 * n)
    tr, va, te = slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)
    print(f"\nTrain {i_tr} | Val {i_va - i_tr} | Test {n - i_va}")

    # Majority-class baseline
    base = np.full(len(y[te]), int(round(y[tr].mean())))
    evaluate("Baseline (majority class)", y[te], base)

    # ---------------- Solution 1: Rule-based patterns + Feature Selection + kNN ----------------
    sc = StandardScaler().fit(X1[tr])
    
    # Compulsory Filter Feature Selection (Fisher Score / ANOVA via SelectKBest)
    selector_s1 = SelectKBest(score_func=f_classif, k=15).fit(sc.transform(X1[tr]), y[tr])
    X1_tr_sel = selector_s1.transform(sc.transform(X1[tr]))
    X1_te_sel = selector_s1.transform(sc.transform(X1[te]))

    gs = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [5, 11, 21, 31, 51], "weights": ["uniform", "distance"]},
                      cv=TimeSeriesSplit(5), scoring="accuracy", n_jobs=-1)
    gs.fit(X1_tr_sel, y[tr])
    print("kNN best params:", gs.best_params_)
    knn = gs.best_estimator_
    evaluate("Sol1 Rule-based + FS + kNN", y[te], knn.predict(X1_te_sel),
             knn.predict_proba(X1_te_sel)[:, 1])

    # ---------------- Solution 2: HOG + Feature Selection + SVM ----------------
    H = np.stack([hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
                  for im in images])
    print("HOG feature length before selection:", H.shape[1])
    
    sc2 = StandardScaler().fit(H[tr])
    
    # Compulsory Feature Selection for high-dimensional HOG features
    selector_s2 = SelectKBest(score_func=f_classif, k=60).fit(sc2.transform(H[tr]), y[tr])
    H_tr_sel = selector_s2.transform(sc2.transform(H[tr]))
    H_te_sel = selector_s2.transform(sc2.transform(H[te]))
    print("HOG feature length after selection:", H_tr_sel.shape[1])

    gs2 = GridSearchCV(SVC(kernel="rbf", probability=True, random_state=SEED),
                       {"C": [0.1, 1, 10], "gamma": ["scale", 0.001]},
                       cv=TimeSeriesSplit(3), scoring="accuracy", n_jobs=-1)
    gs2.fit(H_tr_sel, y[tr])
    print("SVM best params:", gs2.best_params_)
    svm = gs2.best_estimator_
    evaluate("Sol2 HOG + FS + SVM", y[te], svm.predict(H_te_sel),
             svm.predict_proba(H_te_sel)[:, 1])

    # ---------------- Solution 3: CNN ----------------
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    tf.random.set_seed(SEED)
    Xi = (images.astype("float32") / 255.0)[..., None]
    cnn = models.Sequential([
        layers.Input((IMG, IMG, 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dropout(0.5),
        layers.Dense(64, activation="relu"), layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")])
    cnn.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    hist = cnn.fit(Xi[tr], y[tr], validation_data=(Xi[va], y[va]), epochs=args.epochs, batch_size=32, verbose=2,
                   callbacks=[callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")])
    prob = cnn.predict(Xi[te], verbose=0).ravel()
    evaluate("Sol3 CNN", y[te], (prob >= 0.5).astype(int), prob)
    cnn.save(f"{OUT}/cnn_model.keras")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/cnn_training_curves.png", dpi=150); plt.close()

    res = pd.DataFrame(results).round(4)
    res.to_csv(f"{OUT}/metrics.csv", index=False)
    print("\n================ FINAL TEST RESULTS ================\n", res.to_string(index=False))

    ax = res.set_index("Model")[["Accuracy", "F1"]].plot(kind="bar", figsize=(8, 4), rot=15)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8); ax.set_ylim(0, 1); plt.tight_layout()
    plt.savefig(f"{OUT}/model_comparison.png", dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="nifty50.csv")
    ap.add_argument("--epochs", type=int, default=40)
    main(ap.parse_args())