"""Short-term (15-30 min) crowd forecasting and accuracy evaluation.

Model ("seasonal profile + decaying residual")
----------------------------------------------
For every location the history is resampled to a 5-minute grid and split into
  s(t)  : typical crowd for that time of day (separate weekday / weekend profile)
  r(t)  : residual = y(t) - s(t)  (today's deviation from the typical pattern)
Forecast for t0 + k steps:   s(t0+k) + mean(r over last 15 min) * phi^k
where phi is the lag-1 autocorrelation of the residual, so today's deviation
fades back towards the usual pattern. Persistence ("same as now") is the
baseline used to show whether the model adds value.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import LOCATIONS, DEFAULT_THRESHOLDS, level_index_array

STEP = 5  # minutes
BUCKETS = 24 * 60 // STEP


def build_series(obs_loc):
    """5-minute regular series (NaN where there was no reading)."""
    s = obs_loc.set_index("ts")["count"].resample(f"{STEP}min").mean()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq=f"{STEP}min"))


class Forecaster:
    def __init__(self):
        self.profile = np.zeros((2, BUCKETS))
        self.phi = 0.9
        self.sigma = np.ones(12)

    @staticmethod
    def _keys(idx):
        return (idx.dayofweek >= 5).astype(int), ((idx.hour * 60 + idx.minute) // STEP).astype(int)

    def seasonal(self, idx):
        dt, bk = self._keys(idx)
        return pd.Series(self.profile[dt, bk], index=idx)

    def fill(self, raw):
        y = raw.interpolate(limit=12)
        return y.fillna(self.seasonal(raw.index))

    def fit(self, raw):
        vals = raw.dropna()
        if len(raw) >= 2 * BUCKETS and len(vals):
            dt, bk = self._keys(vals.index)
            tab = (pd.DataFrame({"d": dt, "b": bk, "v": vals.values})
                   .groupby(["d", "b"])["v"].mean().unstack().reindex(range(2)).reindex(columns=range(BUCKETS)))
            prof = tab.values.copy()
            if np.isnan(prof[1]).all():
                prof[1] = prof[0] * 0.4  # no weekend in history
            if np.isnan(prof[0]).all():
                prof[0] = prof[1] / 0.4
            for d in range(2):  # fill gaps then smooth circularly (15 min)
                row = pd.Series(prof[d]).interpolate(limit_direction="both").values
                padded = np.concatenate([row[-1:], row, row[:1]])
                prof[d] = np.convolve(padded, np.ones(3) / 3, mode="valid")
            self.profile = np.nan_to_num(prof, nan=float(vals.mean()))
        else:  # too little history: flat profile = overall mean
            self.profile = np.full((2, BUCKETS), float(vals.mean()) if len(vals) else 0.0)
        y = self.fill(raw)
        r = (y - self.seasonal(raw.index)).values
        r = np.nan_to_num(r)
        if len(r) > 10 and r.std() > 0:
            self.phi = float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], 0.3, 0.98))
        rsm = pd.Series(r).rolling(3, min_periods=1).mean().values
        sig = []
        for k in range(1, 13):
            if len(r) > k + 5:
                sig.append(np.std(r[k:] - rsm[:-k] * self.phi**k))
            else:
                sig.append(max(np.std(r), 1.0))
        self.sigma = np.array(sig)
        return self

    def forecast(self, raw, steps=6):
        y = self.fill(raw)
        r = (y - self.seasonal(raw.index)).iloc[-3:].mean()
        origin = raw.index[-1]
        times = pd.date_range(origin + pd.Timedelta(minutes=STEP), periods=steps, freq=f"{STEP}min")
        k = np.arange(1, steps + 1)
        pred = np.clip(self.seasonal(times).values + r * self.phi**k, 0, None)
        sig = self.sigma[np.minimum(k, 12) - 1]
        return pd.DataFrame({"time": times, "pred": pred,
                             "lo": np.clip(pred - 1.28 * sig, 0, None), "hi": pred + 1.28 * sig})


def forecast_all(obs, steps=6):
    """dict location -> (raw series, forecast DataFrame)."""
    out = {}
    for loc in LOCATIONS:
        d = obs[obs.location == loc]
        if len(d) < 3:
            continue
        raw = build_series(d)
        out[loc] = (raw, Forecaster().fit(raw).forecast(raw, steps))
    return out


# ------------------------------ ML model (scikit-learn) --------------------
def _ml_frame(y, f, k):
    """Features for predicting k steps ahead from every origin in series y."""
    idx = y.index
    sec = idx.hour * 60 + idx.minute
    tgt = idx + pd.Timedelta(minutes=STEP * k)
    X = pd.DataFrame({
        "sin": np.sin(2 * np.pi * sec / 1440), "cos": np.cos(2 * np.pi * sec / 1440),
        "wk": (idx.dayofweek >= 5).astype(int),
        "y0": y.values, "y3": y.shift(3).values, "y6": y.shift(6).values, "y12": y.shift(12).values,
        "s0": f.seasonal(idx).values, "sk": f.seasonal(tgt).values}, index=idx)
    return X, y.shift(-k)


def _new_model():
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=5, random_state=0)


def forecast_ml(obs, horizons=(15, 30)):
    """Gradient-boosting forecast: dict location -> {horizon: people}."""
    out = {}
    for loc in LOCATIONS:
        d = obs[obs.location == loc]
        if len(d) < 3 * BUCKETS // 4:
            continue
        raw = build_series(d)
        f = Forecaster().fit(raw)
        y = f.fill(raw)
        res = {}
        for h in horizons:
            k = h // STEP
            X, yt = _ml_frame(y, f, k)
            m = yt.notna().values
            if m.sum() < 50:
                continue
            mdl = _new_model().fit(X[m], yt[m])
            res[h] = float(max(mdl.predict(X.iloc[[-1]])[0], 0))
        if res:
            out[loc] = res
    return out


# ------------------------------ evaluation --------------------------------
def evaluate(obs, thresholds=DEFAULT_THRESHOLDS, test_days=3, horizons=(15, 30)):
    """Walk-forward back-test: fit on data before the cut-off, predict every
    5-minute origin after it. Returns (metrics DataFrame, details DataFrame)."""
    rows = []
    for loc in LOCATIONS:
        d = obs[obs.location == loc]
        if len(d) < 50:
            continue
        raw = build_series(d)
        span_days = (raw.index[-1] - raw.index[0]).total_seconds() / 86400
        td = min(test_days, max(span_days * 0.3, 0.2))
        cutoff = raw.index[-1] - pd.Timedelta(days=td)
        f = Forecaster().fit(raw[:cutoff])
        y = f.fill(raw)
        s = f.seasonal(raw.index).values
        r = pd.Series(y.values - s).rolling(3, min_periods=1).mean().values
        for h in horizons:
            k = h // STEP
            if len(raw) <= k:
                continue
            origin_idx = raw.index[:-k]
            target_idx = raw.index[k:]
            actual = raw.values[k:]
            pred = np.clip(s[k:] + r[:-k] * f.phi**k, 0, None)
            base = y.values[:-k]
            m = (origin_idx >= cutoff) & ~np.isnan(actual) & ~np.isnan(base)
            X, yt = _ml_frame(y, f, k)
            X, yt = X.iloc[:-k], yt.iloc[:-k]
            tr = (X.index + pd.Timedelta(minutes=STEP * k) <= cutoff) & yt.notna().values
            if tr.sum() >= 50:
                ml_pred = np.clip(_new_model().fit(X[tr], yt[tr]).predict(X), 0, None)
            else:
                ml_pred = pred
            rows.append(pd.DataFrame({
                "time": target_idx[m], "Location": loc, "Horizon": h, "actual": actual[m],
                "pred": pred[m], "pred_ml": ml_pred[m], "baseline": base[m],
                "capacity": LOCATIONS[loc]["capacity"]}))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    details = pd.concat(rows, ignore_index=True)
    return metrics_from(details, thresholds), details


def _metrics(g, thresholds):
    err = g["pred"] - g["actual"]
    mae = err.abs().mean()
    base_mae = (g["baseline"] - g["actual"]).abs().mean()
    la = level_index_array(g["actual"], g["capacity"], thresholds)
    lp = level_index_array(g["pred"], g["capacity"], thresholds)
    err_ml = g["pred_ml"] - g["actual"]
    lm = level_index_array(g["pred_ml"], g["capacity"], thresholds)
    return pd.Series({
        "Samples": len(g),
        "MAE (people)": mae,
        "RMSE (people)": float(np.sqrt((err**2).mean())),
        "WAPE %": 100 * err.abs().sum() / max(g["actual"].sum(), 1e-9),
        "Baseline MAE": base_mae,
        "Improvement vs baseline %": 100 * (base_mae - mae) / max(base_mae, 1e-9),
        "Level accuracy %": 100 * (la == lp).mean(),
        "Within 1 level %": 100 * (np.abs(la - lp) <= 1).mean(),
        "ML MAE (people)": err_ml.abs().mean(),
        "ML RMSE (people)": float(np.sqrt((err_ml**2).mean())),
        "ML Improvement vs baseline %": 100 * (base_mae - err_ml.abs().mean()) / max(base_mae, 1e-9),
        "ML Level accuracy %": 100 * (la == lm).mean(),
    })


def metrics_from(details, thresholds=DEFAULT_THRESHOLDS):
    per = []
    for (loc, h), g in details.groupby(["Location", "Horizon"]):
        m = _metrics(g, thresholds)
        m["Location"], m["Horizon"] = loc, h
        per.append(m)
    per = pd.DataFrame(per)
    allm = []
    for h, g in details.groupby("Horizon"):
        m = _metrics(g, thresholds)
        m["Location"], m["Horizon"] = "ALL CAMPUS", h
        allm.append(m)
    allm = pd.DataFrame(allm)
    out = pd.concat([allm, per], ignore_index=True)
    cols = ["Location", "Horizon"] + [c for c in out.columns if c not in ("Location", "Horizon")]
    return out[cols].round(2)
