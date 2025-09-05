import os
import json
import random
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import date, datetime, time
  # ✅ Needed for calendar_features

from flask import Blueprint, render_template, request, jsonify, abort
from flask_login import login_required, current_user
from keras.models import load_model
from keras.saving import register_keras_serializable
from sqlalchemy import func
from functools import wraps
import traceback
import calendar

from .auth import Prediction
from . import db

# --- Define paths ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")

# --- Initialize country priors (will be updated by CSV if available) ---
COUNTRY_PRIORS = {}

# --- Load forecast CSV if present ---
def _try_load_forecast_csv(models_dir):
    ...
    # (keep CSV loader function here)
    ...

_try_load_forecast_csv(MODELS_DIR)

     


# =============================================================================
# Deterministic seeds (prevent random drift between runs)
# =============================================================================
os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# Blueprint
# =============================================================================
main = Blueprint('main', __name__)

# =============================================================================
# Load Models (compile=False to avoid custom metrics issues)
# =============================================================================
classifier_model = None
regressor_model = None

# ---------- CSV loader: forecast_summary_2026 ----------

import pandas as pd


import glob

CSV_DEBUG_INFO = {"path": None, "exists": False, "columns": [], "rows_preview": [], "all_csvs": []}

# --- CSV Loader (place this near the top, after MODELS_DIR and COUNTRY_PRIORS) ---
import pandas as pd

def load_csv_forecast_2026(models_dir: str) -> int:
    """Load 'forecast_summary_2026 (1).csv' (or ..._2026.csv) into COUNTRY_PRIORS."""
    global COUNTRY_PRIORS
    candidates = [
        os.path.join(models_dir, "forecast_summary_2026 (1).csv"),
        os.path.join(models_dir, "forecast_summary_2026.csv"),
    ]

    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        print("ℹ️ CSV loader: file not found in", models_dir)
        return 0

    df = pd.read_csv(path, encoding="utf-8-sig")

    required = [
        "Country", "Total Annual Demand", "Avg Daily Demand", "Max Daily Demand",
        "Days with Demand", "Demand Frequency", "Monthly Average", "Quarterly Average"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("⚠️ CSV loader: missing columns:", missing)
        return 0

    def to_float(x, default=None):
        try:
            if isinstance(x, str):
                x = x.replace(",", "").replace("%", "").strip()
            return float(x)
        except Exception:
            return default

    loaded = 0
    for _, row in df.iterrows():
        country = str(row["Country"]).strip()
        if not country:
            continue
        pri = COUNTRY_PRIORS.get(country, {}).copy()
        # parse/assign everything
        annual        = to_float(row["Total Annual Demand"], pri.get("annual"))
        avg_daily     = to_float(row["Avg Daily Demand"],  pri.get("avg_daily"))
        max_daily     = to_float(row["Max Daily Demand"],  pri.get("max_daily"))
        days_with     = to_float(row["Days with Demand"],  pri.get("days_with_demand"))
        freq_percent  = to_float(row["Demand Frequency"],  pri.get("freq"))  # e.g. 50.1 (not fraction)
        monthly_avg   = to_float(row["Monthly Average"],   pri.get("monthly_avg"))
        quarterly_avg = to_float(row["Quarterly Average"], pri.get("quarterly_avg"))

        pri["annual"]           = annual
        pri["avg_daily"]        = avg_daily
        pri["max_daily"]        = max_daily
        pri["days_with_demand"] = days_with
        pri["freq"]             = freq_percent
        pri["monthly_avg"]      = monthly_avg
        pri["quarterly_avg"]    = quarterly_avg

        COUNTRY_PRIORS[country] = pri
        loaded += 1

    print(f"✅ CSV loader: loaded priors for {loaded} countries from '{os.path.basename(path)}'")
    return loaded


    #     if not c_country:
    #         print(f"⚠️ CSV loader: 'Country' column not found in {path}. Columns = {df.columns.tolist()}")
    #         continue

    #     loaded = 0
    #     for _, row in df.iterrows():
    #         country = str(row[c_country]).strip()
    #         if not country:
    #             continue
    #         pri = COUNTRY_PRIORS.get(country, {}).copy()

    #         def num(val, default=None):
    #             try:
    #                 if isinstance(val, str):
    #                     val = val.replace(",", "").strip()
    #                 return float(val)
    #             except Exception:
    #                 return default

    #         if c_annual: pri["annual"] = num(row.get(c_annual), pri.get("annual"))
    #         if c_avgday: pri["avg_daily"] = num(row.get(c_avgday), pri.get("avg_daily"))
    #         if c_maxday: pri["max_daily"] = num(row.get(c_maxday), pri.get("max_daily"))
    #         if c_days:   pri["days_with_demand"] = num(row.get(c_days), pri.get("days_with_demand"))
    #         if c_freq:   pri["freq"] = num(row.get(c_freq), pri.get("freq"))
    #         if c_month:  pri["monthly_avg"] = num(row.get(c_month), pri.get("monthly_avg"))
    #         if c_quart:  pri["quarterly_avg"] = num(row.get(c_quart), pri.get("quarterly_avg"))

    #         COUNTRY_PRIORS[country] = pri
    #         loaded += 1

    #     print(f"✅ Loaded forecast CSV '{os.path.basename(path)}' and updated priors for {loaded} countries.")
    #     return loaded > 0

    # print("ℹ️ CSV loader: matched files but none were usable.")
    # return False

# Call at import time
_try_load_forecast_csv(MODELS_DIR)




try:
    APP_DIR = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(APP_DIR, "models")

    classifier_model = load_model(
        os.path.join(MODELS_DIR, "optimized_classifier_model.keras"),
        compile=False
    )
    regressor_model = load_model(
        os.path.join(MODELS_DIR, "optimized_regressor_model.keras"),
        compile=False
    )
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"⚠️ WARNING: Could not load model files. Error: {e}")
    classifier_model, regressor_model = None, None
# Try loading Bayesian optimization results


def _try_load_bo_results(models_dir):
    import os, pickle

    candidates = [
        "bayesian_optimization_results (1).pkl",
        "bayesian_optimization_results.pkl",
        "bayesian_optimizer_model.pkl",
        "bayes_opt_results.pkl",
    ]

    def normalize(obj):
        """
        Return a display-friendly structure:
        - If it's an Optuna Study, return dict with best_params, best_value, trials list.
        - If it's a pandas DataFrame, convert to records.
        - If it's dict/list already, return as-is.
        """
        try:
            import optuna
            from optuna.study.study import Study
            if isinstance(obj, Study):
                st = obj
                try:
                    df = st.trials_dataframe()
                    trials = df.to_dict(orient="records")
                except Exception:
                    trials = [{"number": t.number, "value": t.value, "params": t.params} for t in st.trials]
                return {
                    "best_params": st.best_params,
                    "best_value": st.best_value,
                    "trials": trials,
                    "_source": "optuna.Study",
                }
        except Exception:
            pass

        try:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                return {"trials": obj.to_dict(orient="records"), "_source": "pandas.DataFrame"}
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
        except Exception:
            pass

        if isinstance(obj, (list, dict)):
            return obj
        # Fallback: wrap as generic
        return {"object_repr": repr(obj), "_source": type(obj).__name__}

    def try_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def try_joblib(path):
        import joblib
        return joblib.load(path)

    def try_pandas(path):
        import pandas as pd
        return pd.read_pickle(path)

    def try_compressed_pickle(path, mode):
        import gzip, bz2, lzma, pickle as pkl
        if mode == "gzip":
            with gzip.open(path, "rb") as f: return pkl.load(f)
        if mode == "bz2":
            with bz2.open(path, "rb") as f: return pkl.load(f)
        if mode == "lzma":
            with lzma.open(path, "rb") as f: return pkl.load(f)

    for name in candidates:
        p = os.path.join(models_dir, name)
        if not os.path.exists(p):
            continue
        # Try plain pickle
        try:
            obj = try_pickle(p)
            print(f"✅ Loaded (pickle): {p}")
            return normalize(obj)
        except Exception as e:
            print(f"… pickle failed for {p}: {e}")

        # Try joblib
        try:
            obj = try_joblib(p)
            print(f"✅ Loaded (joblib): {p}")
            return normalize(obj)
        except Exception as e:
            print(f"… joblib failed for {p}: {e}")

        # Try pandas.read_pickle
        try:
            obj = try_pandas(p)
            print(f"✅ Loaded (pandas.read_pickle): {p}")
            return normalize(obj)
        except Exception as e:
            print(f"… pandas.read_pickle failed for {p}: {e}")

        # Try compressed pickles with same filename
        for mode in ("gzip", "bz2", "lzma"):
            try:
                obj = try_compressed_pickle(p, mode)
                print(f"✅ Loaded ({mode}): {p}")
                return normalize(obj)
            except Exception as e:
                print(f"… {mode} failed for {p}: {e}")

    # Print dir listing to help debug
    try:
        listing = os.listdir(models_dir)
    except Exception as e:
        listing = [f"<could not list: {e}>"]
    print("❌ Unable to load BO results. Looked in:", models_dir)
    print("📂 Models dir contents:", listing)
    return None



# =============================================================================
# Country priors from your ML summary (2026)
# =============================================================================
# avg_daily: average daily demand (0.3 / 0.4)
# freq: demand frequency fraction (49.6% => 0.496)
# annual: Total Annual Demand (from your table)
COUNTRY_PRIORS = {
    "Bangladesh":   {"avg_daily": 0.4, "freq": 0.496, "annual": 128},
    "Benin":        {"avg_daily": 0.3, "freq": 0.449, "annual": 120},
    "Burkina Faso": {"avg_daily": 0.3, "freq": 0.499, "annual": 121},
    "Congo DRC":    {"avg_daily": 0.4, "freq": 0.488, "annual": 132},
    "Ghana":        {"avg_daily": 0.4, "freq": 0.485, "annual": 131},
    "Haiti":        {"avg_daily": 0.4, "freq": 0.532, "annual": 146},
    "Malawi":       {"avg_daily": 0.4, "freq": 0.474, "annual": 135},
    "Mali":         {"avg_daily": 0.4, "freq": 0.474, "annual": 131},
    "Rwanda":       {"avg_daily": 0.4, "freq": 0.518, "annual": 141},
    "Senegal":      {"avg_daily": 0.4, "freq": 0.510, "annual": 137},
    "Tanzania":     {"avg_daily": 0.3, "freq": 0.488, "annual": 125},
    "Togo":         {"avg_daily": 0.4, "freq": 0.510, "annual": 130},
    "Uganda":       {"avg_daily": 0.3, "freq": 0.482, "annual": 121},
    "Zambia":       {"avg_daily": 0.3, "freq": 0.493, "annual": 122},
}

# Global calibration: scale per-day baselines into the model’s magnitude.
# Tune this once you see the range of your regressor outputs.
CALIBRATION_SCALE = 1200.0  # try 900 ~ 1500 to calibrate

# =============================================================================
# Helpers
# =============================================================================
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not getattr(current_user, "is_admin", False):
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


def calendar_features(d: date):
    """7 calendar features in required order."""
    dow = d.weekday()                              # 0..6
    moy = d.month                                  # 1..12
    qtr = (d.month - 1) // 3 + 1                   # 1..4
    doy = d.timetuple().tm_yday                    # 1..365/366
    is_weekend = 1 if dow >= 5 else 0
    last_dom = calendar.monthrange(d.year, d.month)[1]
    is_month_end = 1 if d.day == last_dom else 0
    q_end_month = qtr * 3
    q_end_last_dom = calendar.monthrange(d.year, q_end_month)[1]
    is_quarter_end = 1 if (d.month == q_end_month and d.day == q_end_last_dom) else 0
    return [dow, moy, qtr, doy, is_weekend, is_month_end, is_quarter_end]


def order_inputs_by_model(model, static_input, sequence_input, time_features_input):
    """Return inputs in the exact order the model expects (by input tensor names)."""
    name_to_tensor = {
        "static_input": static_input,
        "sequence_input": sequence_input,
        "time_features_input": time_features_input,
    }
    ordered = []
    for t in model.inputs:
        in_name = t.name.split(":")[0]
        if in_name not in name_to_tensor:
            raise ValueError(f"Missing tensor for required input '{in_name}'")
        ordered.append(name_to_tensor[in_name])
    return ordered


def build_sequence_from_history(user_id, country, medicine, window=14, pred_date=None):
    """
    Build (window,) sequence from the user's *past* predictions for (country, medicine),
    strictly before pred_date (or before today). Pads with zeros on the left.
    """
    if pred_date is None:
        cutoff_dt = datetime.combine(datetime.today().date(), time.min)
    else:
        cutoff_dt = datetime.combine(pred_date, time.min)

    recent = (
        Prediction.query
        .filter(
            Prediction.user_id == user_id,
            Prediction.country == country,
            Prediction.medicine == medicine,
            Prediction.timestamp < cutoff_dt  # exclude same-day predictions (prevents drift)
        )
        .order_by(Prediction.timestamp.desc())
        .limit(window)
        .all()
    )

    if not recent:
        return np.zeros((window,), dtype=np.float32)

    recent = list(reversed(recent))  # chronological
    vals = []
    for p in recent:
        try:
            v = float(p.predicted_demand.split(' ')[0].replace(',', ''))
        except Exception:
            v = 0.0
        vals.append(v)

    if len(vals) < window:
        vals = [0.0] * (window - len(vals)) + vals

    return np.asarray(vals[:window], dtype=np.float32)


def build_sequence_from_global_history(country, medicine, window=14):
    """
    Build sequence from ALL users' predictions for (country, medicine).
    Returns None if no records exist.
    """
    recent = (
        Prediction.query
        .filter_by(country=country, medicine=medicine)
        .order_by(Prediction.timestamp.desc())
        .limit(window)
        .all()
    )
    if not recent:
        return None
    recent = list(reversed(recent))
    vals = []
    for p in recent:
        try:
            v = float(p.predicted_demand.split(' ')[0].replace(',', ''))
        except Exception:
            v = 0.0
        vals.append(v)
    if len(vals) < window:
        vals = [0.0] * (window - len(vals)) + vals
    return np.asarray(vals[:window], dtype=np.float32)


def build_country_baseline_sequence(country_name: str, month: int, window=14):
    """
    Deterministic, country-specific baseline if no history exists anywhere.
    Uses per-country Annual + Avg Daily + Demand Frequency from priors,
    then applies a mild seasonal wobble so it's stable but not flat.
    """
    avg_daily = 0.35
    freq = 0.50
    annual = 130.0

    if country_name in COUNTRY_PRIORS:
        avg_daily = COUNTRY_PRIORS[country_name]["avg_daily"]
        freq = COUNTRY_PRIORS[country_name]["freq"]
        annual = COUNTRY_PRIORS[country_name]["annual"]

    # Two equivalent daily baselines from your table; blend them to be robust:
    # A) avg_daily from table (0.3/0.4)
    base_a = avg_daily
    # B) annual / 365
    base_b = float(annual) / 365.0
    # Blend + frequency factor
    per_day = 0.5 * (base_a + base_b) * max(0.2, min(1.0, freq))

    # Scale into the model’s magnitude
    base = CALIBRATION_SCALE * per_day

    seq = []
    for t in range(window):
        wobble = 0.08 * base * np.sin((t + month) / 3.0)   # mild seasonality
        seq.append(max(0.0, base + wobble))

    return np.asarray(seq, dtype=np.float32)

# Mappings used by the model
COUNTRY_ID_MAP = {
    'Bangladesh': 0, 'Benin': 1, 'Burkina Faso': 2, 'Congo DRC': 3,
    'Ghana': 4, 'Haiti': 5, 'Malawi': 6, 'Mali': 7, 'Rwanda': 8,
    'Senegal': 9, 'Tanzania': 10, 'Togo': 11, 'Uganda': 12, 'Zambia': 13
}
ID_TO_COUNTRY = {v: k for k, v in COUNTRY_ID_MAP.items()}

# =============================================================================
# Routes
# =============================================================================
@main.route('/')
def home():
    return render_template('index.html')


@main.route('/dashboard')
@login_required
def dashboard():
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()

    unique_countries = (
        db.session.query(func.count(func.distinct(Prediction.country)))
        .filter_by(user_id=current_user.id)
        .scalar() or 0
    )

    most_predicted_medicine_query = (
        db.session.query(Prediction.medicine, func.count(Prediction.medicine).label('count'))
        .filter_by(user_id=current_user.id)
        .group_by(Prediction.medicine)
        .order_by(func.count(Prediction.medicine).desc())
        .first()
    )
    most_predicted_medicine = most_predicted_medicine_query[0] if most_predicted_medicine_query else "N/A"

    quotes = [
        {"text": "In God we trust. All others must bring data.", "author": "W. Edwards Deming"},
        {"text": "The goal is to turn data into information, and information into insight.", "author": "Carly Fiorina"},
        {"text": "Without data, you're just another person with an opinion.", "author": "W. Edwards Deming"},
        {"text": "Data is the new oil.", "author": "Clive Humby"},
    ]
    random_quote = random.choice(quotes)

    return render_template(
        'dashboard.html',
        total_predictions=total_predictions,
        unique_countries=unique_countries,
        most_predicted_medicine=most_predicted_medicine,
        quote=random_quote
    )


@main.route('/predict', methods=['POST'])
@login_required
def predict():
    country = request.form.get('country') or ""
    medicine = request.form.get('medicine') or ""
    # NEW: year selection (optional, defaults to current year)
    year_str = request.form.get('year', '').strip()

    # Default prediction anchor date: July 15 of selected year (seasonally central & stable)
    try:
        target_year = int(year_str) if year_str else date.today().year
        pred_date = date(target_year, 7, 15)
    except Exception:
        pred_date = date.today()

    predicted_demand_str = "Prediction Error"
    demand_category = "Unknown"

    # If models not loaded, short-circuit
    if regressor_model is None or classifier_model is None:
        return render_template(
            'result.html',
            country=country,
            medicine=medicine,
            prediction="Models not loaded.",
            demand_category="Unknown",
            history_labels=json.dumps([]),
            history_values=json.dumps([])
        )

    # Mappings (must match training)
    medicine_mapping = {'Family Planning and Reproduction': 0}
    if country not in COUNTRY_ID_MAP or medicine not in medicine_mapping:
        return render_template(
            'result.html',
            country=country,
            medicine=medicine,
            prediction="Unsupported country/medicine for this model.",
            demand_category="Unknown",
            history_labels=json.dumps([]),
            history_values=json.dumps([])
        )

    try:
        # 1) static_input: (1,1) int32
        country_id = np.int32(COUNTRY_ID_MAP[country])
        static_input = np.array([[country_id]], dtype=np.int32)

        # 2) sequence_input: (1,14,1) float32 — user history -> global history -> country baseline (year-aware)
        seq = build_sequence_from_history(current_user.id, country, medicine, window=14, pred_date=pred_date)

        if not np.any(seq):  # no user history
            gseq = build_sequence_from_global_history(country, medicine, window=14)
            if gseq is not None and np.any(gseq):
                seq = gseq

        if not np.any(seq):  # still none -> priors baseline (country-specific)
            month = pred_date.month
            seq = build_country_baseline_sequence(country, month, window=14)

        sequence_input = seq.reshape(1, 14, 1).astype(np.float32)

        # 3) time_features_input: (1,11) float32
        # [day_of_week, month_of_year, quarter, day_of_year,
        #  is_weekend, is_month_end, is_quarter_end,
        #  rolling_mean_7, rolling_std_7, rolling_mean_14, rolling_mean_30]
        cal_feats = calendar_features(pred_date)

        def rmean(a, w): return float(np.mean(a[-w:])) if len(a) >= w else 0.0
        def rstd(a, w):  return float(np.std(a[-w:], ddof=0)) if len(a) >= w else 0.0

        rolling_mean_7  = rmean(seq, 7)
        rolling_std_7   = rstd(seq, 7)
        rolling_mean_14 = rmean(seq, 14)
        rolling_mean_30 = 0.0  # not enough history in-app; replace if you have a longer series

        time_features_input = np.array([cal_feats + [
            rolling_mean_7, rolling_std_7, rolling_mean_14, rolling_mean_30
        ]], dtype=np.float32)

        # Order inputs exactly as model expects (by tensor names)
        reg_inputs = order_inputs_by_model(regressor_model, static_input, sequence_input, time_features_input)
        cls_inputs = order_inputs_by_model(classifier_model, static_input, sequence_input, time_features_input)

        # Debug (optional)
        print("Regressor ordered shapes:", [x.shape for x in reg_inputs])
        print("Classifier ordered shapes:", [x.shape for x in cls_inputs])

        # Predict
        # --- CSV (Bayesian) first: if 2026 + we have priors for the country ---
        reg_value = None
        model_used = "Default"

        pri = COUNTRY_PRIORS.get(country)
        if pred_date.year == 2026 and pri is not None:
            # Reconstruct a per-day baseline using your CSV priors (same logic as our baseline function)
            avg_daily = float(pri.get("avg_daily", 0.35))
            freq      = float(pri.get("freq", 0.50))
            annual    = pri.get("annual")
            # Blend avg_daily and annual/365 for robustness
            per_day_from_annual = (float(annual) / 365.0) if annual is not None else avg_daily
            per_day = 0.5 * (avg_daily + per_day_from_annual) * max(0.2, min(1.0, freq))

            # Scale into model-like units using your existing calibration scale
            # (This keeps the magnitude similar to your Keras output so the UI looks consistent.)
            monthly_est = per_day * 30.0 * CALIBRATION_SCALE
            reg_value = float(monthly_est)
            model_used = "Bayesian (CSV 2026)"

        # If CSV did not provide a value (different year or missing row), fall back to the live model:
        if reg_value is None:
            reg_inputs = order_inputs_by_model(reg_model, static_input, sequence_input, time_features_input)
            reg_out = reg_model.predict(reg_inputs, verbose=0)
            reg_value = float(reg_out[0][0])
            model_used = "Default"

        predicted_demand_str = f"{int(round(reg_value)):,} units"

        # (Classifier can still run live, or you can also add a CSV-based class if you saved it.)


        # Save successful prediction (NOTE: we don't use today's prediction as history next time)
        new_prediction = Prediction(
            country=country,
            medicine=medicine,
            predicted_demand=predicted_demand_str,
            user_id=current_user.id
        )
        db.session.add(new_prediction)
        db.session.commit()

    except Exception as e:
        traceback.print_exc()
        predicted_demand_str = f"Prediction failed: {str(e)}"
        demand_category = "Unknown"

    # Build history for charts
    historical_preds = (
        Prediction.query
        .filter_by(user_id=current_user.id, country=country, medicine=medicine)
        .order_by(Prediction.timestamp.asc())
        .all()
    )
    history_labels = [p.timestamp.strftime('%b %d, %Y') for p in historical_preds]
    history_values = []
    for p in historical_preds:
        try:
            history_values.append(float(p.predicted_demand.split(' ')[0].replace(',', '')))
        except Exception:
            continue

    return render_template(
        'result.html',
        country=country,
        medicine=medicine,
        prediction=predicted_demand_str,
        demand_category=demand_category,
        history_labels=json.dumps(history_labels),
        history_values=json.dumps(history_values)
    )


@main.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    user_predictions = (
        Prediction.query
        .filter_by(user_id=current_user.id)
        .order_by(Prediction.timestamp.desc())
        .paginate(page=page, per_page=10)
    )
    return render_template('history.html', predictions=user_predictions)


@main.route('/map')
@login_required
def map_view():
    return render_template('map.html')


@main.route('/map_data')
@login_required
def map_data():
    subq = (
        db.session.query(
            Prediction.country,
            func.max(Prediction.timestamp).label('max_ts')
        ).group_by(Prediction.country)
        .subquery()
    )
    predictions = (
        db.session.query(Prediction)
        .join(subq, (Prediction.country == subq.c.country) & (Prediction.timestamp == subq.c.max_ts))
        .all()
    )
    data_for_map = {}
    for p in predictions:
        try:
            val = float(p.predicted_demand.split(' ')[0].replace(',', ''))
            data_for_map[p.country] = val
        except Exception:
            continue
    return jsonify(data_for_map)


@main.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    page = request.args.get('page', 1, type=int)
    all_predictions = (
        Prediction.query
        .order_by(Prediction.timestamp.desc())
        .paginate(page=page, per_page=10)
    )
    return render_template('admin_dashboard.html', all_predictions=all_predictions)


# -----------------------------------------------------------------------------
# Debug: See the model's true input names, shapes, and dtypes
# -----------------------------------------------------------------------------
@main.route('/debug_inputs')
@login_required
def debug_inputs():
    def desc(m):
        if not m:
            return []
        return [(t.name.split(':')[0], tuple(t.shape.as_list()), str(t.dtype.name)) for t in m.inputs]
    reg = desc(regressor_model)
    cls = desc(classifier_model)
    print("Regressor expected inputs:", reg)
    print("Classifier expected inputs:", cls)
    return f"<pre>Regressor: {reg}\nClassifier: {cls}</pre>"


@main.route('/insights')
@login_required
def insights():
    # Basic model status + inputs (safe to render)
    def desc(m):
        if not m:
            return []
        return [(t.name.split(':')[0], tuple(t.shape.as_list()), str(t.dtype.name)) for t in m.inputs]

    reg_inputs = desc(regressor_model)
    cls_inputs = desc(classifier_model)

    model_status = {
        "classifier_loaded": classifier_model is not None,
        "regressor_loaded": regressor_model is not None,
        "regressor_inputs": reg_inputs,
        "classifier_inputs": cls_inputs,
    }

    # Reuse your priors table so you can see what’s being used
    return render_template(
        'insights.html',
        model_status=model_status,
        country_priors=COUNTRY_PRIORS,
        calibration_scale=CALIBRATION_SCALE
    )

@main.route('/optimization')
@login_required
def optimization():
    """
    Safe viewer for Bayesian Optimization results.
    Never raises 500: shows a friendly message instead.
    """
    try:
        # If you defined `optimization_results` at import-time:
        data = globals().get("optimization_results", None)

        # Allow re-load on demand via query param: /optimization?reload=1
        reload_flag = request.args.get("reload")
        if reload_flag:
            # Reuse the same loader you added earlier
            data = _try_load_bo_results(MODELS_DIR)
            globals()["optimization_results"] = data  # cache globally

        # Normalize to something the template can render
        best = None
        rows = []

        if data is None:
            return render_template("optimization.html",
                                   error="No optimization results found (file missing/unreadable).",
                                   best=None, rows=[])

        # If loader returned an Optuna-like dict we created earlier
        if isinstance(data, dict):
            # Optional keys:
            best = {
                "params": data.get("best_params"),
                "value": data.get("best_value"),
                "source": data.get("_source"),
            } if ("best_params" in data or "best_value" in data) else None

            trials = data.get("trials", data)
            if isinstance(trials, list):
                for t in trials:
                    if isinstance(t, dict):
                        rows.append(t)
                    elif isinstance(t, (list, tuple)) and len(t) >= 3:
                        rows.append({"number": t[0], "value": t[1], "params": t[2]})
                    else:
                        rows.append({"item": repr(t)})
            elif isinstance(trials, dict):
                rows = [{"param": k, "value": v} for k, v in trials.items()]
            else:
                rows = [{"item": repr(trials)}]

        elif isinstance(data, list):
            # Generic list
            for t in data:
                if isinstance(t, dict):
                    rows.append(t)
                else:
                    rows.append({"item": repr(t)})

        else:
            # Fallback: just show repr
            rows = [{"item": repr(data)}]

        return render_template("optimization.html",
                               error=None, best=best, rows=rows)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("optimization.html",
                               error=f"Failed to render optimization results: {e}",
                               best=None, rows=[])




@main.route('/debug_bo_file')
@login_required
def debug_bo_file():
    import os, binascii
    p = os.path.join(os.path.dirname(__file__), "models", "bayesian_optimization_results (1).pkl")
    if not os.path.exists(p):
        return f"File not found: {p}", 404
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        head = f.read(32)
    hexhead = binascii.hexlify(head).decode("ascii")
    return f"<pre>Path: {p}\nSize: {size} bytes\nHead(32): {hexhead}</pre>"

@main.route("/csv_results")
@login_required
def csv_results():
    # Convert COUNTRY_PRIORS dict -> sorted list of (country, stats) by annual desc
    rows = []
    for c, v in (COUNTRY_PRIORS or {}).items():
        try:
            annual = float(v.get("annual") or 0.0)
        except Exception:
            annual = 0.0
        rows.append((c, v, annual))
    rows.sort(key=lambda x: x[2], reverse=True)
    rows = [(c, v) for (c, v, _) in rows]
    return render_template("csv_results.html", rows=rows)

@main.route("/csv_debug")
@login_required
def csv_debug():
    # Shows file(s) found, columns, preview, and how many countries loaded
    return jsonify({
        "models_dir": MODELS_DIR,
        "debug": CSV_DEBUG_INFO,
        "countries_loaded": len(COUNTRY_PRIORS),
        "sample": dict(list(COUNTRY_PRIORS.items())[:5]),
    })


@main.route("/list_models")
@login_required
def list_models():
    try:
        files = os.listdir(MODELS_DIR)
        return jsonify({"models_dir": MODELS_DIR, "files": files})
    except Exception as e:
        return jsonify({"error": str(e)})
@main.route("/csv_check")
@login_required
def csv_check():
    import glob, pandas as pd
    pattern = os.path.join(MODELS_DIR, "forecast_summary_2026*.csv")
    files = glob.glob(pattern)
    if not files:
        return jsonify({"error": f"No CSV files matched {pattern}"})

    path = files[0]
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        return jsonify({
            "path": path,
            "columns": list(df.columns),
            "sample_rows": df.head(3).to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e), "path": path})


@main.route("/csv_reload")
@login_required
def csv_reload():
    try:
        count = load_csv_forecast_2026(MODELS_DIR)
        return jsonify({"reloaded": True, "countries_loaded": count, "total_in_memory": len(COUNTRY_PRIORS)})
    except Exception as e:
        return jsonify({"reloaded": False, "error": str(e), "total_in_memory": len(COUNTRY_PRIORS)}), 500

