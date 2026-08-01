import joblib

import numpy as np
import pandas as pd
import shap

from utils import feature_in_XGB

# ==========================================
# --- KONFIGURACJA BIZNESOWA I ŚCIEŻKI ---
# ==========================================

# Próg "Precyzyjny" – precision > 89%. Mniej telefonów, prawie każdy to faktyczny uciekinier.
THRESHOLD_PRECYZYJNY = 0.79

# Próg "Szeroki zasięg" – maksymalne F1 (0.639). Łapiesz 2x więcej uciekinierów, ale więcej fałszywych alarmów.
THRESHOLD_SZEROKI = 0.53

MODEL_VOTING_PATH = 'models/voting_model.pkl'
MODEL_XGB_PATH    = 'models/xgboost_for_shap.pkl'
INPUT_DATA_PATH   = 'data/Bank Customer Churn Prediction Test.csv'
OUTPUT_DATA_PATH  = 'results/dzisiejsze_cele_ratunkowe.csv'

# ==========================================
# --- USTAWIENIA WIDOKU KONSOLI ---
# ==========================================
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)
pd.reset_option("display.max_rows")
pd.set_option("display.width", 1000)
pd.set_option("display.expand_frame_repr", False)


def run_prediction(df_input: pd.DataFrame, strategy: str = "precyzyjny") -> pd.DataFrame:
    """
    Uruchamia pełny pipeline predykcji churnu na podanym DataFrame.

    Parametry
    ----------
    df_input : pd.DataFrame
        Dane klientów. Musi zawierać kolumnę 'customer_id' oraz wszystkie
        wymagane cechy modelu. Kolumna 'churn' nie jest wymagana.
    strategy : str
        Strategia marketingowa:
          - "precyzyjny"  → próg 0.79 (precision > 89%, mniej fałszywych alarmów)
          - "szeroki"     → próg 0.53 (max F1, więcej uratowanych klientów)

    Zwraca
    -------
    tuple[pd.DataFrame, shap.Explanation]
        - DataFrame z kolumnami: customer_id, churn_probability, marketing_action,
          oryginalne cechy oraz shap_<feature> dla każdej cechy.
        - Obiekt Explanation SHAP (shap_vals) potrzebny do wykresów waterfall.
    """
    threshold = THRESHOLD_PRECYZYJNY if strategy == "precyzyjny" else THRESHOLD_SZEROKI

    voting_model = joblib.load(MODEL_VOTING_PATH)
    model_XGB    = joblib.load(MODEL_XGB_PATH)

    customer_ids = df_input['customer_id']
    X_new = df_input.drop(columns=['customer_id'])

    y_proba = voting_model.predict_proba(X_new)[:, 1]

    preproc   = model_XGB.named_steps['preprocessor']
    XGB_model = model_XGB.named_steps['model']

    X_feat = feature_in_XGB(X_new)
    X_t    = preproc.transform(X_feat)

    feature_names = [k.split('__')[1] for k in preproc.get_feature_names_out()]

    explainer  = shap.TreeExplainer(XGB_model)
    shap_vals  = explainer(X_t)
    shap_vals.feature_names = feature_names

    df_shap = pd.DataFrame(
        shap_vals.values,
        columns=[f"shap_{col}" for col in feature_names],
        index=X_new.index
    )

    marketing_flag = (y_proba >= threshold).astype(int)

    df_results = pd.DataFrame({
        'customer_id':       customer_ids.values,
        'churn_probability': np.round(y_proba, 3),
        'marketing_action':  marketing_flag,
    }, index=X_new.index)

    df_final = pd.concat([df_results, X_new, df_shap], axis=1)
    return df_final, shap_vals


def main():
    """Uruchamia predykcję z terminala używając hardcodowanych ścieżek."""
    print("1. Wczytywanie danych...")
    df_raw = pd.read_csv(INPUT_DATA_PATH)

    print("2. Uruchamianie pipeline predykcji (strategia: precyzyjny)...")
    df_final, _ = run_prediction(df_raw, strategy="precyzyjny")

    print("\nPodgląd wyników (pierwsze 3 wiersze):")
    print(df_final.head(3))

    df_final.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"\n--- SUKCES! Wyniki zapisano w pliku: {OUTPUT_DATA_PATH} ---")


if __name__ == "__main__":
    main()
