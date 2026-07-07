import streamlit as st
import pandas as pd
import time

st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Konfiguracja")
    st.markdown("---")
    st.caption("Model")
    model_choice = st.selectbox("Model ML", ["XGBoost", "XGBoost + Sieć neuronowa"])
    shap_enabled = st.toggle("Analiza SHAP", value=True)
    langchain_enabled = st.toggle("Generuj raport (LangChain)", value=True)
    n8n_enabled = st.toggle("Wyślij e-maile (N8N)", value=False, disabled=True,
                            help="Wymaga uruchomionego N8N lokalnie")
    st.markdown("---")
    st.caption("v0.1 – prototyp / demo")

# --- Header ---
st.title("📊 System Predykcji Churnu Klientów")
st.markdown(
    "Wgraj plik CSV z danymi klientów, uruchom analizę i otrzymaj listę klientów "
    "zagrożonych odejściem wraz z rekomendacjami dla opiekunów."
)
st.markdown("---")

# --- Upload ---
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.subheader("1. Wgraj dane")
    uploaded_file = st.file_uploader(
        "Przeciągnij lub wybierz plik CSV",
        type=["csv"],
        help="Plik powinien zawierać dane klientów oraz kolumnę z przypisanym opiekunem.",
    )

with col_info:
    st.subheader("Wymagane kolumny")
    st.markdown(
        """
        - `client_id` – ID klienta  
        - `account_manager` – imię / e-mail opiekuna  
        - cechy: historia zamówień, aktywność, itp.  
        - *(opcjonalnie)* `churn` – etykieta (do trenowania)
        """
    )

# --- Preview ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Wgrano plik: **{uploaded_file.name}** ({len(df)} rekordów, {len(df.columns)} kolumn)")
    with st.expander("Podgląd danych", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
else:
    st.info("Brak wgranego pliku. Wgraj CSV, żeby uruchomić analizę.")

st.markdown("---")

# --- Run ---
st.subheader("2. Uruchom analizę")

run_col, status_col = st.columns([1, 3])

with run_col:
    run_button = st.button(
        "▶ Uruchom analizę",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )

with status_col:
    if uploaded_file is None:
        st.warning("Najpierw wgraj plik CSV.")
    else:
        st.success("Dane gotowe. Kliknij 'Uruchom analizę'.")

# --- Dummy pipeline on click ---
if run_button:
    st.markdown("---")
    st.subheader("3. Wyniki")

    progress_bar = st.progress(0, text="Uruchamianie pipeline...")
    steps = [
        (20, "🔄 Przetwarzanie danych..."),
        (45, "🤖 Trenuję model XGBoost..."),
        (65, "🔍 Obliczam wartości SHAP..."),
        (85, "📝 Generuję raport (LangChain)..."),
        (100, "✅ Gotowe!"),
    ]
    for pct, msg in steps:
        time.sleep(0.6)
        progress_bar.progress(pct, text=msg)

    st.success("Analiza zakończona. Poniżej wyniki dla wgranego datasetu.")

    # --- Dummy results ---
    st.markdown("#### Klienci zagrożeni odejściem")

    dummy_results = pd.DataFrame({
        "Klient": ["Firma Alpha Sp. z o.o.", "Beta Industries", "Gamma Corp", "Delta S.A."],
        "Opiekun": ["Jan Kowalski", "Anna Nowak", "Jan Kowalski", "Anna Nowak"],
        "Ryzyko churnu": ["🔴 Wysokie (87%)", "🔴 Wysokie (79%)", "🟡 Średnie (61%)", "🟡 Średnie (54%)"],
        "Główna przyczyna (SHAP)": [
            "Spadek częstotliwości zamówień",
            "Wzrost liczby reklamacji",
            "Brak aktywności > 30 dni",
            "Obniżenie wartości koszyka",
        ],
        "Rekomendacja": [
            "Zadzwoń, zaproponuj rabat 10%",
            "Eskaluj problem reklamacji",
            "Wyślij ofertę reaktywacyjną",
            "Zaproponuj nowy produkt",
        ],
    })

    st.dataframe(dummy_results, use_container_width=True, hide_index=True)

    col_dl, col_n8n, _ = st.columns([1, 1, 2])
    with col_dl:
        st.download_button(
            "⬇ Pobierz raport CSV",
            data=dummy_results.to_csv(index=False),
            file_name="churn_raport.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_n8n:
        st.button(
            "📧 Wyślij e-maile (N8N)",
            use_container_width=True,
            disabled=True,
            help="Funkcja w trakcie implementacji",
        )

    # --- SHAP placeholder ---
    if shap_enabled:
        st.markdown("#### Analiza SHAP – ważność cech")
        st.info("📌 Tu pojawi się wykres SHAP (feature importance + waterfall per klient) po integracji z modelem.")
