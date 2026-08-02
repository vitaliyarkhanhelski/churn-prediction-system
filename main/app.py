import re
import sys
import os
import time
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from datetime import datetime

EMAIL_REGEX = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

sys.path.insert(0, os.path.dirname(__file__))
from predict_churn import run_prediction
from langgraph_agent import run_agent


def _handle_email_send():
    """on_click callback – ustawia flagę wysyłania."""
    if st.session_state.get("df_final") is None:
        return
    st.session_state["sending_emails"] = True

st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
)

with open(os.path.join(os.path.dirname(__file__), "style.css")) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# --- Toast po rerun ---
if "pending_toast" in st.session_state:
    msg, icon = st.session_state.pop("pending_toast")
    st.toast(msg, icon=icon)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Konfiguracja")
    st.markdown("---")

    st.caption("Strategia marketingowa")
    strategy_label = st.radio(
        label="Wybierz strategię",
        options=["🎯 Precyzyjny", "🔍 Szeroki zasięg"],
        index=0,
        help=(
            "**Precyzyjny** – dzwonimy tylko do klientów z bardzo wysokim ryzykiem odejścia. "
            "Mniej kontaktów, prawie każdy to faktyczny uciekinier.\n\n"
            "**Szeroki zasięg** – łapiemy więcej uciekinierów, ale wykonujemy więcej kontaktów, "
            "w tym część do klientów którzy i tak by zostali."
        ),
    )
    strategy = "precyzyjny" if "Precyzyjny" in strategy_label else "szeroki"

    st.markdown("---")
    st.caption("Model AI (generowanie e-maili)")
    model_choice = st.radio(
        label="Wybierz model",
        options=["☁️ OpenAI GPT-4o", "🦙 Bielik (lokalnie)"],
        index=0,
        help=(
            "**OpenAI GPT-4o** – szybki (~10s dla 4 e-maili), wymaga klucza API. "
            "Wysyła dane do chmury OpenAI.\n\n"
            "**Bielik (lokalnie)** – darmowy, dane zostają na Twoim komputerze. "
            "Wymaga uruchomionego Ollama (`ollama serve`). ~20s/e-mail sekwencyjnie."
        ),
    )
    model_backend = "ollama" if "Bielik" in model_choice else "openai"

    if model_backend == "ollama":
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:11434", timeout=1)
            st.caption("🆓 Bezpłatny · dane zostają lokalnie · wolniejszy (e-maile generowane jeden po drugim)")
        except Exception:
            st.warning("Ollama nie działa. Uruchom: `ollama serve`")
    else:
        st.caption("⚡ Kilka e-maili jednocześnie · wymaga klucza OpenAI")

    st.markdown("---")
    st.caption("v0.2 – POC")

# --- Header ---
st.title("🎯 System Predykcji Odpływu Klientów")
st.markdown(
    "Wgraj plik CSV z danymi klientów, wybierz strategię i uruchom analizę. "
    "System wskaże klientów zagrożonych odejściem oraz główne przyczyny ryzyka."
)
st.markdown("---")

# --- Upload ---
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.subheader("1. Wgraj dane")
    uploaded_file = st.file_uploader(
        "Przeciągnij lub wybierz plik CSV",
        type=["csv"],
        help="Plik musi zawierać kolumnę 'customer_id' oraz wymagane cechy modelu.",
    )
    manager_email = st.text_input(
        "📧 Twój e-mail (opiekun klientów)",
        placeholder="jan.kowalski@firma.pl",
        help="E-mail zostanie przypisany do wyników i użyty do wysyłki powiadomień.",
    )
    email_valid = bool(EMAIL_REGEX.match(manager_email)) if manager_email else False
    if manager_email and not email_valid:
        st.warning("Podaj poprawny adres e-mail.")

with col_info:
    st.subheader("Wymagane kolumny")
    st.markdown(
        """
        - `customer_id` – ID klienta
        - `credit_score` – scoring kredytowy
        - `country` – kraj (France / Spain / Germany)
        - `gender` – płeć (Male / Female)
        - `age`, `tenure`, `balance`
        - `products_number`, `credit_card`
        - `active_member`, `estimated_salary`
        """
    )

# --- Preview ---
RESULTS_KEYS = ["df_final", "shap_vals", "output_filename", "manager_email"]

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)

    if st.session_state.get("last_filename") != uploaded_file.name:
        for k in RESULTS_KEYS:
            st.session_state.pop(k, None)
        st.session_state["last_filename"] = uploaded_file.name

    st.success(
        f"Wgrano plik: **{uploaded_file.name}** "
        f"({len(df_uploaded)} rekordów, {len(df_uploaded.columns)} kolumn)"
    )
    with st.expander("Podgląd danych", expanded=True):
        rows = len(df_uploaded)
        row_height = 35
        header = 38
        preview_height = min(rows, 10) * row_height + header
        st.dataframe(df_uploaded, use_container_width=True, height=preview_height)
else:
    df_uploaded = None
    if "last_filename" in st.session_state:
        for k in RESULTS_KEYS + ["last_filename"]:
            st.session_state.pop(k, None)
    st.info("Brak wgranego pliku. Wgraj CSV, żeby uruchomić analizę.")

st.markdown("---")

# --- Strategy info ---
st.subheader("2. Strategia")
if strategy == "precyzyjny":
    st.info(
        "🎯 **Precyzyjny** – próg 0.79. Kontaktujesz się tylko z klientami "
        "o bardzo wysokim ryzyku odejścia. Precision > 89% – mniej przepalonych zniżek."
    )
else:
    st.info(
        "🔍 **Szeroki zasięg** – próg 0.53. Łapiesz ~2× więcej uciekinierów, "
        "ale wykonujesz więcej kontaktów. Optymalny balans precision/recall (F1 = 0.639)."
    )

st.markdown("---")

# --- Run ---
st.subheader("3. Uruchom analizę")

run_col, status_col = st.columns([1, 3])

ready = df_uploaded is not None and email_valid

with run_col:
    run_button = st.button(
        "▶ Uruchom analizę",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )

with status_col:
    if df_uploaded is None and not manager_email:
        st.warning("Wgraj plik CSV i podaj swój e-mail.")
    elif df_uploaded is None:
        st.warning("Wgraj plik CSV.")
    elif not email_valid:
        st.warning("Podaj poprawny adres e-mail.")
    else:
        st.success(
            f"✅ Gotowe do uruchomienia\n\n"
            f"**Rekordów:** {len(df_uploaded)}  \n"
            f"**Strategia:** {strategy_label}  \n"
            f"**Opiekun:** `{manager_email}`  \n"
            f"**Model AI:** {model_choice}"
        )

# --- Uruchomienie pipeline i zapis do session_state ---
if run_button and ready:
    progress = st.progress(0, text="Uruchamianie pipeline...")
    try:
        progress.progress(15, text="⏳ Ładowanie modeli z dysku...")
        time.sleep(0.5)
        progress.progress(40, text="🤖 Wyliczanie prawdopodobieństwa odejścia (Voting Model)...")
        time.sleep(0.5)
        progress.progress(65, text="🔍 Analiza przyczyn (SHAP / XGBoost)...")
        time.sleep(0.5)
        progress.progress(85, text="📊 Generowanie flag marketingowych...")

        df_final, shap_vals = run_prediction(df_uploaded, strategy=strategy)

        df_final.insert(1, "account_manager_email", manager_email)

        time.sleep(0.5)
        progress.progress(100, text="✅ Gotowe!")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"churn_{timestamp}.csv"
        output_path = os.path.join(os.path.dirname(__file__), "results", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)

        st.session_state["df_final"]            = df_final
        st.session_state["shap_vals"]           = shap_vals
        st.session_state["output_filename"]     = output_filename
        st.session_state["manager_email"]       = manager_email

    except Exception as e:
        progress.empty()
        st.error(f"Błąd podczas analizy: {e}")

# --- Wyniki (renderowane z session_state – przeżywają rerenderowanie) ---
if "df_final" in st.session_state:
    df_final        = st.session_state["df_final"]
    shap_vals       = st.session_state["shap_vals"]
    output_filename = st.session_state["output_filename"]

    st.markdown("---")
    st.subheader("4. Wyniki")

    n_flagged = int(df_final["marketing_action"].sum())
    n_total   = len(df_final)
    st.success(
        f"Analiza zakończona. Znaleziono **{n_flagged} z {n_total}** klientów "
        f"wymagających działania. Wyniki zapisano: `results/{output_filename}`"
    )

    # --- Tabela: klienci z flagą ---
    df_flagged = df_final[df_final["marketing_action"] == 1].copy()

    with st.expander(f"🚨 Klienci wymagający działania – {len(df_flagged)} os.", expanded=False):
        if df_flagged.empty:
            st.warning("Brak klientów spełniających kryterium dla wybranej strategii.")
        else:
            shap_cols = [c for c in df_final.columns if c.startswith("shap_")]

            def top3_reasons(row):
                top3 = row[shap_cols].astype(float).abs().nlargest(3).index.tolist()
                return ", ".join(c.replace("shap_", "") for c in top3)

            df_flagged["top_3_powody_SHAP"] = df_flagged.apply(top3_reasons, axis=1)
            st.dataframe(
                df_flagged[["customer_id", "account_manager_email", "churn_probability", "top_3_powody_SHAP"]].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                height=min(400, (len(df_flagged) + 1) * 35 + 10),
            )

    # --- Pełna tabela ---
    with st.expander("Pełne wyniki (wszyscy klienci)"):
        def highlight_at_risk(row):
            if row["marketing_action"] == 1:
                return ["background-color: #fff0f1"] * len(row)
            return [""] * len(row)

        styled = df_final.reset_index(drop=True).style.apply(highlight_at_risk, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Pobierz / Wyślij e-maile ---
    emails_sent_key = f"emails_sent_{output_filename}"
    already_sent    = st.session_state.get(emails_sent_key, False)
    is_sending      = st.session_state.get("sending_emails", False)

    # st.empty() – spinner ZASTĘPUJE przyciski w tym samym miejscu DOM (brak podwójnych)
    actions_slot = st.empty()

    if is_sending:
        with actions_slot.container():
            n_at_risk     = int(df_final["marketing_action"].sum())
            manager_email = st.session_state.get("manager_email", "")
            with st.spinner(f"Generowanie i wysyłanie {n_at_risk} e-mail{'i' if n_at_risk != 1 else 'a'} przez AI..."):
                if n_at_risk > 0:
                    errors = run_agent(df_flagged, to_email=manager_email, model_backend=model_backend)
                    if errors:
                        st.session_state["email_success_msg"] = f"⚠️ Wysłano z błędami: {'; '.join(errors)}"
                        st.session_state["pending_toast"] = (f"⚠️ Wysłano z błędami: {errors[0]}", "⚠️")
                    else:
                        msg = (
                            f"E-maile wysłane – {n_at_risk} klient{'ów' if n_at_risk != 1 else ''} "
                            f"zagrożon{'ych' if n_at_risk != 1 else 'y'} odejściem. Logi w folderze emails/."
                        )
                        st.session_state["email_success_msg"] = msg
                        st.session_state["pending_toast"] = (f"✅ {msg}", "📧")
                else:
                    msg = "Brak klientów zagrożonych odejściem – e-maile nie zostały wysłane."
                    st.session_state["email_success_msg"] = msg
                    st.session_state["pending_toast"] = (msg, "ℹ️")
            st.session_state["sending_emails"] = False
            st.session_state[emails_sent_key]  = True
            st.rerun()
    else:
        with actions_slot.container():
            col_dl, col_email, _ = st.columns([1, 1, 2])
            with col_dl:
                st.download_button(
                    "⬇ Pobierz wyniki CSV",
                    data=df_final.to_csv(index=False),
                    file_name=output_filename,
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_email:
                st.button(
                    "✅ E-maile wysłane" if already_sent else "📧 Wyślij e-maile",
                    type="primary",
                    use_container_width=True,
                    disabled=already_sent,
                    on_click=_handle_email_send,
                    help="E-maile zostały już wysłane dla tego uruchomienia." if already_sent else "Wysyła powiadomienia e-mail do opiekuna klientów zagrożonych odejściem",
                )

    if "email_success_msg" in st.session_state:
        msg = st.session_state["email_success_msg"]
        if "Brak klientów" in msg or "błędy" in msg.lower():
            st.warning(msg)
        else:
            st.success(msg)

    # =============================================
    # --- SHAP Waterfall – analiza klienta ---
    # =============================================
    st.markdown("---")
    st.subheader("5. Analiza SHAP – dlaczego klient może odejść?")
    st.markdown(
        "Wybierz klienta z listy. Wykres pokazuje które cechy **zwiększają** "
        "(czerwone) lub **zmniejszają** (niebieskie) ryzyko odejścia."
    )

    all_ids = df_final["customer_id"].tolist()
    selected_id = st.selectbox("Wybierz klienta (customer_id)", options=all_ids)

    if selected_id is not None:
        row_pos = df_final[df_final["customer_id"] == selected_id].index[0]
        churn_prob = df_final.loc[row_pos, "churn_probability"]
        flag = df_final.loc[row_pos, "marketing_action"]

        col_prob, col_flag = st.columns(2)
        with col_prob:
            st.metric("Ryzyko odejścia", f"{churn_prob * 100:.1f}%")
        with col_flag:
            st.metric(
                "Działanie marketingowe",
                "✅ Tak" if flag == 1 else "➖ Nie",
            )

        _, col_chart, _ = st.columns([1, 2, 1])
        with col_chart:
            fig, ax = plt.subplots(figsize=(6, 4))
            shap.plots.waterfall(shap_vals[row_pos], show=False)
            st.pyplot(plt.gcf(), clear_figure=True)

