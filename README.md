# System Predykcji Odpływu Klientów

Praca dyplomowa – Vitaliy Arkhanhelski & Szymon Wiśniewski

---

## O projekcie

Inteligentny system wspierający dział handlowy w proaktywnym zapobieganiu odejściu klientów. System łączy Machine Learning, Explainable AI oraz automatyzację wysyłki powiadomień e-mail.

Opiekun pracy: dr Grażyna Musiatowicz-Podbiał

---

## Jak działa system

```
1. Opiekun klientów wgrywa plik CSV z danymi klientów przez interfejs webowy (Streamlit)
2. Model ML (Voting Classifier: XGBoost + SVM + Regresja Logistyczna) przewiduje
   prawdopodobieństwo odejścia każdego klienta
3. Analiza SHAP wyjaśnia DLACZEGO klient może odejść (które cechy mają największy wpływ)
4. Agent AI (LangGraph) na podstawie wyników generuje spersonalizowaną treść e-maila
   z rekomendacją działań dla opiekuna
5. Opiekun klika "Wyślij e-maile" – system wysyła powiadomienia przez Gmail SMTP
```

---

## Stos technologiczny

| Warstwa | Technologia |
|---------|-------------|
| Interfejs użytkownika | Streamlit (Python) |
| Model ML | VotingClassifier (XGBoost, SVM, Logistic Regression) |
| Explainable AI | SHAP (SHapley Additive exPlanations) |
| Agent AI / generowanie e-maili | LangGraph + LangChain |
| Model językowy | OpenAI API (POC) / lokalny SLM – Bielik (docelowo) |
| Wysyłka e-maili | Gmail SMTP (Python `smtplib`) |
| Optymalizacja hiperparametrów | Optuna |

---

## Struktura projektu

```
main/
├── app.py                          # Interfejs webowy (Streamlit)
├── predict_churn.py                # Pipeline predykcji ML + SHAP
├── email_service.py                # Wysyłka e-maili przez Gmail SMTP
├── utils.py                        # Feature engineering (transformacje cech)
├── style.css                       # Style CSS interfejsu
├── Final_Vote_model_LC.ipynb       # Notebook: trenowanie modelu i eksport
├── models/
│   ├── voting_model.pkl            # Wytrenowany model (VotingClassifier)
│   └── xgboost_for_shap.pkl        # Model XGBoost do analizy SHAP
├── data/
│   └── Bank Customer Churn Prediction Test.csv   # Dane testowe
├── results/
│   └── churn_YYYY-MM-DD_HH-MM-SS.csv             # Wyniki analiz (per uruchomienie)
└── .streamlit/
    └── config.toml                 # Konfiguracja motywu Streamlit
```

---

## Uruchomienie lokalne

### Wymagania

```bash
pip install streamlit shap xgboost scikit-learn pandas numpy joblib python-dotenv
```

### Konfiguracja e-mail

Utwórz plik `main/.env`:
```
SENDER_EMAIL=twoje.konto@gmail.com
SENDER_APP_PASSWORD=xxxx xxxx xxxx xxxx
```

> App Password generujesz w Google: Konto → Bezpieczeństwo → Weryfikacja dwuetapowa → Hasła do aplikacji

### Start

```bash
cd main
streamlit run app.py
```

Aplikacja dostępna na `http://localhost:8501`

---

## Strategie marketingowe

System oferuje dwa tryby analizy:

| Strategia | Próg | Charakterystyka |
|-----------|------|-----------------|
| 🎯 **Precyzyjny** | 0.79 | Precision > 89% – mniej kontaktów, prawie każdy to faktyczny uciekinier |
| 🔍 **Szeroki zasięg** | 0.53 | Max F1 = 0.639 – więcej uratowanych klientów, więcej kontaktów |

---

## Status projektu

- [x] Model ML (VotingClassifier) + analiza SHAP
- [x] Interfejs webowy (Streamlit) – upload CSV, wizualizacja wyników, wykres SHAP
- [x] Eksport wyników z timestampem
- [x] Serwis e-mail (Gmail SMTP)
- [ ] Integracja z LangGraph (generowanie treści e-maili) – w trakcie
- [ ] Prezentacja końcowa
