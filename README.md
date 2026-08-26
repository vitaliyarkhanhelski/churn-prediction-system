# System Predykcji Odpływu Klientów

Praca dyplomowa – Vitaliy Arkhanhelski & Szymon Wiśniewski

---

## O projekcie

Inteligentny system wspierający dział handlowy w proaktywnym zapobieganiu odejściu klientów. System łączy Machine Learning, Explainable AI, dopasowanie kampanii marketingowych (RAG) oraz automatyzację wysyłki powiadomień e-mail.

Opiekun pracy: dr Grażyna Musiatowicz-Podbiał

---

## Jak działa system

```
1. Opiekun klientów wgrywa plik CSV z danymi klientów przez interfejs webowy (Streamlit)
2. Model ML (Voting Classifier: XGBoost + SVM + Regresja Logistyczna) przewiduje
   prawdopodobieństwo odejścia każdego klienta
3. Analiza SHAP wyjaśnia DLACZEGO klient może odejść (które cechy mają największy wpływ)
4. Opiekun klika "Wyślij e-maile" – system natychmiast potwierdza zlecenie (toast)
   i uruchamia wysyłkę asynchronicznie w tle (nie blokuje UI)
5. Agent AI (LangGraph) generuje spersonalizowaną treść e-maila per klient, wyszukuje
   kandydatów w bazie wektorowej (RAG), wybiera JEDNĄ najlepiej dopasowaną kampanię
   i wysyła e-mail z załączonym PDF-em tej kampanii przez Gmail SMTP
6. Każda wysyłka jest logowana w emails/ ze statusem ✅/❌ i treścią wiadomości
```

---

## Stos technologiczny

| Warstwa | Technologia |
|---------|-------------|
| Interfejs użytkownika | Streamlit (Python) |
| Model ML | VotingClassifier (XGBoost, SVM, Logistic Regression) |
| Explainable AI | SHAP (SHapley Additive exPlanations) |
| Agent AI / generowanie e-maili | LangGraph + LangChain |
| Dopasowanie kampanii marketingowej (RAG) | Chroma (baza wektorowa) + OpenAI Embeddings |
| Model językowy | OpenAI GPT-4o (async, 2 e-maile naraz, szybszy niż Bielik) / Bielik via Ollama (lokalnie, bezpłatny, sekwencyjnie) |
| Wysyłka e-maili | Gmail SMTP (Python `smtplib`) |
| Optymalizacja hiperparametrów | Optuna |

---

## Struktura projektu

```
main/
├── app.py                          # Interfejs webowy (Streamlit)
├── predict_churn.py                # Pipeline predykcji ML + SHAP
├── langgraph_agent.py              # Agent AI (LangGraph) – generowanie i wysyłka e-maili
├── prompts.py                      # Prompty systemowe dla agenta LLM
├── email_service.py                # Wysyłka e-maili przez Gmail SMTP
├── utils.py                        # Feature engineering (transformacje cech)
├── style.css                       # Style CSS interfejsu
├── models/
│   ├── voting_model.pkl            # Wytrenowany model (VotingClassifier)
│   └── xgboost_for_shap.pkl        # Model XGBoost do analizy SHAP
├── data/
│   └── Bank Customer Churn Prediction Test.csv   # Dane testowe
├── results/
│   └── churn_YYYY-MM-DD_HH-MM-SS.csv             # Wyniki analiz (per uruchomienie)
├── emails/
│   └── YYYY-MM-DD_HH-MM-SS_<customer_id>.txt     # Logi wysłanych e-maili (treść + status)
├── rag/
│   ├── Marketing_cam/               # Opisy kampanii marketingowych (.docx) – źródło RAG
│   └── Files_to_attach/             # Broszury kampanii (.pdf) – załączane do e-maili
├── Marketing_cam_test/              # Baza wektorowa Chroma (zbudowana z rag/Marketing_cam)
├── 01_build_vector_db.ipynb         # Notebook budujący bazę wektorową kampanii
└── .streamlit/
    └── config.toml                 # Konfiguracja motywu Streamlit

bank_customer_churn/                # Samodzielny notebook do trenowania modelu
├── bank_customer_churn_model_training.ipynb
├── utils.py
├── data/                           # Dataset z Kaggle (pobierany automatycznie)
├── models/                         # Zapisane modele .pkl
└── output/                         # Wykresy SHAP + CSV z wartościami
```

---

## Uruchomienie lokalne

### Wymagania i instalacja zależności

Aby zainstalować wszystkie biblioteki wymagane do uruchomienia projektu, otwórz terminal w głównym folderze i wpisz:

```bash
pip install -r requirements.txt

### Konfiguracja e-mail i OpenAI

Utwórz plik `main/.env`:
```
SENDER_EMAIL=twoje.konto@gmail.com
SENDER_APP_PASSWORD=xxxx xxxx xxxx xxxx
OPENAI_API_KEY=sk-...
```

> App Password generujesz w Google: Konto → Bezpieczeństwo → Weryfikacja dwuetapowa → Hasła do aplikacji

### Bielik (opcjonalnie – lokalny LLM)

```bash
# Zainstaluj Ollama: https://ollama.com
ollama pull SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M
ollama serve
```

Wymagania: ~6.5 GB miejsca na dysku. Wysyłka w tle – UI nie czeka na zakończenie.

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

## Wysyłka e-maili – architektura async

Po kliknięciu "Wyślij e-maile" system natychmiast odblokowuje UI (toast z potwierdzeniem) i uruchamia wysyłkę w osobnym wątku (`threading.Thread`):

- **OpenAI GPT-4o** – 2 e-maile generowane równolegle (`asyncio.Semaphore(2)`, ograniczone celowo, żeby nie przekraczać limitu tokenów/min OpenAI). Każdy e-mail to kilka wywołań LLM (analiza SHAP, wyszukanie i streszczenie kampanii marketingowej, treść e-maila), więc realnie liczy się to w minutach, nie sekundach.
- **Bielik (Ollama)** – jeden po drugim (`Semaphore(1)`), wolniej niż OpenAI, ale UI nie czeka

Wyszukanie i wybór kampanii (RAG) zawsze korzysta z OpenAI, niezależnie od modelu wybranego w UI – lokalny Bielik zawodził przy porównywaniu kampanii (np. wybierał kampanię dla Hiszpanii dla klienta z Francji). Do OpenAI trafia wtedy tylko zanonimizowany profil ryzyka, bez ID klienta. Uzasadnienie wyboru kampanii zapisywane jest w logu e-maila.

Każda wysyłka zapisuje log do `emails/<timestamp>_<customer_id>.txt` z treścią e-maila i statusem `✅ Wysłano pomyślnie` lub `❌ Błąd wysyłki`. Wywołania LLM automatycznie ponawiają się przy chwilowym przekroczeniu limitu OpenAI (rate limit), a streszczenia kampanii marketingowych są cache'owane, żeby nie liczyć ich od nowa dla każdego klienta. Jeśli cały proces dla klienta się nie powiedzie, log trafia do `emails/<timestamp>_<customer_id>_FAILED.txt` zamiast znikać bez śladu.

---

## Wdrożenie

**Lokalnie (POC/demo):** `streamlit run app.py` → dostępne pod `http://localhost:8501`

**Sieć firmowa (małe firmy):** uruchom na jednym komputerze w sieci lokalnej – pracownicy wchodzą przez `http://<IP>:8501`. Ollama działa jako lokalny serwer LLM dostępny w sieci.

**Produkcja:** architektura modułowa pozwala na migrację bez przepisywania kodu – Streamlit na VPS, Ollama API za nginxem, modele ML na osobnym serwisie.

---

## Status projektu

- [x] Model ML (VotingClassifier) + analiza SHAP – ROC-AUC = 0.86
- [x] Interfejs webowy (Streamlit) – upload CSV, wizualizacja wyników, wykres SHAP
- [x] Eksport wyników z timestampem
- [x] Serwis e-mail (Gmail SMTP)
- [x] Agent AI (LangGraph) – generowanie spersonalizowanych e-maili per klient
- [x] Wybór modelu LLM (OpenAI GPT-4o / Bielik) z poziomu UI
- [x] Asynchroniczna wysyłka e-maili (nie blokuje UI)
- [x] Logowanie e-maili ze statusem dostarczenia
- [x] Dopasowanie kampanii marketingowej (RAG + Chroma) z załącznikiem PDF
- [x] Podgląd statusu wysyłki e-maili na żywo w UI
- [x] Notebook trenowania modelu z pobieraniem danych z Kaggle
- [ ] Prezentacja końcowa
