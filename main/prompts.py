"""
prompts.py – Szablony promptów dla węzłów LangGraph.
"""

# Mapowanie nazw technicznych SHAP → czytelne opisy po polsku
FEATURE_LABELS: dict[str, str] = {
    "shap_credit_score":    "scoring kredytowy klienta",
    "shap_age":             "wiek klienta",
    "shap_tenure":          "długość relacji z bankiem (w latach)",
    "shap_balance":         "saldo konta bankowego",
    "shap_products_number": "liczba posiadanych produktów bankowych",
    "shap_active_member":   "aktywność klienta (czy aktywnie korzysta z banku)",
    "shap_country_France":  "kraj zamieszkania: Francja",
    "shap_country_Germany": "kraj zamieszkania: Niemcy",
    "shap_country_Spain":   "kraj zamieszkania: Hiszpania",
    "shap_gender_Female":   "płeć: kobieta",
    "shap_gender_Male":     "płeć: mężczyzna",
}

EXPLAIN_SHAP_SYSTEM = """Przekształcasz techniczne czynniki ryzyka w zdania dla managera sprzedaży.

PRZYKŁAD WEJŚCIA:
🔴 aktywność klienta: duży wpływ – zwiększa ryzyko odejścia
🟡 saldo konta bankowego: umiarkowany wpływ – zwiększa ryzyko odejścia
🟢 wiek klienta: zmniejsza ryzyko odejścia – czynnik ochronny

PRZYKŁAD WYJŚCIA (skopiuj ten format dokładnie):
1. 🔴 Klient rzadko korzysta z usług banku, co świadczy o niskim zaangażowaniu i wysokim ryzyku odejścia.
2. 🟡 Saldo konta utrzymuje się na umiarkowanym poziomie, co może sygnalizować stopniowe wycofywanie środków.
3. 🟢 Wiek klienta jest czynnikiem stabilizującym – zmniejsza prawdopodobieństwo odejścia.

ZASADY (bezwzględnie obowiązujące):
- Zacznij każdy punkt od numeru i DOKŁADNIE tego samego emoji co na wejściu (🔴, 🟡 lub 🟢).
- Użyj numeracji 1. 2. 3. – nigdy myślników, gwiazdek ani innych znaków.
- Jedno zdanie na punkt, po polsku, językiem biznesowym.
- NIE dodawaj żadnego tekstu przed ani po liście."""

EXPLAIN_SHAP_HUMAN = """Czynniki ryzyka:
{dane_analiza}

Zwróć ponumerowaną listę zgodnie z przykładem. Zachowaj emoji."""

PREPARE_MAIL_SYSTEM = """Składasz e-mail alertowy do opiekuna klienta na podstawie gotowej analizy.

PRZYKŁAD GOTOWEGO E-MAILA (skopiuj ten format dokładnie, wstaw tylko właściwe wartości):

Szanowny Opiekunie Klienta,

Klient o ID 15785367 wykazuje 81.9% ryzyko odpływu i wymaga pilnego kontaktu.

Przyczyny tego ryzyka:
1. 🔴 Klient rzadko korzysta z usług banku, co świadczy o niskim zaangażowaniu.
2. 🟡 Saldo konta utrzymuje się na umiarkowanym poziomie, co może sygnalizować wycofywanie środków.
3. 🟢 Wiek klienta jest czynnikiem stabilizującym – zmniejsza prawdopodobieństwo odejścia.

Proszę o podjęcie działań retencyjnych w ciągu 48 godzin.

Z poważaniem,
System Predykcji Odpływu Klientów

ZASADY (bezwzględnie obowiązujące):
- Pierwsze zdanie ZAWSZE: "Klient o ID [ID] wykazuje [X]% ryzyko odpływu i wymaga pilnego kontaktu."
- Nagłówek listy ZAWSZE: "Przyczyny tego ryzyka:"
- Skopiuj wszystkie punkty z analizy słowo w słowo – zachowaj emoji 🔴 🟡 🟢.
- Używaj numeracji 1. 2. 3. – nigdy myślników.
- Ostatnie zdanie ZAWSZE: "Proszę o podjęcie działań retencyjnych w ciągu 48 godzin."
- Podpis ZAWSZE: "System Predykcji Odpływu Klientów"
- NIE dodawaj żadnego tekstu spoza szablonu."""

PREPARE_MAIL_HUMAN = """ID klienta: {customer_id}
Ryzyko odpływu: {churn_probability}

Analiza przyczyn (skopiuj te punkty słowo w słowo):
{data_work}"""
