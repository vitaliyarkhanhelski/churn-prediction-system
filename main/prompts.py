"""
prompts.py – Szablony promptów dla węzłów LangGraph.
"""

# Mapowanie nazw technicznych SHAP → czytelne opisy po polsku.
# Cechy liczbowe: do etykiety doklejana jest FAKTYCZNA wartość klienta (patrz _shap_to_text),
# żeby model nie musiał zgadywać, czy np. "liczba produktów" jest wysoka czy niska.
FEATURE_LABELS: dict[str, str] = {
    "shap_credit_score":    "scoring kredytowy klienta",
    "shap_age":             "wiek klienta",
    "shap_tenure":          "długość relacji z bankiem (w latach)",
    "shap_balance":         "saldo konta bankowego",
    "shap_products_number": "liczba posiadanych produktów bankowych",
    "shap_active_member":   "aktywność klienta",
}

# Cechy zakodowane one-hot: shap_country_Germany dotyczy CECHY "czy klient jest z Niemiec",
# a nie faktu, że klient tam mieszka (dla Francuza ta cecha ma wartość 0!). Dlatego
# _shap_to_text bierze pod uwagę tylko tę kolumnę, która odpowiada RZECZYWISTEJ wartości
# klienta, i opisuje ją jako fakt. Klucz: (kolumna w danych, wartość) -> etykieta.
ONE_HOT_LABELS: dict[tuple[str, str], str] = {
    ("country", "France"):  "kraj zamieszkania klienta (Francja)",
    ("country", "Germany"): "kraj zamieszkania klienta (Niemcy)",
    ("country", "Spain"):   "kraj zamieszkania klienta (Hiszpania)",
    ("gender", "Female"):   "płeć klienta (kobieta)",
    ("gender", "Male"):     "płeć klienta (mężczyzna)",
}

# Jak sformatować faktyczną wartość cechy liczbowej w etykiecie.
FEATURE_VALUE_FORMATTERS: dict[str, str] = {
    "shap_credit_score":    "{value:.0f} pkt",
    "shap_age":             "{value:.0f} lat",
    "shap_tenure":          "{value:.0f} lat",
    "shap_balance":         "{value:,.0f}",
    "shap_products_number": "{value:.0f}",
}

# Rozkłady cech w zbiorze treningowym (10 000 klientów). Bez tego model zna wartość, ale nie
# wie, czy jest wysoka czy niska – i zgadywał (np. opisywał saldo z górnych 10% jako
# "wycofywanie środków", a scoring z dolnych 10% jako "umiarkowany").
# Klucz: kolumna SHAP -> (25. percentyl, mediana, 75. percentyl).
FEATURE_QUARTILES: dict[str, tuple[float, float, float]] = {
    "shap_credit_score": (584, 652, 718),
    "shap_age":          (32, 37, 44),
    "shap_tenure":       (3, 5, 7),
    "shap_balance":      (0, 97_199, 127_644),
}

# products_number ma silnie nieliniowy związek z odejściem, więc zamiast percentyli
# podajemy faktyczny udział i odsetek odejść z danych treningowych.
PRODUCTS_NUMBER_CONTEXT: dict[int, str] = {
    1: "51% klientów, odchodzi 28% z nich",
    2: "46% klientów, odchodzi tylko 8% – najbezpieczniejsza grupa",
    3: "rzadkość: 3% klientów, odchodzi aż 83% z nich",
    4: "skrajna rzadkość: <1% klientów, odchodzą praktycznie wszyscy",
}

EXPLAIN_SHAP_SYSTEM = """Przekształcasz techniczne czynniki ryzyka w zdania dla managera sprzedaży.

PRZYKŁAD WEJŚCIA:
🔴 aktywność klienta (nieaktywny): duży wpływ – zwiększa ryzyko odejścia
🟡 saldo konta bankowego (12,500 – niska wartość, mediana to 97,199): umiarkowany wpływ – zwiększa ryzyko odejścia
🟢 wiek klienta (52 lat – wysoka wartość, mediana to 37): zmniejsza ryzyko odejścia – czynnik ochronny

PRZYKŁAD WYJŚCIA (skopiuj ten format dokładnie):
1. 🔴 Klient nie korzysta aktywnie z usług banku, co świadczy o niskim zaangażowaniu i wysokim ryzyku odejścia.
2. 🟡 Saldo konta wynosi 12 500 i jest wyraźnie niższe od typowego poziomu, co może sygnalizować wycofywanie środków.
3. 🟢 Klient ma 52 lata, czyli powyżej typowego wieku klientów banku – to czynnik stabilizujący, zmniejszający prawdopodobieństwo odejścia.

ZASADY (bezwzględnie obowiązujące):
- Zacznij każdy punkt od numeru i DOKŁADNIE tego samego emoji co na wejściu (🔴, 🟡 lub 🟢).
- Użyj numeracji 1. 2. 3. – nigdy myślników, gwiazdek ani innych znaków.
- Jedno zdanie na punkt, po polsku, językiem biznesowym.
- Wartości w nawiasach to FAKTY o kliencie – opisuj je zgodnie z prawdą, NIGDY nie zgaduj ani nie odwracaj ich znaczenia (np. przy "liczba produktów (3)" nie pisz, że klient ma niewiele produktów).
- Jeśli w nawiasie jest komentarz o rozkładzie (np. "rzadkość: 3% klientów", "niska wartość, mediana to 652"), oprzyj na nim ocenę wartości. NIE dodawaj własnych ocen typu "tylko" czy "aż", jeśli przeczą temu komentarzowi.
- Wniosek w zdaniu musi wynikać z FAKTYCZNEJ wartości tego klienta – nie kopiuj interpretacji z przykładu, jeśli wartość jest inna (np. przy saldzie POWYŻEJ mediany NIE pisz o wycofywaniu środków, bo to wniosek dla salda NISKIEGO).
- Każdy punkt musi być pełnym, poprawnym gramatycznie zdaniem – nigdy równoważnikiem (źle: "Klientka, co zwiększa ryzyko"; dobrze: "Płeć klienta (kobieta) umiarkowanie zwiększa ryzyko odejścia").
- NIE dodawaj żadnego tekstu przed ani po liście."""

EXPLAIN_SHAP_HUMAN = """Czynniki ryzyka:
{dane_analiza}

Zwróć ponumerowaną listę zgodnie z przykładem. Zachowaj emoji."""

PREPARE_MAIL_SYSTEM = """Składasz e-mail alertowy do opiekuna klienta na podstawie gotowej analizy oraz dopasowanej kampanii marketingowej.

PRZYKŁAD GOTOWEGO E-MAILA (skopiuj ten format dokładnie, wstaw tylko właściwe wartości):

Szanowny Opiekunie Klienta,

Klient o ID 15785367 wykazuje 81.9% ryzyko odpływu i wymaga pilnego kontaktu.

Przyczyny tego ryzyka:
1. 🔴 Klient rzadko korzysta z usług banku, co świadczy o niskim zaangażowaniu.
2. 🟡 Saldo konta utrzymuje się na umiarkowanym poziomie, co może sygnalizować wycofywanie środków.
3. 🟢 Wiek klienta jest czynnikiem stabilizującym – zmniejsza prawdopodobieństwo odejścia.

Rekomendowana kampania retencyjna:
Kampania "Dekada Zaufania" to program lojalnościowy oferujący preferencyjne warunki depozytów długoletnim klientom banku. Dobrze pasuje do tego klienta, ponieważ rzadkie korzystanie z usług banku wskazuje, że potrzebuje zachęty do ponownego zaangażowania się w relację z bankiem.

Proszę o podjęcie działań retencyjnych w ciągu 48 godzin.

Z poważaniem,
System Predykcji Odpływu Klientów

ZASADY (bezwzględnie obowiązujące):
- Pierwsze zdanie ZAWSZE: "Klient o ID [ID] wykazuje [X]% ryzyko odpływu i wymaga pilnego kontaktu."
- Nagłówek listy ZAWSZE: "Przyczyny tego ryzyka:"
- Skopiuj wszystkie punkty z analizy słowo w słowo – zachowaj emoji 🔴 🟡 🟢.
- Używaj numeracji 1. 2. 3. – nigdy myślników.
- Nagłówek sekcji kampanii ZAWSZE: "Rekomendowana kampania retencyjna:"
- W sekcji kampanii napisz DOKŁADNIE 2 zdania w tej strukturze: pierwsze zdanie – WŁASNYMI SŁOWAMI, jednym zdaniem, opisz czym jest podana kampania (nie kopiuj zdań z opisu); drugie zdanie – WŁASNYMI SŁOWAMI wyjaśnij dlaczego ta kampania pasuje akurat do TEGO klienta, łącząc ją z konkretnym czynnikiem ZWIĘKSZAJĄCYM ryzyko odejścia (🔴 lub 🟡) – NIGDY nie uzasadniaj wyboru czynnikiem ochronnym (🟢). Jeśli podana informacja mówi, że brak dopasowanej kampanii w bazie, napisz jedno zdanie rekomendujące standardowe kroki retencyjne.
- Ostatnie zdanie ZAWSZE: "Proszę o podjęcie działań retencyjnych w ciągu 48 godzin."
- Podpis ZAWSZE: "System Predykcji Odpływu Klientów"
- NIE dodawaj żadnego tekstu spoza szablonu."""

PREPARE_MAIL_HUMAN = """ID klienta: {customer_id}
Ryzyko odpływu: {churn_probability}

Analiza przyczyn (skopiuj te punkty słowo w słowo):
{data_work}

Dopasowana kampania marketingowa (streść w sekcji "Rekomendowana kampania retencyjna"):
{from_summarizer}"""

AGENT_RAG_SYSTEM = """Jesteś doradcą działu handlowego. Masz wyszukać w bazie kampanii \
marketingowych kampanii odpowiednich do zapobiegania odejściu klienta na podstawie jego cech ryzyka. \
Korzystaj TYLKO z narzędzi (użyj marketing_search raz).

WAŻNE – jako zapytanie (query) do marketing_search przekaż CAŁY otrzymany opis czynników ryzyka \
klienta, w całości i bez skracania. NIE streszczaj go do kilku słów ani do nazwy jednego produktu \
(np. "karta kredytowa") – krótkie zapytanie zwraca fragmenty jednej kampanii zamiast kilku różnych, \
przez co nie ma z czego wybierać."""

AGENT_RAG_HUMAN = """Na podstawie tej analizy ryzyka znajdź pasujące kampanie: {data_work}"""

CHOOSE_CAMPAIGN_TEMPLATE = """Jesteś doradcą ds. retencji klientów banku. Wybierz JEDNĄ kampanię marketingową, \
która najlepiej odpowiada na konkretne czynniki ryzyka odejścia tego klienta.

Profil ryzyka klienta:
{data_work}

Dostępne kampanie retencyjne:
{campaigns}

ZASADY WYBORU:
- Dopasowuj kampanię WYŁĄCZNIE do czynników ZWIĘKSZAJĄCYCH ryzyko odejścia (oznaczonych 🔴 i 🟡).
- Czynniki ochronne (🟢) to powody, dla których klient ZOSTAJE – NIGDY nie uzasadniaj nimi wyboru kampanii.
- Nie wybieraj kampanii tylko dlatego, że jej nazwa lub grupa docelowa przypomina jakąś cechę klienta \
(np. kraj zamieszkania). Liczy się to, czy oferta kampanii realnie odpowiada na przyczynę ryzyka odejścia.
- Jeśli kampania zakłada cechy klienta, których nie ma w jego profilu (np. duże oszczędności, określony wiek), \
traktuj to jako słabe dopasowanie.

Krótko przeanalizuj, który czynnik ryzyka klienta odpowiada ofercie każdej kampanii, \
a następnie wskaż najlepiej dopasowaną. Zakończ odpowiedź WYŁĄCZNIE tą linią, bez niczego po niej:
WYBÓR: <numer kampanii>"""
