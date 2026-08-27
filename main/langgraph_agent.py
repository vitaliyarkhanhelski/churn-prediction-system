"""
langgraph_agent.py – Agent AI generujący i wysyłający e-maile per klient (Z RAG).
Flow: explain_shap -> agent_RAG -> (tools) -> summarize -> choose_campaign -> prepare_mail -> send_and_log
"""

import asyncio
import os
import re
import threading
import traceback
from datetime import datetime
from typing import TypedDict, Annotated
from pathlib import Path

import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from openai import RateLimitError

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from prompts import (
    AGENT_RAG_HUMAN,
    AGENT_RAG_SYSTEM,
    CHOOSE_CAMPAIGN_TEMPLATE,
    EXPLAIN_SHAP_HUMAN,
    EXPLAIN_SHAP_SYSTEM,
    FEATURE_LABELS,
    FEATURE_QUARTILES,
    FEATURE_VALUE_FORMATTERS,
    ONE_HOT_LABELS,
    PRODUCTS_NUMBER_CONTEXT,
    PREPARE_MAIL_HUMAN,
    PREPARE_MAIL_SYSTEM,
    SUMMARIZE_CHUNK_TEMPLATE,
    SUMMARIZE_FINAL_TEMPLATE,
)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

EMAILS_DIR = os.path.join(os.path.dirname(__file__), "emails")
os.makedirs(EMAILS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Postęp wysyłki – w pamięci procesu, do podglądu na żywo w UI (Streamlit odpytuje
# get_progress() z fragmentu strony; wątek w tle nie może bezpiecznie pisać do
# st.session_state, więc trzymamy to poza nim).
# ---------------------------------------------------------------------------
_progress_lock = threading.Lock()
_progress_store: dict[str, list[dict]] = {}


def _push_progress(run_id: str, customer_id: str, log_file: str, status: str) -> None:
    if not run_id:
        return
    with _progress_lock:
        _progress_store.setdefault(run_id, []).append({
            "customer_id": customer_id,
            "log_file":    log_file,
            "status":      status,
        })


def get_progress(run_id: str) -> list[dict]:
    with _progress_lock:
        return list(_progress_store.get(run_id, []))


# ---------------------------------------------------------------------------
# Cache streszczeń kampanii – w pamięci procesu, per plik. Bez tego każdy
# klient re-streszcza od zera te same ~10 statycznych dokumentów kampanii
# (kilka wywołań LLM na plik), co przy kilku klientach naraz szybko wybija
# limit tokenów/min (TPM) w OpenAI. Blokada per-plik zapobiega temu, żeby
# dwóch klientów naraz streszczało ten sam, jeszcze nie zcache'owany plik.
# ---------------------------------------------------------------------------
_summary_cache_lock = threading.Lock()
_summary_cache: dict[str, str] = {}
_summary_file_locks: dict[str, threading.Lock] = {}


def _get_file_lock(file_name: str) -> threading.Lock:
    with _summary_cache_lock:
        return _summary_file_locks.setdefault(file_name, threading.Lock())

OLLAMA_MODEL = "SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M"

# Model OpenAI używany w całym pipeline (także wymuszony dla RAG i wyboru kampanii).
# Testowano "gpt-4o-mini" (tańszy, wyższe limity TPM) – odpadł: w choose_campaign
# potrafił odrzucić kampanię jako niedopasowaną, a zdanie dalej wybrać ją jako najlepszą
# (patrz emails/2026-08-26_14-23-21_15624512.txt). Porównywanie kampanii to jedyny krok
# wymagający realnego osądu, więc zostaje gpt-4o.
OPENAI_MODEL = "gpt-4o"

def _get_model(backend: str, temperature: float = 0):
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        # num_ctx=8192 – Ollama domyślnie tnie okno kontekstowe do 2048 tokenów i robi to
        # PO CICHU (bez błędu), więc przy większych fragmentach model dostawałby sam początek
        # tekstu. 8192 to okno Bielika 11B v2.3.
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature, num_ctx=8192)
    return init_chat_model(model=OPENAI_MODEL, temperature=temperature)


def _retryable(runnable):
    """Automatyczny retry z exponential backoff na 429 (rate limit) z OpenAI. Bez tego
    jeden 429 zabija całą wysyłkę dla klienta (patrz emails/*_FAILED.txt) zamiast poczekać
    parę sekund i spróbować ponownie. Uwaga: with_retry() gubi bind_tools(), więc trzeba
    to nakładać PO bind_tools()/zbudowaniu chaina, nigdy na surowym modelu z RAG."""
    return runnable.with_retry(
        retry_if_exception_type=(RateLimitError,),
        wait_exponential_jitter=True,
        stop_after_attempt=5,
    )


# ---------------------------------------------------------------------------
# Konfiguracja bazy RAG (Chroma)
# ---------------------------------------------------------------------------
# Inicjujemy globalnie, żeby nie ładować bazy na nowo dla każdego e-maila
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
marketing_vector_store = Chroma(
    persist_directory="./Marketing_cam_test",
    embedding_function=embeddings,
    collection_name="Marketing_cam"
)
MAX_CAMPAIGN_CANDIDATES = 4

# k=40: nawet bardzo wąskie zapytanie (np. "karta kredytowa") trafia w ~22 chunki tej samej
# kampanii, więc potrzeba zapasu, żeby po deduplikacji zostały 4 różne kampanie.
marketing_retriever = marketing_vector_store.as_retriever(search_kwargs={"k": 40})

@tool(response_format="content_and_artifact")
def marketing_search(query: str) -> tuple[str, list]:
    """Baza kampanii marketingowych do wykorzystania w celu zapobiegania odejściu klienta"""
    docs = marketing_retriever.invoke(query)
    # Uwaga: k liczy CHUNKI, nie dokumenty – jeden dokument kampanii ma ~20 chunków,
    # więc wąskie zapytanie (np. "karta kredytowa") potrafiło zwrócić wszystkie chunki
    # z jednej kampanii i agent nie miał z czego wybierać. Pobieramy więcej chunków,
    # deduplikujemy po źródle i zwracamy pierwsze RÓŻNE kampanie (najtrafniejszy chunk
    # z każdej), żeby liczba kandydatów była stabilna niezależnie od treści zapytania.
    best_per_source = {}
    for doc in docs:
        src = doc.metadata.get("source")
        if src and src not in best_per_source:
            best_per_source[src] = doc
        if len(best_per_source) == MAX_CAMPAIGN_CANDIDATES:
            break
    docs = list(best_per_source.values())
    content = "\n\n".join(f"Source:{doc.metadata}\n{doc.page_content}" for doc in docs)
    return content, docs

tools_list = [marketing_search]


# ---------------------------------------------------------------------------
# Stan grafu
# ---------------------------------------------------------------------------
class StanGrafu(TypedDict):
    Json_shap:          dict
    customer_id:        str
    churn_probability:  float
    to_email:           str
    run_id:             str
    data_work:          str
    mail_body:          str
    subject:            str
    
    # Zmienne RAG
    messages:                 Annotated[list[AnyMessage], add_messages]
    from_summarizer:          str
    campaigns:                 list[dict]
    chosen_campaign_source:    str
    chosen_campaign_summary:   str
    campaign_choice_reasoning: str
    attachments:               list[str]


# ---------------------------------------------------------------------------
# Logika pomocnicza
# ---------------------------------------------------------------------------
def _value_context(shap_key: str, value: float) -> str:
    """Krótki opis, czy wartość cechy jest typowa na tle innych klientów. Model zna samą
    liczbę, ale nie wie, czy 502 pkt to dużo czy mało – bez tego zgadywał (i mylił się)."""
    if shap_key == "shap_products_number":
        return PRODUCTS_NUMBER_CONTEXT.get(int(value), "")
    # Zerowe saldo to osobny przypadek (36% klientów), a nie "wartość typowa" –
    # mediana 97 199 sugerowałaby, że puste konto jest czymś normalnym.
    if shap_key == "shap_balance" and value == 0:
        return "konto bez środków, dotyczy 36% klientów"
    if (quartiles := FEATURE_QUARTILES.get(shap_key)):
        q25, median, q75 = quartiles
        if value < q25:
            return f"niska wartość, mediana to {median:,.0f}"
        if value > q75:
            return f"wysoka wartość, mediana to {median:,.0f}"
        return f"wartość typowa (mediana {median:,.0f})"
    return ""


def _shap_to_text(raw: dict, features: dict | None = None) -> str:
    """Zamienia wartości SHAP na opis czynników ryzyka.

    `features` to surowe dane klienta (country, gender, products_number, ...). Są potrzebne
    z dwóch powodów:
      1. Cechy one-hot (country_*, gender_*) opisują CECHĘ, nie fakt o kliencie – dla klienta
         z Francji kolumna shap_country_Germany dotyczy tego, że klient NIE jest z Niemiec.
         Bez tego system pisał "kraj zamieszkania: Niemcy" Francuzowi i podawał obie płcie naraz.
      2. Cechy liczbowe bez wartości zmuszały model do zgadywania kierunku (np. "wiele" vs
         "niewiele produktów" dla tego samego klienta w różnych uruchomieniach).
    """
    features = features or {}

    # Etykiety cech one-hot, które faktycznie opisują tego klienta (np. country == "France").
    active_one_hot = {}
    for (col, val), label in ONE_HOT_LABELS.items():
        if str(features.get(col, "")) == val:
            active_one_hot[f"shap_{col}_{val}"] = label

    scored = []
    for key, shap_value in raw.items():
        if key in FEATURE_LABELS:
            label = FEATURE_LABELS[key]
            col = key.removeprefix("shap_")
            if col in features and (fmt := FEATURE_VALUE_FORMATTERS.get(key)):
                try:
                    raw_value = float(features[col])
                    parts = [fmt.format(value=raw_value)]
                    if (context := _value_context(key, raw_value)):
                        parts.append(context)
                    label = f"{label} ({' – '.join(parts)})"
                except (TypeError, ValueError):
                    pass
            elif col == "active_member" and col in features:
                label = f"{label} ({'aktywny' if float(features[col]) == 1 else 'nieaktywny'})"
        elif key in active_one_hot:
            label = active_one_hot[key]
        else:
            # Cecha one-hot nieopisująca tego klienta – pominięta, żeby nie tworzyć fałszywych
            # stwierdzeń typu "kraj zamieszkania: Niemcy" dla klienta z Francji.
            continue
        scored.append((label, shap_value))

    lines = []
    for name, value in sorted(scored, key=lambda x: abs(x[1]), reverse=True)[:6]:
        if abs(value) < 0.03: continue
        if value > 0:
            icon, magnitude = ("🔴", "bardzo duży wpływ") if abs(value) > 0.5 else ("🔴", "duży wpływ") if abs(value) > 0.2 else ("🟡", "umiarkowany wpływ")
            direction = "zwiększa ryzyko odejścia"
        else:
            icon, magnitude, direction = "🟢", "zmniejsza ryzyko odejścia", "czynnik ochronny"
        lines.append(f"{icon} {name}: {magnitude} – {direction}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Węzły grafu
# ---------------------------------------------------------------------------
def _make_nodes(model_backend):
    # model_low nigdy nie jest używany z bind_tools(), więc retry można nałożyć od razu.
    model_low = _retryable(_get_model(model_backend, temperature=0))

    # Zwróć uwagę: RAG i streszczanie dużo lepiej działają na modelu OpenAI ze wsparciem Tool Calling
    # Dlatego wymuszamy model OpenAI (OPENAI_MODEL) dla operacji RAG, by uniknąć halucynacji lokalnego modelu.
    # Surowy, bez retry – patrz _retryable(): bind_tools() musi być nałożony PRZED retry.
    model_for_rag = _get_model("openai", temperature=0)

    def explain_shap(stan: StanGrafu) -> dict:
        shap_text = _shap_to_text(
            stan["Json_shap"]["feature_importance"],
            stan["Json_shap"].get("features"),
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXPLAIN_SHAP_SYSTEM),
            ("human",  EXPLAIN_SHAP_HUMAN),
        ])
        result = model_low.invoke(prompt.invoke({"dane_analiza": shap_text}))
        return {"data_work": result.content}

    def agent_RAG(stan: StanGrafu) -> dict:
        llm = _retryable(model_for_rag.bind_tools(tools_list))
        if not stan.get("messages"):
            mese = [
                SystemMessage(content=AGENT_RAG_SYSTEM),
                HumanMessage(content=AGENT_RAG_HUMAN.format(data_work=stan["data_work"])),
            ]
        else:
            mese = stan["messages"]
        response = llm.invoke(mese)
        return {"messages": [response]}

    def route_after_rag(stan: StanGrafu) -> str:
        last_message = stan["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "Summarize"

    def summarize(stan: StanGrafu) -> dict:
        # Lista (nie set) – zachowujemy kolejność zwróconą przez wyszukiwarkę wektorową
        # (najbardziej trafne wyniki pierwsze), potrzebną w choose_campaign do numeracji.
        source_list = []
        for m in stan["messages"]:
            if m.type == "tool" and hasattr(m, 'artifact') and m.artifact:
                for doc in m.artifact:
                    source = doc.metadata.get("source")
                    if source and source not in source_list:
                        source_list.append(source)

        campaigns = []
        text_to_body = ""
        for file_name in source_list:
            title = Path(file_name).stem.replace("_", " ")

            with _summary_cache_lock:
                summary = _summary_cache.get(file_name)

            if summary is None:
                # Blokada per-plik: jeśli dwóch klientów trafi na ten sam, jeszcze nie
                # zcache'owany plik w tym samym momencie, drugi czeka i dostaje wynik
                # z cache'u zamiast też odpytywać LLM od zera.
                with _get_file_lock(file_name):
                    with _summary_cache_lock:
                        summary = _summary_cache.get(file_name)
                    if summary is None:
                        try:
                            loader = Docx2txtLoader(f"{file_name}")
                            chunk_text = ""
                            doc = loader.load()
                            # chunk_size=12000 znaków (~4 200 tokenów): każdy z 10 dokumentów
                            # kampanii dzieli się równo na 2 kawałki → 3 wywołania LLM zamiast 21.
                            # Zostaje ~4 000 tokenów zapasu w oknie Bielika (8192) na odpowiedź.
                            # overlap=200 – bez niego drugi kawałek zaczyna się w środku
                            # wypunktowania, bez nagłówka mówiącego, czego lista dotyczy.
                            splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=200)
                            chunks = splitter.split_documents(doc)

                            # Testowano streszczanie na Bieliku (model_low) – treść była poprawna,
                            # ale format nie: prefiks "Streszczenie tekstu:" mimo zakazu w prompcie
                            # (4/4 kampanie), lista numerowana zamiast prozy, a przy dokumencie
                            # dzielonym na 2 kawałki etap scalający zwrócił DWA streszczenia obok
                            # siebie zamiast jednego. Zostaje OpenAI.
                            for chunk in chunks:
                                prompt = ChatPromptTemplate.from_template(SUMMARIZE_CHUNK_TEMPLATE)
                                chain = _retryable(prompt | model_for_rag | StrOutputParser())
                                chunk_text += chain.invoke({"chunk": chunk.page_content})

                            prompt_sum = ChatPromptTemplate.from_template(SUMMARIZE_FINAL_TEMPLATE)
                            chain_sum = _retryable(prompt_sum | model_for_rag | StrOutputParser())
                            summary = chain_sum.invoke({"tekst": chunk_text})

                            with _summary_cache_lock:
                                _summary_cache[file_name] = summary
                        except Exception as e:
                            print(f"Błąd przetwarzania pliku {file_name}: {e}")
                            summary = None

            if summary:
                text_to_body += f"🔹 **Kampania: {title}**\n{summary}\n\n"
                campaigns.append({"source": file_name, "title": title, "summary": summary})

        if not text_to_body:
            text_to_body = "Brak specyficznej kampanii w bazie. Zaproponuj standardowe kroki retencyjne."

        return {
            "from_summarizer": text_to_body.strip(),
            "campaigns": campaigns,
        }

    def choose_campaign(stan: StanGrafu) -> dict:
        """Wybiera JEDNĄ kampanię spośród kandydatów znalezionych przez RAG – zawsze na
        OpenAI (model_for_rag), niezależnie od wybranego backendu, bo to wymaga realnego
        porównania kandydatów z profilem ryzyka klienta, a Bielik zawodził przy tym zadaniu
        (np. wybierał kampanię dla Hiszpanii dla klienta z Niemiec)."""
        campaigns = stan.get("campaigns", [])

        if not campaigns:
            return {
                "chosen_campaign_source": "",
                "chosen_campaign_summary": "Brak specyficznej kampanii w bazie. Zaproponuj standardowe kroki retencyjne.",
                "campaign_choice_reasoning": "",
            }

        if len(campaigns) == 1:
            c = campaigns[0]
            return {
                "chosen_campaign_source": c["source"],
                "chosen_campaign_summary": f"🔹 **Kampania: {c['title']}**\n{c['summary']}",
                "campaign_choice_reasoning": "Tylko jedna kampania znaleziona przez RAG – wybór automatyczny, bez porównania.",
            }

        numbered = "\n\n".join(f"{i + 1}. {c['title']}\n{c['summary']}" for i, c in enumerate(campaigns))
        prompt = ChatPromptTemplate.from_template(CHOOSE_CAMPAIGN_TEMPLATE)
        chain = _retryable(prompt | model_for_rag | StrOutputParser())
        raw = chain.invoke({"data_work": stan["data_work"], "campaigns": numbered})

        match = re.search(r"WYBÓR:\s*(\d+)", raw)
        if match:
            idx = int(match.group(1)) - 1
        else:
            all_digits = re.findall(r"\d+", raw)
            idx = int(all_digits[-1]) - 1 if all_digits else 0
        if not (0 <= idx < len(campaigns)):
            idx = 0

        chosen = campaigns[idx]
        return {
            "chosen_campaign_source": chosen["source"],
            "chosen_campaign_summary": f"🔹 **Kampania: {chosen['title']}**\n{chosen['summary']}",
            "campaign_choice_reasoning": raw.strip(),
        }

    def prepare_mail(stan: StanGrafu) -> dict:
        subject = f"ALERT – Ryzyko odpływu klienta ID {stan['customer_id']} ({stan['churn_probability']:.1%})"

        prompt = ChatPromptTemplate.from_messages([
            ("system", PREPARE_MAIL_SYSTEM),
            ("human",  PREPARE_MAIL_HUMAN),
        ])

        result = model_low.invoke(prompt.invoke({
            "customer_id":       stan["customer_id"],
            "churn_probability": f"{stan['churn_probability']:.1%}",
            "data_work":         stan["data_work"],
            "chosen_campaign":   stan["chosen_campaign_summary"]
        }))

        # Kampania wybrana jest deterministycznie w choose_campaign (przed tym węzłem),
        # więc załącznik budujemy wprost ze znanego źródła – bez zgadywania po treści maila.
        pdf_attachments = []
        chosen_source = stan.get("chosen_campaign_source")
        if chosen_source:
            pdf_path = os.path.join(os.path.dirname(__file__), "rag", "Files_to_attach", f"{Path(chosen_source).stem}.pdf")
            pdf_attachments.append(pdf_path)

        return {
            "mail_body": result.content,
            "subject": subject,
            "attachments": pdf_attachments # Przekazujemy ścieżki do wysyłki
        }

    return explain_shap, agent_RAG, route_after_rag, summarize, choose_campaign, prepare_mail


def send_and_log(stan: StanGrafu) -> dict:
    from email_service import send_emails

    errors = send_emails(
        [{
            "subject": stan["subject"], 
            "body": stan["mail_body"],
            "attachments": stan.get("attachments", [])
        }],
        to_email=stan["to_email"],
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(EMAILS_DIR, f"{timestamp}_{stan['customer_id']}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Customer ID:  {stan['customer_id']}\n")
        f.write(f"Ryzyko:       {stan['churn_probability']:.1%}\n")
        f.write(f"Do:           {stan['to_email']}\n")
        f.write(f"Temat:        {stan['subject']}\n")
        f.write(f"Data:         {timestamp}\n")
        f.write(f"\n--- DELIVERY STATUS ---\n")
        f.write(f"❌ Błąd wysyłki: {errors}\n" if errors else "✅ Wysłano pomyślnie\n")
        f.write(f"\n--- SHAP (data_work) ---\n{stan['data_work']}\n")
        f.write(f"\n--- ZNALEZIONA KAMPANIA ---\n{stan['from_summarizer']}\n")
        if stan.get("campaign_choice_reasoning"):
            f.write(f"\n--- UZASADNIENIE WYBORU KAMPANII ---\n{stan['campaign_choice_reasoning']}\n")
        f.write(f"\n{'=' * 60}\n\n")
        f.write(stan["mail_body"])

    _push_progress(
        stan.get("run_id", ""),
        stan["customer_id"],
        os.path.basename(log_path),
        "❌ Błąd" if errors else "✅ Sukces",
    )

    return {}


# ---------------------------------------------------------------------------
# Budowa grafu
# ---------------------------------------------------------------------------
def _build_graph(model_backend: str):
    explain_shap, agent_RAG, route_after_rag, summarize, choose_campaign, prepare_mail = _make_nodes(model_backend)

    build = StateGraph(StanGrafu)
    build.add_node("explain_shap", explain_shap)
    build.add_node("Agent_RAG", agent_RAG)
    build.add_node("tools", ToolNode(tools_list))
    build.add_node("Summarize", summarize)
    build.add_node("ChooseCampaign", choose_campaign)
    build.add_node("prepare_mail", prepare_mail)
    build.add_node("send_and_log", send_and_log)

    build.add_edge(START, "explain_shap")
    build.add_edge("explain_shap", "Agent_RAG")
    build.add_conditional_edges("Agent_RAG", route_after_rag, {"tools": "tools", "Summarize": "Summarize"})
    build.add_edge("tools", "Agent_RAG")
    build.add_edge("Summarize", "ChooseCampaign")
    build.add_edge("ChooseCampaign", "prepare_mail")
    build.add_edge("prepare_mail", "send_and_log")
    build.add_edge("send_and_log", END)
    
    return build.compile()


# ---------------------------------------------------------------------------
# API publiczne
# ---------------------------------------------------------------------------
def _log_failure(state: dict, exc: BaseException) -> None:
    """Zapisuje log błędu, żeby nieudana wysyłka nie zniknęła bez śladu (asyncio.gather
    z return_exceptions=True + wywołanie w tle w app.py inaczej wyciszają wszystko)."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    customer_id = state.get("customer_id", "unknown")
    log_path = os.path.join(EMAILS_DIR, f"{timestamp}_{customer_id}_FAILED.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Customer ID:  {customer_id}\n")
        f.write(f"Data:         {timestamp}\n")
        f.write("\n--- BŁĄD PRZETWARZANIA (e-mail NIE został wysłany) ---\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    _push_progress(state.get("run_id", ""), customer_id, os.path.basename(log_path), "❌ Błąd (wyjątek)")


async def _run_one(state: dict, graf, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        try:
            await graf.ainvoke(state)
        except Exception as e:
            _log_failure(state, e)
            raise


def run_agent(df_flagged: pd.DataFrame, to_email: str, model_backend: str = "openai", run_id: str = "") -> list[str]:
    graf = _build_graph(model_backend)

    # 2 dla OpenAI – każdy klient robi kilka sekwencyjnych wywołań LLM (explain_shap,
    # RAG, streszczenia kampanii, prepare_mail), więc kilku klientów naraz szybko
    # sumuje się do limitu tokenów/min (TPM) w OpenAI i kończy się 429 (patrz
    # emails/*_FAILED.txt). Retry z backoffem to łata na wypadek przekroczenia, a to
    # ogranicza, jak często w ogóle do niego dochodzi.
    max_concurrent = 1 if model_backend == "ollama" else 2
    shap_cols = [c for c in df_flagged.columns if c.startswith("shap_")]
    # Surowe cechy klienta – potrzebne, żeby opis SHAP podawał faktyczne wartości
    # i poprawnie obsługiwał cechy one-hot (patrz _shap_to_text).
    _meta_cols = {"customer_id", "account_manager_email", "churn_probability", "marketing_action"}
    feature_cols = [c for c in df_flagged.columns if not c.startswith("shap_") and c not in _meta_cols]

    async def _run_all() -> list[str]:
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        for _, row in df_flagged.iterrows():
            feature_importance = {col: float(row[col]) for col in shap_cols}
            features = {col: row[col] for col in feature_cols}
            state: StanGrafu = {
                "Json_shap":         {"feature_importance": feature_importance, "features": features},
                "customer_id":       str(int(row["customer_id"])),
                "churn_probability": float(row["churn_probability"]),
                "to_email":          to_email,
                "run_id":            run_id,
                "data_work":         "",
                "mail_body":         "",
                "subject":           "",
                "messages":          [],
                "from_summarizer":          "",
                "campaigns":                [],
                "chosen_campaign_source":   "",
                "chosen_campaign_summary":  "",
                "campaign_choice_reasoning": "",
                "attachments":              [],
            }
            tasks.append(_run_one(state, graf, semaphore))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [str(r) for r in results if isinstance(r, Exception)]

    try:
        loop = asyncio.get_running_loop()
        nest_asyncio.apply()
        return loop.run_until_complete(_run_all())
    except RuntimeError:
        return asyncio.run(_run_all())