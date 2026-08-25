"""
langgraph_agent.py – Agent AI generujący i wysyłający e-maile per klient (Z RAG).
Flow: explain_shap -> agent_RAG -> (tools) -> summarize -> prepare_mail -> send_and_log
"""

import asyncio
import os
import re
import threading
import traceback
import unicodedata
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
    EXPLAIN_SHAP_HUMAN,
    EXPLAIN_SHAP_SYSTEM,
    FEATURE_LABELS,
    PREPARE_MAIL_HUMAN,
    PREPARE_MAIL_SYSTEM,
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

def _get_model(backend: str, temperature: float = 0):
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)
    return init_chat_model(model="gpt-4o", temperature=temperature)


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
marketing_retriever = marketing_vector_store.as_retriever(search_kwargs={"k": 4})

@tool(response_format="content_and_artifact")
def marketing_search(query: str) -> tuple[str, list]:
    """Baza kampanii marketingowych do wykorzystania w celu zapobiegania odejściu klienta"""
    docs = marketing_retriever.invoke(query)
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
    messages:           Annotated[list[AnyMessage], add_messages]
    camp_source:        set
    from_summarizer:    str
    attachments:        list[str]


# ---------------------------------------------------------------------------
# Logika pomocnicza
# ---------------------------------------------------------------------------
def _shap_to_text(raw: dict) -> str:
    translated = {FEATURE_LABELS.get(k, k): v for k, v in raw.items()}
    sorted_features = sorted(translated.items(), key=lambda x: abs(x[1]), reverse=True)
    lines = []
    for name, value in sorted_features[:6]:
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
    # Dlatego wymuszamy gpt-4o-mini dla operacji RAG, by uniknąć halucynacji lokalnego modelu.
    # Surowy, bez retry – patrz _retryable(): bind_tools() musi być nałożony PRZED retry.
    model_for_rag = _get_model("openai", temperature=0)

    def explain_shap(stan: StanGrafu) -> dict:
        shap_text = _shap_to_text(stan["Json_shap"]["feature_importance"])
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
                SystemMessage(content="Jesteś doradcą działu handlowego. Masz wyszukać w bazie kampanii marketingowych odpowiedniej kampanii do zapobiegania odejściu klienta na podstawie jego cech ryzyka. Korzystaj TYLKO z narzędzi (użyj marketing_search raz)."),
                HumanMessage(content=f"Na podstawie tej analizy ryzyka znajdź pasującą kampanię: {stan['data_work']}")
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
        source_set = set()
        for m in stan["messages"]:
            if m.type == "tool" and hasattr(m, 'artifact') and m.artifact:
                for doc in m.artifact:
                    source = doc.metadata.get("source")
                    if source:
                        source_set.add(source)
                        
        text_to_body = ""
        for file_name in source_set:
            with _summary_cache_lock:
                cached = _summary_cache.get(file_name)
            if cached is not None:
                text_to_body += cached
                continue

            # Blokada per-plik: jeśli dwóch klientów trafi na ten sam, jeszcze nie
            # zcache'owany plik w tym samym momencie, drugi czeka i dostaje wynik
            # z cache'u zamiast też odpytywać LLM od zera.
            with _get_file_lock(file_name):
                with _summary_cache_lock:
                    cached = _summary_cache.get(file_name)
                if cached is not None:
                    text_to_body += cached
                    continue

                try:
                    loader = Docx2txtLoader(f"{file_name}")
                    title = Path(file_name).stem.replace("_", " ")
                    chunk_text = ""
                    doc = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                    chunks = splitter.split_documents(doc)

                    for chunk in chunks:
                        prompt = ChatPromptTemplate.from_template("Streść tekst kampanii, cel i założenia:\n{chunk}")
                        chain = _retryable(prompt | model_for_rag | StrOutputParser())
                        chunk_text += chain.invoke({"chunk": chunk.page_content})

                    prompt_sum = ChatPromptTemplate.from_template("Streść maksymalnie w 5 zdaniach podany tekst, skup się na celu kampanii dla opiekuna klienta:\n{tekst}")
                    chain_sum = _retryable(prompt_sum | model_for_rag | StrOutputParser())
                    response_sum = chain_sum.invoke({"tekst": chunk_text})
                    entry = f"🔹 **Kampania: {title}**\n{response_sum}\n\n"

                    with _summary_cache_lock:
                        _summary_cache[file_name] = entry
                    text_to_body += entry
                except Exception as e:
                    print(f"Błąd przetwarzania pliku {file_name}: {e}")

        if not text_to_body:
            text_to_body = "Brak specyficznej kampanii w bazie. Zaproponuj standardowe kroki retencyjne."

        return {"camp_source": source_set, "from_summarizer": text_to_body.strip()}

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
            "from_summarizer":   stan["from_summarizer"]
        }))
        
        # RAG zwraca kilku kandydatów (camp_source), ale mail_body opisuje tylko JEDNĄ,
        # wybraną przez LLM kampanię (patrz PREPARE_MAIL_SYSTEM – "Rekomendowana kampania
        # retencyjna"). Załączamy PDF tylko tej kampanii, której tytuł faktycznie pojawił
        # się w treści e-maila, a nie wszystkich kandydatów zwróconych przez wyszukiwarkę.
        # Nazwy plików z polskimi znakami bywają zapisane na dysku (macOS) w formie NFD
        # (np. "ś" jako "s" + osobny znak akcentu), a tekst z LLM przychodzi w formie NFC
        # – wizualnie identyczne, ale różne pod względem code pointów, więc `in` zawodzi
        # bez normalizacji obu stron do tej samej formy.
        mail_text_norm = unicodedata.normalize("NFC", result.content)
        pdf_attachments = []
        for k in stan.get("camp_source", []):
            title = unicodedata.normalize("NFC", Path(k).stem.replace("_", " "))
            # Model czasem pomija powtórzone słowo "Kampania" (np. tytuł "Kampania Smart
            # Credit" -> pisze tylko "Smart Credit"), więc sprawdzamy też wariant bez niego.
            title_core = re.sub(r"(?i)^kampania\s+", "", title).strip()
            if title in mail_text_norm or (title_core and title_core in mail_text_norm):
                pdf_path = os.path.join(os.path.dirname(__file__), "rag", "Files_to_attach", f"{Path(k).stem}.pdf")
                pdf_attachments.append(pdf_path)

        return {
            "mail_body": result.content, 
            "subject": subject,
            "attachments": pdf_attachments # Przekazujemy ścieżki do wysyłki
        }

    return explain_shap, agent_RAG, route_after_rag, summarize, prepare_mail


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
    explain_shap, agent_RAG, route_after_rag, summarize, prepare_mail = _make_nodes(model_backend)
    
    build = StateGraph(StanGrafu)
    build.add_node("explain_shap", explain_shap)
    build.add_node("Agent_RAG", agent_RAG)
    build.add_node("tools", ToolNode(tools_list))
    build.add_node("Summarize", summarize)
    build.add_node("prepare_mail", prepare_mail)
    build.add_node("send_and_log", send_and_log)
    
    build.add_edge(START, "explain_shap")
    build.add_edge("explain_shap", "Agent_RAG")
    build.add_conditional_edges("Agent_RAG", route_after_rag, {"tools": "tools", "Summarize": "Summarize"})
    build.add_edge("tools", "Agent_RAG")
    build.add_edge("Summarize", "prepare_mail")
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
    # sumuje się do limitu 30k tokenów/min (TPM) na gpt-4o i kończy się 429 (patrz
    # emails/*_FAILED.txt). Retry z backoffem to łata na wypadek przekroczenia, a to
    # ogranicza, jak często w ogóle do niego dochodzi.
    max_concurrent = 1 if model_backend == "ollama" else 2
    shap_cols = [c for c in df_flagged.columns if c.startswith("shap_")]

    async def _run_all() -> list[str]:
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        for _, row in df_flagged.iterrows():
            feature_importance = {col: float(row[col]) for col in shap_cols}
            state: StanGrafu = {
                "Json_shap":         {"feature_importance": feature_importance},
                "customer_id":       str(int(row["customer_id"])),
                "churn_probability": float(row["churn_probability"]),
                "to_email":          to_email,
                "run_id":            run_id,
                "data_work":         "",
                "mail_body":         "",
                "subject":           "",
                "messages":          [],
                "camp_source":       set(),
                "from_summarizer":   "",
                "attachments":       [],
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