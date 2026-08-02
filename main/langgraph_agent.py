"""
langgraph_agent.py – Agent AI generujący i wysyłający e-maile per klient.

Flow: explain_shap → prepare_mail → send_and_log
"""

import asyncio
import os
from datetime import datetime
from typing import TypedDict

import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

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

OLLAMA_MODEL = "SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M"

def _get_model(backend: str):
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)
    return init_chat_model(model="gpt-4o", temperature=0)



# ---------------------------------------------------------------------------
# Stan grafu
# ---------------------------------------------------------------------------

class StanGrafu(TypedDict):
    Json_shap:          dict   # {"feature_importance": {shap_col: value, ...}}
    customer_id:        str
    churn_probability:  float
    to_email:           str
    data_work:          str    # pośredni tekst analizy SHAP
    mail_body:          str    # gotowa treść e-maila
    subject:            str    # temat e-maila


# ---------------------------------------------------------------------------
# Węzły grafu
# ---------------------------------------------------------------------------

def _shap_to_text(raw: dict) -> str:
    """Zamienia wartości SHAP na czytelny tekst BEZ liczb – tylko kierunek i siła."""
    translated = {
        FEATURE_LABELS.get(k, k): v
        for k, v in raw.items()
    }
    sorted_features = sorted(translated.items(), key=lambda x: abs(x[1]), reverse=True)
    lines = []
    for name, value in sorted_features[:6]:
        if abs(value) < 0.03:
            continue
        if value > 0:
            if abs(value) > 0.5:
                icon, magnitude = "🔴", "bardzo duży wpływ"
            elif abs(value) > 0.2:
                icon, magnitude = "🔴", "duży wpływ"
            else:
                icon, magnitude = "🟡", "umiarkowany wpływ"
            direction = "zwiększa ryzyko odejścia"
        else:
            icon, magnitude = "🟢", "zmniejsza ryzyko odejścia"
            direction = "czynnik ochronny"
        lines.append(f"{icon} {name}: {magnitude} – {direction}")
    return "\n".join(lines)


def _make_nodes(model):
    """Zwraca węzły grafu zamknięte na dany model."""

    def explain_shap(stan: StanGrafu) -> dict:
        shap_text = _shap_to_text(stan["Json_shap"]["feature_importance"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXPLAIN_SHAP_SYSTEM),
            ("human",  EXPLAIN_SHAP_HUMAN),
        ])
        result = model.invoke(prompt.invoke({"dane_analiza": shap_text}))
        return {"data_work": result.content}

    def prepare_mail(stan: StanGrafu) -> dict:
        subject = (
            f"ALERT – Ryzyko odpływu klienta ID {stan['customer_id']} "
            f"({stan['churn_probability']:.1%})"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", PREPARE_MAIL_SYSTEM),
            ("human",  PREPARE_MAIL_HUMAN),
        ])
        result = model.invoke(prompt.invoke({
            "customer_id":       stan["customer_id"],
            "churn_probability": f"{stan['churn_probability']:.1%}",
            "data_work":         stan["data_work"],
        }))
        return {"mail_body": result.content, "subject": subject}

    return explain_shap, prepare_mail


def send_and_log(stan: StanGrafu) -> dict:
    from email_service import send_emails

    errors = send_emails(
        [{"subject": stan["subject"], "body": stan["mail_body"]}],
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
        if errors:
            f.write(f"BŁĘDY:        {errors}\n")
        f.write(f"\n--- SHAP (data_work) ---\n{stan['data_work']}\n")
        f.write(f"\n{'=' * 60}\n\n")
        f.write(stan["mail_body"])

    return {}


# ---------------------------------------------------------------------------
# Budowa grafu
# ---------------------------------------------------------------------------

def _build_graph(model):
    explain_shap, prepare_mail = _make_nodes(model)
    build = StateGraph(StanGrafu)
    build.add_node("explain_shap", explain_shap)
    build.add_node("prepare_mail", prepare_mail)
    build.add_node("send_and_log", send_and_log)
    build.add_edge(START,          "explain_shap")
    build.add_edge("explain_shap", "prepare_mail")
    build.add_edge("prepare_mail", "send_and_log")
    build.add_edge("send_and_log", END)
    return build.compile()


# ---------------------------------------------------------------------------
# API publiczne
# ---------------------------------------------------------------------------

async def _run_one(state: dict, graf, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        await graf.ainvoke(state)


def run_agent(df_flagged: pd.DataFrame, to_email: str, model_backend: str = "openai") -> list[str]:
    """
    Generuje i wysyła e-maile per klient. Wywołanie synchroniczne (kompatybilne ze Streamlit).

    Parametry
    ----------
    df_flagged     : DataFrame z wierszami marketing_action == 1
    to_email       : adres e-mail managera z UI
    model_backend  : "openai" (GPT-4o, async) lub "ollama" (Bielik, sekwencyjnie)

    Zwraca
    -------
    list[str] – lista błędów (pusta jeśli wszystko OK)
    """
    nest_asyncio.apply()

    model = _get_model(model_backend)
    graf  = _build_graph(model)

    # Ollama działa sekwencyjnie – Semaphore(1) zapobiega złudnym oczekiwaniom
    max_concurrent = 1 if model_backend == "ollama" else 10
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
                "data_work":         "",
                "mail_body":         "",
                "subject":           "",
            }
            tasks.append(_run_one(state, graf, semaphore))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [str(r) for r in results if isinstance(r, Exception)]

    return asyncio.get_event_loop().run_until_complete(_run_all())
