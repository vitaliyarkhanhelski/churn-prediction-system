import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

# Ustaw w pliku .env lub zmiennych środowiskowych:
#   SENDER_EMAIL=twoj@gmail.com
#   SENDER_APP_PASSWORD=xxxx xxxx xxxx xxxx
SENDER_EMAIL        = os.getenv("SENDER_EMAIL", "")
SENDER_APP_PASSWORD = os.getenv("SENDER_APP_PASSWORD", "")


def send_emails(emails: list[dict], to_email: str) -> list[str]:
    """
    Wysyła listę e-maili przez Gmail SMTP.

    Parametry
    ----------
    emails : list[dict]
        Lista słowników z kluczami (format wyjścia LangGraph):
          - "subject" : str  – temat wiadomości
          - "body"    : str  – treść (HTML lub plain text)
    to_email : str
        Adres odbiorcy – e-mail managera wpisany w UI.

    Zwraca
    -------
    list[str]
        Lista błędów (pusta jeśli wszystkie e-maile wysłane pomyślnie).
    """
    if not SENDER_EMAIL or not SENDER_APP_PASSWORD:
        raise ValueError(
            "Brak konfiguracji nadawcy. Ustaw zmienne środowiskowe "
            "SENDER_EMAIL i SENDER_APP_PASSWORD."
        )

    errors = []

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)

        for email in emails:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = email["subject"]
                msg["From"]    = SENDER_EMAIL
                msg["To"]      = to_email

                body_type = "html" if "<" in email["body"] else "plain"
                msg.attach(MIMEText(email["body"], body_type, "utf-8"))

                server.send_message(msg)

            except Exception as e:
                errors.append(f"Błąd dla {to_email}: {e}")

    return errors


if __name__ == "__main__":
    # Zmień "to" na adres na który chcesz otrzymać testowego maila
    test_emails = [
        {
            "to": "vitaliyarkhanhelski@gmail.com",
            "subject": "Test – Churn Prediction System",
            "body": "<h3>Test wysyłki działa poprawnie.</h3>",
        }
    ]
    errs = send_emails(test_emails, to_email="vitaliyarkhanhelski@gmail.com")
    if errs:
        print("Błędy:", errs)
    else:
        print("✅ E-mail testowy wysłany pomyślnie.")
