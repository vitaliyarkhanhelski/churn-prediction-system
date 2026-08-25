import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

SENDER_EMAIL        = os.getenv("SENDER_EMAIL", "")
SENDER_APP_PASSWORD = os.getenv("SENDER_APP_PASSWORD", "")


def send_emails(emails: list[dict], to_email: str) -> list[str]:
    """
    Wysyła listę e-maili przez Gmail SMTP wraz z opcjonalnymi załącznikami PDF.

    Parametry
    ----------
    emails : list[dict]
        Lista słowników:
          - "subject"     : str
          - "body"        : str
          - "attachments" : list[str] (opcjonalnie: ścieżki do plików PDF)
    to_email : str
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
                # Zmiana z "alternative" na "mixed", żeby móc bezpiecznie dołączać pliki
                msg = MIMEMultipart("mixed")
                msg["Subject"] = email["subject"]
                msg["From"]    = SENDER_EMAIL
                msg["To"]      = to_email

                # Kontener na treść (HTML / Plain text)
                msg_body = MIMEMultipart("alternative")
                body_type = "html" if "<" in email["body"] else "plain"
                msg_body.attach(MIMEText(email["body"], body_type, "utf-8"))
                msg.attach(msg_body)

                # Obsługa załączników PDF
                attachments = email.get("attachments", [])
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
                        # add_header koduje nazwę pliku wg RFC 2231 (obsługa polskich znaków) –
                        # ręczne przypisanie part['Content-Disposition'] = f'...' koduje CAŁY
                        # nagłówek jako jeden blob RFC 2047, co część klientów odrzuca jako
                        # nieprawidłowy i pomija załącznik.
                        part.add_header("Content-Disposition", "attachment", filename=os.path.basename(file_path))
                        msg.attach(part)
                    else:
                        print(f"⚠️ Ostrzeżenie: Nie znaleziono pliku do załączenia: {file_path}")

                server.send_message(msg)

            except Exception as e:
                errors.append(f"Błąd dla {to_email}: {e}")

    return errors