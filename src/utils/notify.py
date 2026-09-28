import os
import requests
from dotenv import load_dotenv


def send_telegram_message(title: str, message: str):
    """Send a notification via Telegram bot. Requires TELEGRAM_BOT_TOKEN and
    TELEGRAM_CHAT_ID in .env."""
    load_dotenv('.env')
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        print("[notify] Telegram credentials not set — skipping notification.")
        return None

    try:
        response = requests.post(
            f'https://api.telegram.org/bot{token}/sendMessage',
            data={'chat_id': chat_id, 'text': f'{title}\n{message}'},
            timeout=30,
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[notify] Failed to send Telegram message: {e}")
        return None
