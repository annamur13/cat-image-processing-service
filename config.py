import os
from dotenv import load_dotenv

load_dotenv()

CAT_API_KEY = os.getenv('CAT_API_KEY')
CAT_API_BASE_URL = os.getenv('CAT_API_BASE_URL', 'https://api.thecatapi.com/v1')

if CAT_API_KEY is None:
    print("Предупреждение: CAT_API_KEY не найден в .env файле!")