import logging
from dotenv import load_dotenv
import os
import json
import requests
import pandas as pd
load_dotenv()
API_TOKEN = os.getenv("KINOPOISK_API_KEY")
films = pd.read_csv("/home/umbilnm/RecSysPlayground/data/interim/5k_selected_items.csv")
base_url = "https://api.kinopoisk.dev/v1.4/movie/search"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


def get_info_from_api(movie_id: int):
    logging.info(f"Запрос информации по фильму с ID: {movie_id}")
    title, title_orig, release_year = films[films['item_id'] == movie_id].loc[:, ["title", "title_orig", "release_year"]].values[0]

    headers = {
        "X-API-KEY": f"{API_TOKEN}"
    }

    params = {
        "query": title
    }

    try:
        logging.debug(f"Отправка GET запроса к API: {base_url} с параметрами: {params}")
        
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            response = response.json()
            if 'docs' not in response:
                logging.error(f"Некорректный ответ от API для movie_id: {movie_id}. Отсутствует ключ 'docs'.")
                return None, response
            
            response_df = pd.DataFrame(response['docs'])
            if title_orig is not None:
                item = response_df[(response_df['name'] == title) & 
                    (response_df['alternativeName'] == title_orig) & 
                    (response_df['year'] == release_year)]
            else:
                item = response_df[(response_df['name'] == title) & 
                    (response_df['year'] == release_year)]

            if len(item) == 0:
                logging.warning(f"Фильм не найден по указанным критериям: {title}, {title_orig}, {release_year}")
                return None, response 
            else:
                logging.info(f"Фильм найден: {item.iloc[0].to_dict()}")
                return item.iloc[0].to_dict(), response 
    except Exception as e:
        logging.error(f"Ошибка при запросе API для movie_id: {movie_id}. Ошибка: {e}")
        return None, None

def save_info_from_api(movie_id: int, info: dict, folder: str = "found_films"):
    path = f"/home/umbilnm/RecSysPlayground/data/external/{folder}/{movie_id}_info.json"
    if not os.path.exists(path):
        try:
            logging.info(f"Сохранение информации в файл: {path}")
            with open(path, "w") as f:
                json.dump(info, f)
        except Exception as e:
            logging.error(f"Ошибка при сохранении информации для movie_id: {movie_id}. Ошибка: {e}")
    else:
        logging.warning(f"Файл {path} уже существует. Сохранение отменено.")


ids = films['item_id'].values.tolist()
for cur_id in ids:
    found_path = f"/home/umbilnm/RecSysPlayground/data/external/found_films/{cur_id}_info.json"
    not_found_path = f"/home/umbilnm/RecSysPlayground/data/external/not_found/{cur_id}_info.json"
    
    if not os.path.exists(found_path) and not os.path.exists(not_found_path):
        info, response = get_info_from_api(cur_id)
        if info is not None:
            save_info_from_api(cur_id, info, folder="found_films")
        elif response is not None:
            save_info_from_api(cur_id, response, folder="not_found")
    else:
        logging.warning(f"Информация о фильме {cur_id} уже существует.")
