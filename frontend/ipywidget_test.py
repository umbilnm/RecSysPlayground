import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Данные о пользователях и просмотренных фильмах
users_data = {
    666262: [
        {'title': 'Последний викинг', 'genres': 'боевик, историческое, приключения', 'country': 'Великобритания', 'year': 2018, 'watches': 746, 'poster': 'link_to_poster1.jpg'},
        {'title': 'Робин Гуд: Начало', 'genres': 'боевик, триллер, приключения', 'country': 'США', 'year': 2018, 'watches': 485, 'poster': 'link_to_poster2.jpg'},
        {'title': 'Томирис', 'genres': 'боевик, драма, историческое, военные', 'country': 'Казахстан', 'year': 2020, 'watches': 10370, 'poster': 'link_to_poster3.jpg'}
    ]
}

# Рекомендации по моделям
recommendations_data = {
    '2_stage_v0_competition': [
        {'title': 'Гнев человеческий', 'genres': 'боевик, триллер', 'country': 'Великобритания, США', 'year': 2021, 'watches': 132665, 'score': 0.84, 'poster': 'link_to_poster4.jpg'},
        {'title': 'Прабабушка легкого поведения', 'genres': 'комедия', 'country': 'Россия', 'year': 2021, 'watches': 74803, 'score': 0.76, 'poster': 'link_to_poster5.jpg'}
    ]
}

# Виджет для выбора пользователя
user_select = widgets.Dropdown(
    options=users_data.keys(),
    description='User:',
    value=666262
)

# Виджет для выбора модели рекомендаций
model_select = widgets.Dropdown(
    options=recommendations_data.keys(),
    description='Model:',
    value='2_stage_v0_competition'
)

# Вывод просмотренных фильмов
viewed_movies_output = widgets.Output()

def display_viewed_movies(user_id):
    with viewed_movies_output:
        viewed_movies_output.clear_output()
        movies = users_data[user_id]
        for movie in movies:
            response = requests.get(movie['poster'])
            img = Image.open(BytesIO(response.content))
            display(img)
            print(f"Title: {movie['title']}")
            print(f"Genres: {movie['genres']}")
            print(f"Country: {movie['country']}")
            print(f"Year: {movie['year']}")
            print(f"Watches: {movie['watches']}")
            print('---')

# Вывод рекомендаций
recommendations_output = widgets.Output()

def display_recommendations(model_name):
    with recommendations_output:
        recommendations_output.clear_output()
        recos = recommendations_data[model_name]
        for reco in recos:
            response = requests.get(reco['poster'])
            img = Image.open(BytesIO(response.content))
            display(img)
            print(f"Title: {reco['title']}")
            print(f"Genres: {reco['genres']}")
            print(f"Country: {reco['country']}")
            print(f"Year: {reco['year']}")
            print(f"Watches: {reco['watches']}")
            print(f"Score: {reco['score']}")
            print('---')

# Функции, вызываемые при изменении выбора
def on_user_change(change):
    display_viewed_movies(change['new'])

def on_model_change(change):
    display_recommendations(change['new'])

user_select.observe(on_user_change, names='value')
model_select.observe(on_model_change, names='value')

# Инициализация отображений
display_viewed_movies(user_select.value)
display_recommendations(model_select.value)

# Интерфейс приложения
display(widgets.HBox([user_select, model_select]))
display(viewed_movies_output)
display(recommendations_output)
