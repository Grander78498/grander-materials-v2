import random
import pandas as pd
from faker import Faker
from typing import List, Dict

fake = Faker()

# Генерация списка стран
COUNTRIES = [fake.country() for _ in range(20)]

# Список жанров музыки
GENRES = [
    'Pop', 'Rock', 'Hip-Hop', 'R&B', 'Electronic',
    'Jazz', 'Classical', 'Country', 'Metal', 'Indie',
    'Reggae', 'Blues', 'Folk', 'Disco', 'Techno'
]

NUM_USERS = 1000
NUM_SONGS = 5000
NUM_INTERACTIONS = 50000

def generate_user(num: int = 1) -> List[Dict]:
    """Генерация пользователей"""
    users = []
    for i in range(num):
        user = {
            'user_id': i,
            'age': random.randint(12, 99),
            'country': random.choice(COUNTRIES),
            'gender': random.choice(['M', 'F']),
            'like_count': random.randint(0, 500),
            'artists_count': random.randint(0, 100)
        }
        users.append(user)
    return users

def generate_song(num: int = 1) -> List[Dict]:
    """Генерация песен"""
    songs = []
    for i in range(num):
        song = {
            'song_id': i,
            'listen_count': random.randint(0, 1000000),
            'like_count': random.randint(0, 50000),
            'view_count': random.randint(0, 100000),
            'genre': random.choice(GENRES),
            'duration': random.randint(120, 600),  # секунды
            'explicit': random.choice([True, False])
        }
        songs.append(song)
    return songs

def generate_interactions(users: List[Dict], songs: List[Dict], num: int = 1000) -> pd.DataFrame:
    """Генерация взаимодействий пользователей с песнями"""
    data = []
    for _ in range(num):
        user = random.choice(users)
        song = random.choice(songs)
        
        interaction = {
            'user_id': user['user_id'],
            'song_id': song['song_id'],
            'listen_time': random.randint(1, 600),  # секунд
            'liked': random.choices([True, False], weights=[0.3, 0.7])[0],
            'timestamp': fake.date_time_this_year()
        }
        data.append(interaction)
    
    return pd.DataFrame(data)

def save_dataset(users: List[Dict], songs: List[Dict], interactions: pd.DataFrame):
    """Сохранение датасета в файлы"""
    pd.DataFrame(users).to_csv('users.csv', index=False)
    pd.DataFrame(songs).to_csv('songs.csv', index=False)
    interactions.to_csv('interactions.csv', index=False)

def generate_music_dataset(num_users: int = 1000, num_songs: int = 5000, num_interactions: int = 50000):
    """Генерация полного датасета"""
    print("Генерация пользователей...")
    users = generate_user(num_users)
    
    print("Генерация песен...")
    fake.unique.clear()
    songs = generate_song(num_songs)
    
    print("Генерация взаимодействий...")
    interactions = generate_interactions(users, songs, num_interactions)
    
    print("Сохранение данных...")
    save_dataset(users, songs, interactions)
    
    print(f"Датасет создан: {num_users} пользователей, {num_songs} песен, {num_interactions} взаимодействий")
    return users, songs, interactions

# Пример использования
if __name__ == "__main__":
    generate_music_dataset(NUM_USERS, NUM_SONGS, NUM_INTERACTIONS)