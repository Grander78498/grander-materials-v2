import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch, DataLoader
import pandas as pd
import numpy as np
from generate import GENRES, NUM_USERS, NUM_SONGS

class MusicGNN(nn.Module):
    def __init__(self, hidden_dim=128, heads=4):
        super(MusicGNN, self).__init__()

        self.user_age_encoder = nn.Embedding(90, 8)
        self.user_country_encoder = nn.Embedding(20, 8)
        self.user_gender_encoder = nn.Embedding(3, 4)

        self.user_like_encoder = nn.Linear(1, 8)
        self.user_artists_encoder = nn.Linear(1, 8)
        self.user_proj = nn.Linear(8+8+4+8+8, hidden_dim)

        self.song_genre_encoder = nn.Embedding(len(GENRES), 12)
        self.song_explicit_encoder = nn.Embedding(2, 4)

        self.song_listen_encoder = nn.Linear(1, 16)
        self.song_like_encoder = nn.Linear(1, 16)
        self.song_view_encoder = nn.Linear(1, 16)
        self.song_duration_encoder = nn.Linear(1, 8)
        self.song_proj = nn.Linear(12+4+16+16+16+8, hidden_dim)

        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)

        self.rating_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def encode_user(self, user_data):
        age_emb = self.user_age_encoder(torch.clamp(user_data['age']-12, 0, 89))
        country_emb = self.user_country_encoder(user_data['country'])
        gender_emb = self.user_gender_encoder(user_data['gender'])
        like_emb = self.user_like_encoder(user_data['like_count'].float().unsqueeze(-1))
        artists_emb = self.user_artists_encoder(user_data['artists_count'].float().unsqueeze(-1))

        user_feats = torch.cat([age_emb, country_emb, gender_emb, like_emb, artists_emb], dim=-1)
        return F.relu(self.user_proj(user_feats))

    def encode_song(self, song_data):
        genre_emb = self.song_genre_encoder(song_data['genre'])
        explicit_emb = self.song_explicit_encoder(song_data['explicit'].long())

        listen_emb = self.song_listen_encoder(song_data['listen_count'].float().unsqueeze(-1))
        like_emb = self.song_like_encoder(song_data['like_count'].float().unsqueeze(-1))
        view_emb = self.song_view_encoder(song_data['view_count'].float().unsqueeze(-1))
        duration_emb = self.song_duration_encoder(song_data['duration'].float().unsqueeze(-1))

        song_feats = torch.cat([genre_emb, explicit_emb, listen_emb, like_emb, view_emb, duration_emb], dim=-1)
        return F.relu(self.song_proj(song_feats))

    def forward(self, data):
        user_emb = self.encode_user(data.x_user)
        track_emb = self.encode_song(data.x_track)
        x = torch.cat([user_emb, track_emb], dim=0)

        x = F.relu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)

        user_emb_final = x[:user_emb.size(0)]
        track_emb_final = x[user_emb.size(0):]

        if hasattr(data, 'user_batch'):
            user_emb_final = global_mean_pool(user_emb_final, data.user_batch)
            track_emb_final = global_mean_pool(track_emb_final, data.track_batch)

        if hasattr(data, 'user_idx') and hasattr(data, 'track_idx'):
            user_selected = user_emb_final[data.user_idx]
            track_selected = track_emb_final[data.track_idx]
            pair_emb = torch.cat([user_selected, track_selected], dim=-1)
            ratings = self.rating_predictor(pair_emb)
            return ratings

        return user_emb_final, track_emb_final


class MusicRecommender:
    def __init__(self, model, graph_data, user_id_to_idx, song_id_to_idx, device='cpu'):
        self.model = model.to(device)
        self.graph_data = graph_data.to(device)
        self.user_id_to_idx = user_id_to_idx
        self.song_id_to_idx = song_id_to_idx
        self.reverse_song_idx = {v: k for k, v in song_id_to_idx.items()}
        self.device = device
        
        with torch.no_grad():
            self.model.eval()
            self.user_embs, self.song_embs = model(self.graph_data)
    
    def recommend_for_user(self, user_id, top_k=10, filter_explicit=False):
        user_idx = self.user_id_to_idx.get(user_id, None)
        if user_idx is None:
            raise ValueError(f"User ID {user_id} not found in dataset")
        
        user_emb = self.user_embs[user_idx].unsqueeze(0)
        
        scores = torch.mm(user_emb, self.song_embs.t()).squeeze(0)
        
        listened_songs = self._get_listened_songs(user_idx)
        scores[listened_songs] = -float('inf')
        
        if filter_explicit:
            explicit_songs = self._get_explicit_songs()
            scores[explicit_songs] = -float('inf')
        
        top_scores, top_indices = torch.topk(scores, top_k)
        
        recommended_songs = [self.reverse_song_idx[idx.item()] for idx in top_indices]
        recommendation_scores = top_scores.cpu().numpy()
        
        return recommended_songs, recommendation_scores
    
    def _get_listened_songs(self, user_idx):
        edge_index = self.graph_data.edge_index.cpu().numpy()
        user_mask = edge_index[0] == user_idx
        listened_songs = edge_index[1][user_mask]
        return torch.tensor(listened_songs, device=self.device)
    
    def _get_explicit_songs(self):
        return torch.where(self.graph_data.x_track['explicit'] == 1)[0]


class MusicDataLoader:
    def __init__(self, users_path='users.csv', songs_path='songs.csv', interactions_path='interactions.csv'):
        self.users = pd.read_csv(users_path)
        self.songs = pd.read_csv(songs_path)
        self.interactions = pd.read_csv(interactions_path)

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.users['user_id'])}
        self.song_id_to_idx = {sid: idx for idx, sid in enumerate(self.songs['song_id'])}

        self._encode_categorical()

    def _encode_categorical(self):
        self.users['country_code'] = self.users['country'].astype('category').cat.codes
        self.users['gender_code'] = self.users['gender'].astype('category').cat.codes

        self.songs['genre_code'] = self.songs['genre'].astype('category').cat.codes
        self.songs['explicit_code'] = self.songs['explicit'].astype(int)

    def prepare_graph_data(self):
        user_indices = self.interactions['user_id'].map(self.user_id_to_idx).values
        song_indices = self.interactions['song_id'].map(self.song_id_to_idx).values

        edge_index = torch.tensor([user_indices, song_indices], dtype=torch.long)

        edge_attr = torch.tensor(self.interactions['listen_time'].values / 600.0,  # нормализация
                                dtype=torch.float).unsqueeze(-1)

        x_user = {
            'age': torch.tensor(self.users['age'].values, dtype=torch.long),
            'country': torch.tensor(self.users['country_code'].values, dtype=torch.long),
            'gender': torch.tensor(self.users['gender_code'].values, dtype=torch.long),
            'like_count': torch.tensor(self.users['like_count'].values, dtype=torch.float),
            'artists_count': torch.tensor(self.users['artists_count'].values, dtype=torch.float)
        }

        x_track = {
            'listen_count': torch.tensor(self.songs['listen_count'].values, dtype=torch.float),
            'like_count': torch.tensor(self.songs['like_count'].values, dtype=torch.float),
            'view_count': torch.tensor(self.songs['view_count'].values, dtype=torch.float),
            'genre': torch.tensor(self.songs['genre_code'].values, dtype=torch.long),
            'duration': torch.tensor(self.songs['duration'].values, dtype=torch.float),
            'explicit': torch.tensor(self.songs['explicit_code'].values, dtype=torch.long)
        }

        return Data(
            x_user=x_user,
            x_track=x_track,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
    

def train_model(graph_data, epochs=20):
    model = MusicGNN(hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    loader = DataLoader([graph_data], batch_size=1, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            pos_edges = batch.edge_index
            neg_users = torch.randint(0, NUM_USERS, (1, pos_edges.shape[1]), dtype=torch.long)
            neg_songs = torch.randint(0, NUM_SONGS, (1, pos_edges.shape[1]), dtype=torch.long)
            neg_edges = torch.cat([neg_users, neg_songs], dim=0)
            
            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
            labels = torch.cat([
                torch.ones(pos_edges.shape[1]),
                torch.zeros(neg_edges.shape[1])
            ]).unsqueeze(1)
            
            batch.user_idx = all_edges[0]
            batch.track_idx = all_edges[1]            
            predictions = model(batch)
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    return model


if __name__ == "__main__":
    print("Загрузка данных...")
    loader = MusicDataLoader('users.csv', 'songs.csv', 'interactions.csv')
    graph_data = loader.prepare_graph_data()
    
    print("\nОбучение модели...")
    model = train_model(graph_data, epochs=15)
    
    print("\nИнициализация рекомендательной системы...")
    recommender = MusicRecommender(
        model=model,
        graph_data=graph_data,
        user_id_to_idx=loader.user_id_to_idx,
        song_id_to_idx=loader.song_id_to_idx,
        device='cpu'
    )
    random_user_id = np.random.choice(list(loader.user_id_to_idx.keys()))
    print(f"\nГенерация рекомендаций для пользователя {random_user_id}:")
    
    recommended_songs, scores = recommender.recommend_for_user(
        random_user_id, 
        top_k=5,
        filter_explicit=True
    )
    
    print("\nТоп-5 рекомендуемых песен:")
    songs_df = loader.songs.set_index('song_id')
    for i, (song_id, score) in enumerate(zip(recommended_songs, scores)):
        song_info = songs_df.loc[song_id]
        print(f"{i+1}. ID: {song_id}")
        print(f"   Жанр: {song_info['genre']}")
        print(f"   Длительность: {song_info['duration']} сек")
        print(f"   Лайков: {song_info['like_count']}")
        print(f"   Оценка рекомендации: {score:.3f}")
        print()