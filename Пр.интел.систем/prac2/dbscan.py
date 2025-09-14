from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import rand_score
from sklearn.decomposition import PCA
from tqdm import tqdm


CLASSES = ['classical', 'rock', 'hip-hop', 'pop', 'electronic', 'world-music']


class DBSCAN:
    def __init__(self,
                 data: pd.DataFrame,
                 epsilon: float = 2,
                 min_samples: int = 10,
                 seed: int = 78498):
        self.X = data.values
        self.labels_ = np.full(len(self.X), -1)
        self.visited = set()
        self.epsilon = epsilon
        self.min_samples = min_samples
        np.random.seed(seed)

    def fit(self, x: pd.DataFrame | None = None):
        if x is not None:
            self.X = np.vstack((self.X, x.values))
            self.labels_ = np.full(len(self.X), -1)
            
        cluster_id = 0
        
        for point_idx in tqdm(range(len(self.X))):
            if point_idx in self.visited:
                continue
                
            self.visited.add(point_idx)
            neighbors = self._get_neighbors(point_idx)
            
            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1
            else:
                self._expand_cluster(point_idx, neighbors, cluster_id)
                cluster_id += 1

    def _expand_cluster(self, point_idx, neighbors, cluster_id):
        self.labels_[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if neighbor_idx not in self.visited:
                self.visited.add(neighbor_idx)
                new_neighbors = self._get_neighbors(neighbor_idx)
                
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
            
            i += 1

    def _get_neighbors(self, point_idx) -> list[int]:
        neighbors = []
        for other_idx in range(len(self.X)):
            if point_idx == other_idx:
                continue
            if np.linalg.norm(self.X[point_idx] - self.X[other_idx]) < self.epsilon:
                neighbors.append(other_idx)
        return neighbors

    @staticmethod
    def _distance(x, y):
        return np.linalg.norm(x - y)
    
    def denoise(self):
        not_noise_indexes = []
        for i in range(len(self.labels_)):
            if self.labels_[i] != -1:
                not_noise_indexes.append(i)
        return not_noise_indexes
    
    def rand_index(self, real_labels):
        new_label = np.max(self.labels_) + 1
        pred_labels = self.labels_
        for i in range(len(pred_labels)):
            if pred_labels[i] == -1:
                pred_labels[i] = new_label
                new_label += 1
        return rand_score(real_labels, pred_labels)
    
    def cluster_cohesion(self):
        result = 0
        for i in range(max(self.labels_)):
            centers = np.array([self.X[self.labels_ == i].mean(axis=0) 
                                 for i in range(max(self.labels_))])
            center = centers[i]
            for j in range(len(self.X)):
                if self.labels_[j] == i:
                    result += np.linalg.norm(self.X[j] - center) ** 2
        return result


def preprocessing(df: pd.DataFrame):
    df = df.loc[df.genre.isin(CLASSES)]
    df = df.drop_duplicates(subset=['artists', 'track_name']) \
           .reset_index(drop=True)
    df = df.drop(columns=df.columns[:5])

    df = df.drop(columns=['energy', 'loudness'])
    scaler = StandardScaler()
    df.iloc[:, :-1] = df.iloc[:, :-1].astype(float)
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

    return df


def visualize(df: pd.DataFrame, labels: list[int], name: str, label_names: list[str] | None = None):
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(df)

    scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, s=6)
    if label_names is not None:
        plt.legend(handles=scatter.legend_elements()[0], labels=label_names, title='Классы')
    else:
        plt.legend(*scatter.legend_elements(), title='Кластеры')
    plt.title(name)
    plt.show()


def main():
    column_types = defaultdict(np.float32)
    string_columns = ['track_id', 'artists', 'album_name', 'track_name', 'track_genre']
    for column in string_columns:
        column_types[column] = str

    df = pd.read_csv('dataset.csv', dtype=column_types)
    df = df.rename(columns={'track_genre': 'genre'})
    data = preprocessing(df)

    encoder = LabelEncoder()
    track_labels = encoder.fit_transform(data.genre)
    visualize(data.drop(columns=['genre']), track_labels, "Исходные данные", CLASSES)
    
    dbscan = DBSCAN(data.drop(columns=['genre']), epsilon=2.1, min_samples=8)
    dbscan.fit()
    print(f"Количество кластеров: {np.max(dbscan.labels_) + 1}")
    print("Метки кластеров:", dbscan.labels_)
    print(f"Количество шумовых точек: {len(data) - len(dbscan.denoise())}")

    visualize(data.drop(columns=['genre']), dbscan.labels_, "Кластеризованные данные")
    print(f'Значение Rand индекса: {dbscan.rand_index(track_labels)}')

    # processed_data = data.iloc[dbscan.denoise(), :]
    # processed_data.to_csv('processed.csv', encoding='utf-8', index=False)
    print(f'Плотност кластеров: {dbscan.cluster_cohesion()}')


if __name__ == '__main__':
    main()