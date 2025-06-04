from collections import defaultdict
from pathlib import Path
from typing import Literal, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import rand_score
from sklearn.decomposition import PCA
from tqdm import tqdm


CLASSES = ['classical', 'rock', 'hip-hop', 'pop', 'electronic', 'world-music']

METRICS = Literal["euclidean", "manhattan", "chebyshev", "jaccard"]
SEED = 78498
rng = np.random.default_rng(SEED)

class Distance:
    def __init__(self,
                 metric: METRICS = "euclidean"):
        self.metric = metric
        metrics: dict[METRICS, Callable[[np.ndarray, np.ndarray], np.float32]] = {
            "euclidean": self._euclid,
            "manhattan": self._manhattan,
            "chebyshev": self._chebyshev,
            "jaccard": self._jaccard,
        }
        self._func = metrics[self.metric]

    @staticmethod
    def _euclid(x: np.ndarray, y: np.ndarray) -> np.float32:
        return np.linalg.norm(x - y)
    
    @staticmethod
    def _manhattan(x: np.ndarray, y: np.ndarray) -> np.float32:
        return np.sum(np.abs(x - y))
    
    @staticmethod
    def _chebyshev(x: np.ndarray, y: np.ndarray) -> np.float32:
        return np.max(np.abs(x - y))
    
    @staticmethod
    def _jaccard(x: np.ndarray, y: np.ndarray) -> np.float32:
        return np.dot(x, y) / (np.linalg.norm(x) + np.linalg.norm(y) - np.dot(x, y))

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.float32:
        return self._func(x, y)


class KMeans:
    def __init__(self,
                 data: np.ndarray,
                 n_clusters: int = 6,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 metric: METRICS = "euclidean"):
        self.X = data
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.cluster_centers_ = None
        self.metric = Distance(metric)

    def _compute_distances(self, X, centers):
        distances = np.zeros((len(X), len(centers)))
        for i, point in enumerate(X):
            for j, center in enumerate(centers):
                distances[i, j] = self.metric(point, center)
        return distances

    def _assign_clusters(self, X, centers):
        distances = self._compute_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        return labels, min_distances

    def fit(self):
        
        random_indices = rng.permutation(len(self.X))[:self.n_clusters]
        self.cluster_centers_ = self.X[random_indices]
        
        for _ in tqdm(range(self.max_iter)):
            labels, _ = self._assign_clusters(self.X, self.cluster_centers_)
            
            new_centers = np.array([self.X[labels == i].mean(axis=0) 
                                 for i in range(self.n_clusters)])
            
            if all(self.metric(old_center, new_center) < self.tol for (old_center, new_center) in zip(self.cluster_centers_, new_centers)):
                break
                
            self.cluster_centers_ = new_centers
        
        self.labels_ = labels
    
    def rand_index(self, real_labels):
        return rand_score(real_labels, self.labels_)
    
    def cluster_cohesion(self):
        result = 0
        for i in range(self.n_clusters):
            center = self.cluster_centers_[i]
            for j in range(len(self.X)):
                if self.labels_[j] == i:
                    result += np.linalg.norm(self.X[j] - center) ** 2
        return result
    
    def cluster_similarity_matrix(self):
        n = len(self.labels_)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if self.labels_[i] == self.labels_[j]:
                    matrix[i, j] = 1
        return matrix



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
    # visualize(data.drop(columns=['genre']), track_labels, "Исходные данные", CLASSES)
    
    kmeans = KMeans(data.drop(columns=['genre']).values, metric="manhattan")
    kmeans.fit()
    print("Метки кластеров:", kmeans.labels_)

    visualize(data.drop(columns=['genre']), kmeans.labels_, "Кластеризованные данные")
    print(f'Значение Rand индекса: {kmeans.rand_index(track_labels)}')
    print(f'Плотност кластеров: {kmeans.cluster_cohesion()}')


if __name__ == '__main__':
    main()