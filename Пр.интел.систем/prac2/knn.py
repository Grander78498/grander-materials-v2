from collections import defaultdict
from typing import Literal, Callable
from functools import wraps
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm


CLASSES = ['classical', 'rock', 'hip-hop', 'pop', 'electronic', 'world-music']

METRICS = Literal["euclidean", "manhattan", "chebyshev", "jaccard"]
SEED = 78498
rng = np.random.default_rng(SEED)

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Метод {func.__name__} выполнился за {execution_time:.4f} секунд")
        return result
    return wrapper

class Distance:
    def __init__(self, metric: METRICS = "euclidean"):
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
        intersection = np.sum(np.minimum(x, y))
        union = np.sum(np.maximum(x, y))
        return 1 - intersection / union if union != 0 else 0

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.float32:
        return self._func(x, y)


class KNNClassifier:
    def __init__(self,
                 n_neighbors: int = 5,
                 metric: METRICS = "euclidean",
                 weights: Literal["uniform", "distance"] = "uniform"):
        self.n_neighbors = n_neighbors
        self.metric = Distance(metric)
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    @measure_time
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
    
    @measure_time
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        predictions = np.empty(X.shape[0], dtype=self.y_train.dtype)
        
        for i, x in enumerate(tqdm(X, desc="Predicting")):
            # Вычисляем расстояния до всех точек обучающей выборки
            distances = np.array([self.metric(x, x_train) for x_train in self.X_train])
            
            # Находим индексы k ближайших соседей
            k_nearest_indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            if self.weights == "uniform":
                # Простое голосование по большинству
                unique, counts = np.unique(k_nearest_labels, return_counts=True)
                predictions[i] = unique[np.argmax(counts)]
            else:
                # Взвешенное голосование (обратно пропорционально расстоянию)
                k_nearest_distances = distances[k_nearest_indices]
                weights = 1 / (k_nearest_distances + 1e-10)  # Добавляем небольшое значение для избежания деления на 0
                weighted_votes = np.zeros(len(CLASSES))
                
                for label, weight in zip(k_nearest_labels, weights):
                    weighted_votes[label] += weight
                
                predictions[i] = np.argmax(weighted_votes)
        
        return predictions


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
    string_columns = ['genre']
    for column in string_columns:
        column_types[column] = str

    data = pd.read_csv('processed.csv', dtype=column_types)

    encoder = LabelEncoder()
    track_labels = encoder.fit_transform(data.genre)
    # visualize(data.drop(columns=['genre']), track_labels, "Исходные данные", CLASSES)
    
    
    X = data.drop(columns=['genre']).values
    y = track_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn = KNNClassifier(n_neighbors=5, metric="chebyshev", weights="distance")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASSES)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    visualize(X_test, y_test, "Истинные классы (тестовая выборка)", CLASSES)
    visualize(X_test, y_pred, "Предсказанные классы (тестовая выборка)", CLASSES)


if __name__ == '__main__':
    main()