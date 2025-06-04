import typing
import warnings
import time
from functools import wraps
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import dendrogram


from kmeans import KMeans, METRICS

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


CLASSES = ['classical', 'rock', 'hip-hop', 'pop', 'electronic', 'world-music']
SEED = 78498

rng = np.random.default_rng(SEED)


class ClusterEnsemble:
    def __init__(self,
                 data: pd.DataFrame,
                 n_clusters: int = 6,
                 n_estimators: int = 5,
                 method: str = 'ward',
                 ):
        self.X = data.values
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.weights = np.ones((self.n_estimators,))
        self.method = method
        self.labels_ = None
        self.linkage_matrix_ = None

    def _create_estimators(self):
        self.estimators: list[KMeans] = []
        for _ in range(self.n_estimators):
            metric = rng.choice(typing.get_args(METRICS))
            self.estimators.append(KMeans(self.X, metric=metric))

    @measure_time
    def fit(self, supervised=False, true_labels: list[int] | None = None):
        if supervised and true_labels is None:
            raise Exception("Не заданы метки для обучения")
        if not supervised and true_labels is not None:
            warnings.warn(Warning("Метки для обучения будут проигнорированы"))
        self._create_estimators()

        dist_matrix = np.zeros((self.X.shape[0], self.X.shape[0]))
        accuracies = []
        for i in range(self.n_estimators):
            print(f'Количество кластеров: {self.estimators[i].n_clusters}\n'
                  f'Максимальное количество итераций: {self.estimators[i].max_iter}\n'
                  f'Выбранная метрика: {self.estimators[i].metric.metric}\n')
            self.estimators[i].fit()
            accuracies.append(self.estimators[i].rand_index(true_labels)
                              if supervised
                              else 1 / self.estimators[i].cluster_cohesion())
        accuracies = np.array(accuracies)
        self.weights = accuracies / np.sum(accuracies)
        print("Построение согласованной матрицы разбиений")

        for i in range(self.n_estimators):
            dist_matrix += self.weights[i] * self.estimators[i].cluster_similarity_matrix()
        
        self.linkage_matrix_ = linkage(dist_matrix, method=self.method)
        self.labels_ = fcluster(self.linkage_matrix_, t=self.n_clusters, criterion='maxclust') - 1


    def plot_dendrogram(self):
        plt.figure(figsize=(15, 10))
        dendrogram(self.linkage_matrix_)
        plt.title('Дендрограмма')
        plt.xlabel('Номер входного элемента')
        plt.ylabel('Расстояние')
        plt.show()

    def rand_index(self, real_labels):
        return rand_score(real_labels, self.labels_)
    
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
    # visualize(data.drop(columns=['genre']), track_labels, "Исходные данные", CLASSES)
    
    ensemble = ClusterEnsemble(data.drop(columns=['genre']), n_estimators=13)
    ensemble.fit(supervised=False, true_labels=track_labels)
    print("Метки кластеров:", ensemble.labels_)

    visualize(data.drop(columns=['genre']), ensemble.labels_, "Кластеризованные данные")
    print(f'Значение Rand индекса: {ensemble.rand_index(track_labels)}')
    print(f'Плотност кластеров: {ensemble.cluster_cohesion()}')

    ensemble.plot_dendrogram()


if __name__ == '__main__':
    main()