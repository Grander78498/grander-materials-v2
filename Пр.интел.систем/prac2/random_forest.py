import time
from functools import wraps
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


CLASSES = ['classical', 'rock', 'hip-hop', 'pop', 'electronic', 'world-music']
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

@measure_time
def fit(forest, X_train, Y_train):
    forest.fit(X_train, Y_train)

@measure_time
def predict(forest, X_test):
    return forest.predict(X_test)


def visualize(df: pd.DataFrame, labels: list[int], name: str, label_names: list[str] = None):
    """Визуализация данных (без изменений)"""
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
    """Основная функция с использованием бэггинга"""
    column_types = defaultdict(np.float32)
    string_columns = ['genre']
    for column in string_columns:
        column_types[column] = str

    data = pd.read_csv('processed.csv', dtype=column_types)

    # Кодирование меток классов
    encoder = LabelEncoder()
    track_labels = encoder.fit_transform(data.genre)
    # visualize(data.drop(columns=['genre']), track_labels, "Исходные данные", CLASSES)
    
    X = data.drop(columns=['genre']).values
    y = track_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение бэггинга
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
    )
    fit(forest, X_train, y_train)
    
    y_pred = predict(forest, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASSES)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    visualize(X_test, y_test, "Истинные классы (тестовая выборка)", CLASSES)
    visualize(X_test, y_pred, "Предсказанные классы (тестовая выборка)", CLASSES)


if __name__ == '__main__':
    main()