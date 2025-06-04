import time
from functools import wraps
from collections import defaultdict
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math

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

class DecisionNode:
    def __init__(self, 
                 feature_idx: Optional[int] = None, 
                 threshold: Optional[float] = None,
                 left: Optional['DecisionNode'] = None,
                 right: Optional['DecisionNode'] = None,
                 value: Optional[int] = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, 
                 max_depth: int = 5, 
                 min_samples_split: int = 2,
                 criterion: str = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionNode:
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth 
            or n_labels == 1 
            or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return DecisionNode(value=self._most_common_label(y))

        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return DecisionNode(best_feature, best_threshold, left, right)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def _information_gain(self, X: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float) -> float:
        if self.criterion == 'gini':
            parent_impurity = self._gini_impurity(y)
        else:
            parent_impurity = self._entropy(y)

        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = X[:, feature_idx] > threshold
        
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        
        if self.criterion == 'gini':
            child_impurity = (n_left / n) * self._gini_impurity(y[left_idxs]) + \
                            (n_right / n) * self._gini_impurity(y[right_idxs])
        else:
            child_impurity = (n_left / n) * self._entropy(y[left_idxs]) + \
                            (n_right / n) * self._entropy(y[right_idxs])

        return parent_impurity - child_impurity
    
    @staticmethod
    def _gini_impurity(y: np.ndarray) -> float:
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * math.log(p) for p in probabilities if p > 0])
    
    @staticmethod
    def _most_common_label(y: np.ndarray) -> int:
        counts = np.bincount(y)
        return np.argmax(counts)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x: np.ndarray, node: DecisionNode) -> int:
        if node.is_leaf():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


class BaggingClassifier:
    def __init__(self, 
                 n_estimators: int = 10,
                 max_samples: float = 1.0,
                 max_features: float = 1.0,
                 max_depth: int = 5):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_depth = max_depth
        self.estimators = []
    
    @measure_time
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.estimators = []
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        sample_size = int(n_samples * self.max_samples)
        feature_size = int(n_features * self.max_features)
        
        for _ in tqdm(range(self.n_estimators), desc="Обучение деревьев"):
            sample_indices = rng.choice(n_samples, sample_size, replace=True)
            feature_indices = rng.choice(n_features, feature_size, replace=False)
            
            X_sample = X[sample_indices][:, feature_indices]
            y_sample = y[sample_indices]
            
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            
            self.estimators.append((tree, feature_indices))
    
    @measure_time
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros((X.shape[0], self.n_estimators))
        
        for i, (tree, feature_indices) in enumerate(self.estimators):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)
        
        final_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x.astype(int))), 
            axis=1, 
            arr=predictions
        )
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((X.shape[0], len(np.unique(self.y_))))
        
        for tree, feature_indices in self.estimators:
            X_subset = X[:, feature_indices]
            preds = tree.predict(X_subset)
            
            for i, pred in enumerate(preds):
                proba[i, pred] += 1
        
        proba /= self.n_estimators
        return proba



def visualize(df: pd.DataFrame, labels: list[int], name: str, label_names: list[str] = None):
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
    
    bagging = BaggingClassifier(
        n_estimators=50,
        max_depth=5
    )
    bagging.fit(X_train, y_train)
    
    y_pred = bagging.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASSES)
    
    print(f"Точность: {accuracy:.4f}")
    print("Отчёт классификации:")
    print(report)
    
    visualize(X_test, y_test, "Истинные классы (тестовая выборка)", CLASSES)
    visualize(X_test, y_pred, "Предсказанные классы (тестовая выборка)", CLASSES)


if __name__ == '__main__':
    main()