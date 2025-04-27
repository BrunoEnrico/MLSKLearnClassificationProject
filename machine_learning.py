from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class MachineLearning:
    def __init__(self):
        pass

    @staticmethod
    def get_svc(**kwargs) -> SVC:
        return SVC(**kwargs)

    @staticmethod
    def get_linear_svc(**kwargs) -> LinearSVC:
        return LinearSVC(**kwargs)

    @staticmethod
    def get_dummy(**kwargs) -> DummyClassifier:
        return DummyClassifier(**kwargs)

    @staticmethod
    def get_decision_tree(**kwargs) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**kwargs)

    @staticmethod
    def fit_dummy_classifier(dummy_classifier: DummyClassifier, x_train: np.ndarray, y_train: np.ndarray) -> None:
        dummy_classifier.fit(x_train, y_train)

    @staticmethod
    def fit_decision_tree_classifier(decision_tree: DecisionTreeClassifier,
                                     x_train: np.ndarray, y_train: np.ndarray) -> None:
        decision_tree.fit(x_train, y_train)

    @staticmethod
    def predict_dummy_classifier(dummy_classifier: DummyClassifier, feature: np.ndarray) -> np.ndarray:
        return dummy_classifier.predict(feature)

    @staticmethod
    def predict_decision_tree_classifier(decision_tree: DecisionTreeClassifier, feature: np.ndarray) -> np.ndarray:
        return decision_tree.predict(feature)

    @staticmethod
    def get_train_test_split(feature: np.ndarray, target: np.ndarray, **kwargs) -> tuple:
        return train_test_split(feature, target, **kwargs)

    @staticmethod
    def fit_linear_svc(linear_svc: LinearSVC, feature: np.ndarray, target: np.ndarray) -> object | LinearSVC:
        return linear_svc.fit(feature, target)

    @staticmethod
    def fit_svc(svc: SVC, feature: np.ndarray, target: np.ndarray) -> object | LinearSVC:
        return svc.fit(feature, target)

    @staticmethod
    def predict_linear_svc(linear_svc: LinearSVC, feature: np.ndarray) -> np.ndarray:
        return linear_svc.predict(feature)

    @staticmethod
    def get_accuracy_score(y: np.ndarray, prediction: np.ndarray) -> float | int:
        return accuracy_score(y, prediction)

    @staticmethod
    def get_standard_scaler(**kwargs) -> StandardScaler:
        return StandardScaler(**kwargs)

    @staticmethod
    def standard_scaler_fit(scaler: StandardScaler, data: np.ndarray) -> None:
        scaler.fit(data)
