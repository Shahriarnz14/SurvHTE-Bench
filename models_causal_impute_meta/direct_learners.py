from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
import numpy as np
from econml.dml import DML, CausalForestDML
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression

class BaseDirectLearner(ABC):
    """
    Abstract base class for direct causal effect estimators like DML and Causal Forest.
    """
    def __init__(self):
        self.model = None

    @abstractmethod
    def fit(self, X_train, W_train, Y_train):
        pass

    @abstractmethod
    def predict_cate(self, X):
        pass

    def evaluate(self, X, cate_true, W=None):
        """
        Evaluate predicted CATE against ground-truth CATE.
        """
        cate_pred = self.predict_cate(X)
        mse = mean_squared_error(cate_true, cate_pred)
        ate_pred = np.mean(cate_pred)
        return mse, cate_pred, ate_pred
    

class DoubleMachineLearning(BaseDirectLearner):
    """
    Double Machine Learning (Partially Linear) using EconML DML.
    """
    def __init__(self):
        super().__init__()
        self.model = DML(
            model_final=StatsModelsLinearRegression(fit_intercept=False),
            model_y='auto',
            model_t='auto',
            discrete_treatment=True,
            random_state=42
        )

    def fit(self, X_train, W_train, Y_train):
        self.model.fit(Y_train, W_train, X=X_train)

    def predict_cate(self, X):
        return self.model.effect(X)
    

class CausalForest(BaseDirectLearner):
    """
    Causal Forest using EconML's CausalForestDML.
    """
    def __init__(self):
        super().__init__()
        self.model = CausalForestDML(
            model_y='auto',
            model_t='auto',
            discrete_treatment=True,
            random_state=42
        )

    def fit(self, X_train, W_train, Y_train):
        self.model.fit(Y_train, W_train, X=X_train)

    def predict_cate(self, X):
        return self.model.effect(X)
