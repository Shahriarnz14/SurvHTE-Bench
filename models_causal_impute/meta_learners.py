from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import numpy as np
from .regressor_base import RegressorBaseLearner

class BaseMetaLearner(ABC):
    """
    Abstract base class for causal meta-learners.
    
    Subclasses must implement:
    - fit(X_train, W_train, Y_train)
    - predict_cate(X)
    """
    def __init__(self, base_model_name='ridge'):
        """
        Initialize the meta-learner with a specified base model type.
        
        Parameters:
        - base_model_name (str): Model type to use (ridge, lasso, rf, gbr, xgb).
        """
        self.base_model_name = base_model_name
        self.models = {}

    @abstractmethod
    def fit(self, X_train, W_train, Y_train):
        """
        Fit the meta-learner on training data.
        """
        pass

    @abstractmethod
    def predict_cate(self, X):
        """
        Predict Conditional Average Treatment Effect (CATE) on test data.
        """
        pass

    def evaluate(self, X, cate_true, W=None):
        """
        Evaluate CATE predictions using mean squared error.
        
        Parameters:
        - X (np.ndarray): Test features.
        - cate_true (np.ndarray): Ground-truth CATE values (from simulation).
        - W (np.ndarray): Treatment assignment (not used in this method).
        
        Returns:
        - mse (float): Mean squared error.
        - cate_pred (np.ndarray): Predicted CATE values.
        """
        cate_pred = self.predict_cate(X)
        ate_pred = np.mean(cate_pred)
        mse = mean_squared_error(cate_true, cate_pred)
        return mse, cate_pred, ate_pred


class TLearner(BaseMetaLearner):
    """
    T-Learner: Trains separate models for treated and control groups.
    """
    def fit(self, X_train, W_train, Y_train):
        model_treated = RegressorBaseLearner(model_name=self.base_model_name)
        model_control = RegressorBaseLearner(model_name=self.base_model_name)
        model_treated.fit(X_train[W_train == 1], Y_train[W_train == 1])
        model_control.fit(X_train[W_train == 0], Y_train[W_train == 0])
        self.models['treated'] = model_treated
        self.models['control'] = model_control

    def predict_cate(self, X):
        mu1 = self.models['treated'].predict(X)
        mu0 = self.models['control'].predict(X)
        return mu1 - mu0


class SLearner(BaseMetaLearner):
    """
    S-Learner: Trains a single model using treatment as a feature.
    """
    def fit(self, X_train, W_train, Y_train):
        X_aug = np.column_stack((X_train, W_train))
        model = RegressorBaseLearner(model_name=self.base_model_name)
        model.fit(X_aug, Y_train)
        self.models['s'] = model

    def predict_cate(self, X):
        X0 = np.column_stack((X, np.zeros(len(X))))
        X1 = np.column_stack((X, np.ones(len(X))))
        mu0 = self.models['s'].predict(X0)
        mu1 = self.models['s'].predict(X1)
        return mu1 - mu0


class XLearner(BaseMetaLearner):
    """
    X-Learner: Uses imputed treatment effects from T-Learner to train tau models,
    then combines with a propensity model.
    """
    def fit(self, X_train, W_train, Y_train):
        mu1 = RegressorBaseLearner(model_name=self.base_model_name)
        mu0 = RegressorBaseLearner(model_name=self.base_model_name)
        mu1.fit(X_train[W_train == 1], Y_train[W_train == 1])
        mu0.fit(X_train[W_train == 0], Y_train[W_train == 0])
        self.models['mu1'], self.models['mu0'] = mu1, mu0

        tau0 = mu1.predict(X_train[W_train == 0]) - Y_train[W_train == 0]
        tau1 = Y_train[W_train == 1] - mu0.predict(X_train[W_train == 1])

        tau0_model = RegressorBaseLearner(model_name=self.base_model_name)
        tau1_model = RegressorBaseLearner(model_name=self.base_model_name)
        tau0_model.fit(X_train[W_train == 0], tau0)
        tau1_model.fit(X_train[W_train == 1], tau1)
        self.models['tau0'] = tau0_model
        self.models['tau1'] = tau1_model

        self.models['propensity'] = LogisticRegression().fit(X_train, W_train)

    def predict_cate(self, X):
        tau0 = self.models['tau0'].predict(X)
        tau1 = self.models['tau1'].predict(X)
        p = self.models['propensity'].predict_proba(X)[:, 1]
        return p * tau1 + (1 - p) * tau0
