from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
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
    def evaluate_test(self, X_test, Y_test, W_test):
        """
        Evaluate base models on test data.
        
        Parameters:
        - X_test (np.ndarray): Test features
        - Y_test (np.ndarray): Test targets
        
        Returns:
        - dict: Evaluation metrics for each base model
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
    
    def evaluate_test(self, X_test, Y_test, W_test):
        """
        Evaluate base models on test data.
        
        Parameters:
        - X_test (np.ndarray): Test features
        - Y_test (np.ndarray): Test targets
        - W_test (np.ndarray): Treatment indicators
        
        Returns:
        - dict: Evaluation metrics for each base model
        """
        self.evaluation_test_dict = {}
        
        # Evaluate treated model
        treated_eval = self.models['treated'].evaluate(X_test[W_test == 1], Y_test[W_test == 1])
        self.evaluation_test_dict['treated'] = treated_eval
        
        # Evaluate control model
        control_eval = self.models['control'].evaluate(X_test[W_test == 0], Y_test[W_test == 0])
        self.evaluation_test_dict['control'] = control_eval
        
        return self.evaluation_test_dict


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
    
    def evaluate_test(self, X_test, Y_test, W_test):
        """
        Evaluate base model on test data.
        
        Parameters:
        - X_test (np.ndarray): Test features
        - Y_test (np.ndarray): Test targets
        - W_test (np.ndarray): Treatment indicators
        
        Returns:
        - dict: Evaluation metrics for the base model
        """
        self.evaluation_test_dict = {}
        
        # Augment features with treatment indicator
        X_aug = np.column_stack((X_test, W_test))
        
        # Evaluate model
        model_eval = self.models['s'].evaluate(X_aug, Y_test)
        self.evaluation_test_dict['s'] = model_eval
        
        return self.evaluation_test_dict


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

    def evaluate_test(self, X_test, Y_test, W_test):
        """
        Evaluate base models on test data.
        
        Parameters:
        - X_test (np.ndarray): Test features
        - Y_test (np.ndarray): Test targets
        - W_test (np.ndarray): Treatment indicators
        
        Returns:
        - dict: Evaluation metrics for each base model
        """
        self.evaluation_test_dict = {}
        
        # Evaluate treated model
        treated_eval = self.models['mu1'].evaluate(X_test[W_test == 1], Y_test[W_test == 1])
        self.evaluation_test_dict['mu1'] = treated_eval
        
        # Evaluate control model
        control_eval = self.models['mu0'].evaluate(X_test[W_test == 0], Y_test[W_test == 0])
        self.evaluation_test_dict['mu0'] = control_eval

        # Evaluate tau0 model
        tau0_eval = self.models['tau0'].evaluate(X_test[W_test == 0], self.models['mu1'].predict(X_test[W_test == 0]) - Y_test[W_test == 0])
        self.evaluation_test_dict['tau0'] = tau0_eval
        # Evaluate tau1 model
        tau1_eval = self.models['tau1'].evaluate(X_test[W_test == 1], Y_test[W_test == 1] - self.models['mu0'].predict(X_test[W_test == 1]))
        self.evaluation_test_dict['tau1'] = tau1_eval

        # Evaluate propensity model
        propensity_eval = self._eval_propensity_model(X_test, W_test)
        self.evaluation_test_dict['propensity'] = propensity_eval
        
        return self.evaluation_test_dict
    
    def _eval_propensity_model(self, X_test, W_test):
        """
        Evaluate the propensity model using accuracy.
        
        Parameters:
        - X_test (np.ndarray): Test features
        - W_test (np.ndarray): Treatment indicators
        
        Returns:
        - dict: Evaluation metrics for the propensity model
        """
        if 'propensity' not in self.models:
            raise ValueError("Propensity model not fitted yet.")
        
        y_prob = self.models['propensity'].predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(W_test, y_prob)
        f1 = f1_score(W_test, y_pred)
        return {'auc': auc, 'f1': f1}