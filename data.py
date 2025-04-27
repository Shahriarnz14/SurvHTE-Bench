import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import beta, bernoulli


class SyntheticDataGeneratorPlus:
    """
    Generate synthetic survival data scenarios (1–10) with Cox-based event and censoring times where appropriate.

    Each dataset includes:
      - train, test_random, test_quantiles
      - metadata dictionary

    Scenarios 2 & 10: T ~ Cox model with Weibull(λ=1,k=0.5) baseline and λ(t)=0.5 t^-0.5 e^{lp(X,W)}
    Scenarios 1 & 9: C ~ Cox model with Weibull(λ=1,k=2.0) baseline and λ(t)=2 t e^{lp_C(X,W)}

    All other scenarios remain as before.
    """
    def __init__(self, scenario=1, n_samples=5000, n_features=5,
                 random_state=None, dataset_name=None, RCT=False, treatment_proportion=0.5):
        assert 1 <= scenario <= 10, "Scenario must be 1–10"
        self.scenario = scenario
        self.n = n_samples
        self.p = n_features
        self.random_state = random_state
        self.dataset_name = dataset_name or f"scenario_{scenario}"
        self.rct = RCT
        self.treatment_proportion = treatment_proportion # only used if RCT=True
        np.random.seed(self.random_state)

    def _simulate_cox(self, linpred, baseline, params):
        """
        Inverse-transform sampling from a Cox model:
          S(t)=exp(-H0(t) e^{linpred}),  H0 for exponential or Weibull.
        """
        U = np.random.uniform(size=len(linpred))
        if baseline == 'exponential':
            lam0 = params['lambda']
            return -np.log(U) / (lam0 * np.exp(linpred))
        elif baseline == 'weibull':
            lam0 = params['lambda']
            k = params['k']
            return lam0 * ((-np.log(U) / np.exp(linpred)) ** (1.0 / k))
        else:
            raise ValueError("Unsupported baseline for Cox")

    def _gen_X(self, size):
        df = pd.DataFrame(
            np.random.rand(size, self.p),
            columns=[f'X{i+1}' for i in range(self.p)]
        )
        if self.scenario == 7:
            U = np.random.rand(size, 2)
            df['U1'], df['U2'] = U[:,0], U[:,1]
        return df

    def _add_treatment(self, df):
        df = df.copy()
        if self.rct:
            # Randomized treatment assignment
            W = np.random.binomial(1, self.treatment_proportion, size=len(df))
        else:
            X = df[[c for c in df.columns if c.startswith('X')]].values
            beta_pdf = beta.pdf(X[:, 0], 2, 4)
            e = (1 + beta_pdf) / 4 # propensity score e(X)
            W = bernoulli.rvs(e, size=(1, X.shape[0])).flatten()
        # TODO: add setting with unobserved confounders: e(X, U)
        df['W'] = W
        return df

    def _simulate_T(self, df):
        X = df[[c for c in df.columns if c.startswith('X')]].values
        W = df['W'].values
        s = self.scenario
        eps = np.random.normal(size=len(df)) # TODO: why some with eps, some without?
        # Cox-based T for scenarios 2 & 10
        if s in (2, 10):
            # linpred = X1 + (-0.5+X2)*W
            linpred = X[:,0] + (-0.5 + X[:,1]) * W
            return self._simulate_cox(linpred, 'weibull', {'lambda':1.0, 'k':0.5})
        # AFT scenario 1
        if s == 1:
            lp = -1.85 -0.8*(X[:,0]<0.5) +0.7*np.sqrt(X[:,1]) +0.2*X[:,2]
            lp += (0.7 -0.4*(X[:,0]<0.5) -0.4*np.sqrt(X[:,1])) * W
            return np.exp(lp + eps)
        # Poisson-based
        if s in (3,5,6,7,8):
            lam = X[:,1]**2 + X[:,2] + 6 + 2*(np.sqrt(X[:,0]) - 0.3) * W
            if s in [7,8]:
                lam += 1 # TODO: why setting 7,8 have a different constant value added?
            return np.random.poisson(lam)
            
        # Poisson variant 4
        if s == 4:
            lam = X[:,1] + X[:,2] + np.maximum(0, X[:,0] - 0.3) * W + 1e-3 # small constant added for stability
            return np.random.poisson(lam)
        # AFT scenario 9
        if s == 9:
            lp = 0.3 -0.5*(X[:,0]<0.5) +0.5*np.sqrt(X[:,1]) +0.2*X[:,2]
            lp += (1 -0.8*(X[:,0]<0.5) -0.8*np.sqrt(X[:,1])) * W
            return np.exp(lp + eps)
        raise ValueError("Unsupported scenario for T")

    def _simulate_C(self, df):
        X = df[[c for c in df.columns if c.startswith('X')]].values
        W = df['W'].values
        s = self.scenario
        # Cox-based C for scenarios 1 & 9
        if s in (1, 9):
            if s == 1:
                lpC = -1.75 -0.5*np.sqrt(X[:,1]) +0.2*X[:,2]
                lpC += (1.15 +0.5*(X[:,0]<0.5) -0.3*np.sqrt(X[:,1])) * W
            else:  # s==9
                lpC = -0.9 +2*np.sqrt(X[:,1]) +2*X[:,2]
                lpC += (1.15 +0.5*(X[:,0]<0.5) -0.3*np.sqrt(X[:,1])) * W
            return self._simulate_cox(lpC, 'weibull', {'lambda':1.0, 'k':2.0})
        # Uniform scenario 2
        if s == 2:
            return np.random.uniform(0,3, size=len(df))
        # Poisson scenarios 3,4
        if s in (3,4):
            lam = (12 if s == 3 else 1) + np.log1p(np.exp(X[:,2]))
            return np.random.poisson(lam)
        # piecewise uniform 5
        if s == 5:
            u = np.random.rand(len(df)) # Uniform(0,1)
            # if s < 0.6, return inf
            return np.where(u < 0.6, np.inf, 1 + (X[:,3] < 0.5).astype(int))
        # Poisson 6,7,8
        if s in (6,7,8):
            if s == 6:
                lam = 3 + np.log1p(np.exp(2*X[:,1] + X[:,2]))
            elif s == 7: # unknown mechanism censoring
                # TODO: add dependent time to censoring and time to event with U
                U = df[['U1','U2']].values
                lam = 3 + 4*U[:,0] + 2*U[:,1]
            else:
                lam = 3
            return np.random.poisson(lam, size=len(df))
        # interval-censor 10
        if s == 10:
            u = np.random.rand(len(df))
            c1 = np.inf
            c2 = np.random.uniform(0, 0.05,size=len(df))
            return np.where(u<0.1, c1, c2)
        raise ValueError("Unsupported scenario for C")

    def generate_datasets(self):
        # train, random test, quantile test
        df_train = self._add_treatment(self._gen_X(self.n))
        df_train['true_event_time'] = self._simulate_T(df_train)
        df_train['true_censor_time'] = self._simulate_C(df_train)
        df_train['observed_time'], df_train['event'] = (
            np.minimum(df_train['true_event_time'], df_train['true_censor_time']),
            (df_train['true_event_time'] <= df_train['true_censor_time']).astype(int)
        )
        df_train['id'] = np.arange(self.n)
        # random test
        df_rand = self._add_treatment(self._gen_X(self.n))
        df_rand['true_event_time'] = self._simulate_T(df_rand)
        df_rand['true_censor_time'] = self._simulate_C(df_rand)
        df_rand['observed_time'], df_rand['event'] = (
            np.minimum(df_rand['true_event_time'], df_rand['true_censor_time']),
            (df_rand['true_event_time'] <= df_rand['true_censor_time']).astype(int)
        )
        df_rand['id'] = np.arange(self.n)
        # quantile test
        qs = np.linspace(0,1,21)
        Xq = np.tile(qs, (1, self.p)).reshape(21, self.p)
        df_q = pd.DataFrame(Xq, columns=[f'X{i+1}' for i in range(self.p)])
        if self.scenario == 7:
            for j in range(2): df_q[f'U{j+1}'] = qs
        df_q = self._add_treatment(df_q)
        df_q['true_event_time'] = self._simulate_T(df_q)
        df_q['true_censor_time'] = self._simulate_C(df_q)
        df_q['observed_time'], df_q['event'] = (
            np.minimum(df_q['true_event_time'], df_q['true_censor_time']),
            (df_q['true_event_time'] <= df_q['true_censor_time']).astype(int)
        )
        df_q['id'] = np.arange(len(df_q))
        metadata = dict(
            dataset_name=self.dataset_name,
            scenario=self.scenario,
            n_samples=self.n,
            n_features=self.p,
            random_state=self.random_state
        )
        return {'train':df_train, 'test_random':df_rand, 'test_quantiles':df_q, 'metadata':metadata}

class SyntheticDataGenerator:
    """
    Generate synthetic survival data for causal inference benchmarking with flexible treatment mechanisms.

    Each generated dataset includes both the DataFrame and metadata dictionary describing
    the generation parameters.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Number of observed covariate features.
    n_unobserved : int
        Number of unobserved covariate features.
    treatment_proportion : float
        Base probability for treatment when mechanism is 'independent'.
    treatment_mechanism : str
        'independent', 'observed', or 'unobserved' treatment assignment.
    cov_correlation : str
        'independent', 'low', or 'high' correlation among covariates.
    censoring_rate : float
        Exponential rate parameter for censoring distribution.
    censoring_mechanism : str
        'independent', 'dependent', or 'unknown'.
    error_dist : str
        'weibull', 'lognormal', or 'loglogistic'.
    model_assumption : str
        'PH' or 'AFT'.
    dataset_name : str, optional
        A name for the synthetic dataset.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_samples=5000,
        n_features=5,
        n_unobserved=0,
        treatment_proportion=0.5,
        treatment_mechanism='independent',
        cov_correlation='independent',
        censoring_rate=0.3,
        censoring_mechanism='independent',
        error_dist='weibull',
        model_assumption='PH',
        dataset_name=None,
        random_state=None
    ):
        self.n_samples = n_samples
        self.n_obs = n_features
        self.n_unobs = n_unobserved
        self.p = treatment_proportion
        self.treat_mech = treatment_mechanism
        self.cov_corr = cov_correlation
        self.censor_rate = censoring_rate
        self.censor_mech = censoring_mechanism
        self.error_dist = error_dist
        self.model = model_assumption
        self.dataset_name = dataset_name or f"synthetic_{np.random.randint(1e6)}"
        self.random_state = random_state
        np.random.seed(self.random_state)

    def _generate_covariates(self):
        if self.cov_corr == 'independent':
            cov = np.eye(self.n_obs)
        elif self.cov_corr == 'low':
            cov = np.full((self.n_obs, self.n_obs), 0.2)
            np.fill_diagonal(cov, 1)
        elif self.cov_corr == 'high':
            cov = np.full((self.n_obs, self.n_obs), 0.8)
            np.fill_diagonal(cov, 1)
        else:
            raise ValueError("cov_correlation must be 'independent', 'low', or 'high'.")
        mean = np.zeros(self.n_obs)
        X_obs = multivariate_normal.rvs(mean, cov, size=self.n_samples)
        X_unobs = np.random.normal(size=(self.n_samples, self.n_unobs))
        df_obs = pd.DataFrame(
            X_obs, columns=[f'feature{i+1}' for i in range(self.n_obs)]
        )
        df_unobs = pd.DataFrame(
            X_unobs, columns=[f'unobserved{i+1}' for i in range(self.n_unobs)]
        )
        return df_obs, df_unobs

    def _assign_treatment(self, X_obs, X_unobs):
        if self.treat_mech == 'independent':
            return np.random.binomial(1, self.p, size=self.n_samples)
        elif self.treat_mech == 'observed':
            logits = X_obs.dot(np.ones(self.n_obs) * 0.5)
            probs = 1 / (1 + np.exp(-logits))
            return np.random.binomial(1, probs)
        elif self.treat_mech == 'unobserved':
            logits = X_unobs.dot(np.ones(self.n_unobs) * 0.5)
            probs = 1 / (1 + np.exp(-logits))
            return np.random.binomial(1, probs)
        else:
            raise ValueError(
                "treatment_mechanism must be 'independent', 'observed', or 'unobserved'."
            )

    def _simulate_event_times(self, X_obs, W):
        if self.error_dist == 'weibull':
            base_lambda, base_rho = 1.0, 1.5
            U = np.random.uniform(size=self.n_samples)
            linear_pred = X_obs.dot(np.ones(self.n_obs) * 0.5) + W * 0.5
            if self.model == 'PH':
                T = (-np.log(U) / (base_lambda * np.exp(linear_pred))) ** (1 / base_rho)
            elif self.model == 'AFT':
                T = np.exp(( -np.log(-np.log(U) / base_lambda) - linear_pred ) / base_rho)
            else:
                raise ValueError("model_assumption must be 'PH' or 'AFT'.")
        elif self.error_dist == 'lognormal':
            sigma = 0.5
            linear_pred = X_obs.dot(np.ones(self.n_obs) * 0.5) + W * 0.5
            if self.model == 'AFT':
                T = np.random.lognormal(mean=linear_pred, sigma=sigma, size=self.n_samples)
            else:
                raise ValueError("model_assumption must be 'AFT' for log-normal.")
        elif self.error_dist == 'loglogistic':
            alpha, beta = 1.0, 1.5
            linear_pred = X_obs.dot(np.ones(self.n_obs) * 0.5) + W * 0.5
            if self.model == 'PH':
                U = np.random.uniform(size=self.n_samples)
                T = alpha * (U / (1 - U) * np.exp(-linear_pred)) ** (1 / beta)
            else:
                raise ValueError("model_assumption must be 'PH' for log-logistic.")
        else:
            raise ValueError("Unsupported error distribution.")
        return T

    def _simulate_censoring(self, T, X_obs, W):
        rate = self.censor_rate
        if self.censor_mech == 'independent':
            C = np.random.exponential(scale=1 / rate, size=self.n_samples)
        elif self.censor_mech == 'dependent':
            lam = rate * np.exp(-0.3 * (X_obs.sum(axis=1) + W))
            C = np.random.exponential(scale=1 / lam)
        elif self.censor_mech == 'unknown':
            C_ind = np.random.exponential(scale=1 / rate, size=self.n_samples)
            lam = rate * np.exp(-0.3 * (X_obs.sum(axis=1) + W))
            C_dep = np.random.exponential(scale=1 / lam)
            mask = np.random.binomial(1, 0.5, size=self.n_samples).astype(bool)
            C = np.where(mask, C_ind, C_dep)
        else:
            raise ValueError(
                "censoring_mechanism must be 'independent', 'dependent', or 'unknown'."
            )
        observed_time = np.minimum(T, C)
        event = (T <= C).astype(int)
        return observed_time, event

    def generate_dataset(self):
        """
        Returns
        -------
        data : pd.DataFrame
            Generated samples with id, outcomes, treatment, observed and unobserved covariates.
        metadata : dict
            Dictionary containing dataset_name and generation parameters.
        """
        df_obs, df_unobs = self._generate_covariates()
        W = self._assign_treatment(df_obs.values, df_unobs.values)
        T = self._simulate_event_times(df_obs.values, W)
        observed, event = self._simulate_censoring(T, df_obs.values, W)
        df = pd.concat([df_obs, df_unobs], axis=1)
        df['id'] = np.arange(self.n_samples)
        df['treatment_indicator'] = W
        df['observed_time'] = observed
        df['event_indicator'] = event
        cols = ['id', 'observed_time', 'event_indicator', 'treatment_indicator'] + \
               [f'feature{i+1}' for i in range(self.n_obs)] + \
               [f'unobserved{i+1}' for i in range(self.n_unobs)]
        data = df[cols]
        metadata = {
            'dataset_name': self.dataset_name,
            'n_samples': self.n_samples,
            'n_features': self.n_obs,
            'n_unobserved': self.n_unobs,
            'treatment_proportion': self.p,
            'treatment_mechanism': self.treat_mech,
            'cov_correlation': self.cov_corr,
            'censoring_rate': self.censor_rate,
            'censoring_mechanism': self.censor_mech,
            'error_dist': self.error_dist,
            'model_assumption': self.model,
            'random_state': self.random_state
        }
        return data, metadata

class SurvivalDataLoader:
    """
    Load survival causal inference datasets from disk, memory or dict with metadata.

    If a dict is passed, it should have keys 'data' (DataFrame) and 'metadata' (dict).
    """
    def __init__(self, data_source):
        if isinstance(data_source, dict):
            self.df = data_source['data'].copy()
            self.metadata = data_source.get('metadata', {})
        elif isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
            self.metadata = {}
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
            self.metadata = {}
        else:
            raise ValueError("data_source must be a dict, file path, or pandas DataFrame.")

    def get_features(self, include_unobserved=False):
        obs = [c for c in self.df.columns if c.startswith('feature')]
        if include_unobserved:
            unobs = [c for c in self.df.columns if c.startswith('unobserved')]
            return self.df[obs + unobs].values, obs + unobs
        return self.df[obs].values, obs

    def get_outcomes(self):
        return self.df['observed_time'].values, self.df['event_indicator'].values

    def get_treatment(self):
        return self.df['treatment_indicator'].values

    def split_data(self, test_size=0.2, random_state=None):
        from sklearn.model_selection import train_test_split
        strat = self.df['treatment_indicator'] if 'treatment_indicator' in self.df else None
        train, test = train_test_split(
            self.df, test_size=test_size,
            random_state=random_state, stratify=strat
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)
