# impute-causal-survival-analysis

## Synthetic Data Generation and save to h5 file
``` python
from data import SyntheticDataGeneratorPlus

### RCT with treatment rate 0.5
with pd.HDFStore("../synthetic_data/RCT_0_5.h5") as store:
    for i in range(1,11):
        if i in [3,4,6,7,10]:
            continue
        dataset_name = f'RCT_0_5_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                         n_samples=50000, random_state=2025,
                                         RCT=True, treatment_proportion=0.5, unobserved=False, overlap=True)
        dsets = gen.generate_datasets() 
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

### RCT with treatment rate 0.05
with pd.HDFStore("../synthetic_data/RCT_0_05.h5") as store:
    for i in range(1,11):
        if i in [3,4,6,7,10]:
            continue
        dataset_name = f'RCT_0_05_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                         n_samples=50000, random_state=2025,
                                         RCT=True, treatment_proportion=0.05, unobserved=False, overlap=True)
        dsets = gen.generate_datasets() 
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

### non-RCT with ignorability and overlap held - propensity score e(X)
with pd.HDFStore("../synthetic_data/e_X.h5") as store:
    for i in range(1,11):
        if i in [3,4,6,7,10]:
            continue
        dataset_name = f'e_X_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                        n_samples=50000, random_state=2025,
                                         RCT=False, unobserved=False, overlap=True)
        dsets = gen.generate_datasets()
        store[f"../scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

### non-RCT with ignorability violated and overlap held - propensity score e(X, U)
with pd.HDFStore("synthetic_data/e_X_U.h5") as store:
    for i in range(1,11):
        if i in [3,4,6,7,10]:
            continue
        dataset_name = f'e_X_U_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                         n_samples=50000, random_state=2025,
                                         RCT=False, unobserved=True, overlap=True)
        dsets = gen.generate_datasets()
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

### non-RCT with ignorability held but overlap violated - propensity score e(X)_no_overlap
with pd.HDFStore("../synthetic_data/e_X_no_overlap.h5") as store:
    for i in range(1,11):
        if i in [3,4,6,7,10]:
            continue
        dataset_name = f'e_X_no_overlap_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                        n_samples=50000, random_state=2025,
                                         RCT=False, unobserved=False, overlap=False)
        dsets = gen.generate_datasets()
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

### informative censoring
info_censor_baseline=0.1
info_censor_alpha=0.05

### informative_censoring and non-RCT with ignorability and overlap held - propensity score e(X)
with pd.HDFStore("../synthetic_data/e_X_info_censor.h5") as store:
    for i in range(1,11):
        if i in [3,4,6,7,10]:
            continue
        dataset_name = f'e_X_info_censor_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                         n_samples=50000, random_state=2025,
                                         informative_censoring=True, RCT=False, 
                                         unobserved=False, overlap=True,
                                         info_censor_baseline=info_censor_baseline,
                                         info_censor_alpha=info_censor_alpha)
        dsets = gen.generate_datasets() 
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

### informative_censoring and non-RCT with ignorability violated and overlap held - propensity score e(X, U)
with pd.HDFStore("../synthetic_data/e_X_U_info_censor.h5") as store:
    for i in range(1,11):
        if i in [3,4,6,7,10]:
            continue
        dataset_name = f'e_X_U_info_censor_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                         n_samples=50000, random_state=2025,
                                         informative_censoring=True, RCT=False, 
                                         unobserved=True, overlap=True,
                                         info_censor_baseline=info_censor_baseline,
                                         info_censor_alpha=info_censor_alpha)
        dsets = gen.generate_datasets() 
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

### informative_censoring and non-RCT with ignorability held but overlap violated - propensity score e(X)
with pd.HDFStore("../synthetic_data/e_X_no_overlap_info_censor.h5") as store:
    for i in range(1,11):
        if i in [3,4,6,7,10]:
            continue
        dataset_name = f'e_X_no_overlap_info_censor_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                         n_samples=50000, random_state=2025,
                                         informative_censoring=True, RCT=False, 
                                         unobserved=False, overlap=False,
                                         info_censor_baseline=info_censor_baseline,
                                         info_censor_alpha=info_censor_alpha)
        dsets = gen.generate_datasets() 
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']
```


## Loading data from h5 file
``` python
import pandas as pd
import ace_tools_open as tools

# List of HDF5 files to summarize
store_files = [
    "../synthetic_data/RCT_0_5.h5",
    "../synthetic_data/RCT_0_05.h5",
    "../synthetic_data/e_X.h5",
    "../synthetic_data/e_X_U.h5",
    "../synthetic_data/e_X_no_overlap.h5",
    "../synthetic_data/e_X_info_censor.h5",
    "../synthetic_data/e_X_U_info_censor.h5",
    "../synthetic_data/e_X_no_overlap_info_censor.h5"
]

experiment_setups = {}

for path in store_files:
    base_name = os.path.splitext(os.path.basename(path))[0]  # e.g. RCT_0_5
    scenario_dict = {}
    for scenario in range(1, 11):
        try:
            result = load_scenario_data(path, scenario)
            if result is not None:
                scenario_dict[f"scenario_{scenario}"] = result
        except Exception as e:
            # Log or ignore as needed
            print(f"Error loading scenario {scenario} from {path}: {e}")
            continue
    experiment_setups[base_name] = scenario_dict
```


The R implementation (`models_causal_impute_meta/RISTfunctions.r`) for RIST is downloaded from this [link](https://www.dropbox.com/scl/fi/mtfch9t0ogrww2nhz8bl0/RISTfunctions.r?rlkey=mq2ey6nmbvmavnp52b7oeq77n&e=1&dl=0), and can also be found under one of the authors of RIST, Ruoqing Zhu's [webpage](https://sites.google.com/site/teazrq/software?authuser=0)