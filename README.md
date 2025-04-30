# impute-causal-survival-analysis

## Synthetic Data Generation and save to h5 file
``` python
### RCT with treatment rate 0.5
with pd.HDFStore("synthetic_data/RCT_0_5.h5") as store:
    for i in range(1,11):
        if i in [4,7]:
            continue
        dataset_name = f'RCT_0_5_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                         n_samples=50000, random_state=2025,
                                         RCT=True, treatment_proportion=0.5, unobserved=False)
        dsets = gen.generate_datasets() 
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

### RCT with treatment rate 0.05
with pd.HDFStore("synthetic_data/RCT_0_05.h5") as store:
    for i in range(1,11):
        if i in [4,7]:
            continue
        dataset_name = f'RCT_0_05_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                         n_samples=50000, random_state=2025,
                                         RCT=True, treatment_proportion=0.05, unobserved=False)
        dsets = gen.generate_datasets() 
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

### non-RCT with ignorability held - propensity score e(X)
with pd.HDFStore("synthetic_data/e_X.h5") as store:
    for i in range(1,11):
        if i in [4,7]:
            continue
        dataset_name = f'e_X_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                        n_samples=50000, random_state=2025,
                                         RCT=False, unobserved=False)
        dsets = gen.generate_datasets()
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

### non-RCT with ignorability violated - propensity score e(X, U)
with pd.HDFStore("synthetic_data/e_X_U.h5") as store:
    for i in range(1,11):
        if i in [4,7]:
            continue
        dataset_name = f'e_X_U_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                         n_samples=50000, random_state=2025,
                                         RCT=False, unobserved=True)
        dsets = gen.generate_datasets()
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

### informative censoring
info_censor_baseline=0.1
info_censor_alpha=0.05

with pd.HDFStore("synthetic_data/e_X_info_censor.h5") as store:
    for i in range(1,11):
        if i in [4,5,6,7,10]:
            continue
        dataset_name = f'e_X_info_censor_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                         n_samples=50000, random_state=2025,
                                         RCT=False, unobserved=False, informative_censoring=True,
                                         info_censor_baseline=info_censor_baseline,
                                         info_censor_alpha=info_censor_alpha)
        dsets = gen.generate_datasets() 
        store[f"scenario_{i}/data"] = dsets['data']
        store.get_storer(f"scenario_{i}/data").attrs.metadata = dsets['metadata']

with pd.HDFStore("synthetic_data/e_X_U_info_censor.h5") as store:
    for i in range(1,11):
        if i in [4,5,6,7,10]:
            continue
        dataset_name = f'e_X_U_info_censor_scenario_{i}'
        gen = SyntheticDataGeneratorPlus(scenario=i, dataset_name=dataset_name,
                                         n_samples=50000, random_state=2025,
                                         RCT=False, unobserved=True, informative_censoring=True,
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
    "synthetic_data/RCT_0_5.h5",
    "synthetic_data/RCT_0_05.h5",
    "synthetic_data/e_X.h5",
    "synthetic_data/e_X_U.h5",
    "synthetic_data/e_X_info_censor.h5",
    "synthetic_data/e_X_U_info_censor.h5"
]

records = []
for fname in store_files:
    with pd.HDFStore(fname) as store:
        for scenario in range(1, 11):
            key = f"scenario_{scenario}/data"
            if key in store:
                df = store[key]
                metadata = store.get_storer(key).attrs.metadata
                max_time = df["observed_time"].max()
                censor_rate = 1 - df["event"].mean()
                treat_rate = df["W"].mean()
                n_samples = len(df)
                records.append({
                    "file": fname,
                    "scenario": scenario,
                    "n_samples": n_samples,
                    "max_observed_time": max_time,
                    "censoring_rate": censor_rate,
                    "treatment_rate": treat_rate
                })

summary_df = pd.DataFrame(records)
tools.display_dataframe_to_user("Simulation Summary Metrics", summary_df)
```