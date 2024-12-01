from enum import Enum

import numpy as np
import pandas as pd
from pycox.datasets import metabric
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ospkg.constants import DATA_DIR


class Dataset(Enum):
    SNMMI = "snmmi"
    SNMMI_PCA = "snmmi_pca"
    SNMMI_GAUSS = "snmmi_gauss"
    PBC = "pbc"
    FRAMINGHAM = "framingham"
    SUPPORT = "support"
    METABRIC = "metabric"


def load_os_dataset(dataset: str, **kwargs) -> tuple:
    dataset = Dataset(dataset.lower())
    if dataset == Dataset.SNMMI:
        dataset_loader_func = load_snmmi_challenge
    elif dataset == Dataset.SNMMI_PCA:
        dataset_loader_func = load_snmmi_challenge_pca
    elif dataset == Dataset.SNMMI_GAUSS:
        dataset_loader_func = load_snmmi_challenge_gauss
    elif dataset == Dataset.FRAMINGHAM:
        dataset_loader_func = load_pbc_baseline
    elif dataset == Dataset.PBC:
        dataset_loader_func = load_pbc_baseline
    elif dataset == Dataset.SUPPORT:
        dataset_loader_func = load_support_baseline
    elif dataset == Dataset.METABRIC:
        dataset_loader_func = load_pycox_metabric
    else:
        raise ValueError("Unknown dataset")

    features, duration, event = dataset_loader_func(**kwargs)
    return features.astype(np.float32), duration, event


def load_snmmi_challenge(is_train=True):
    if is_train:
        filename = "SNMMI_CHALLENGE_TRAINING_V22OCT2023.xlsx"
    else:
        filename = "SNMMI_CHALLENGE_TESTING_V01112023.xlsx"
    df_raw = pd.read_excel(DATA_DIR / filename, index_col=0, na_values=0)
    cols_num = [col for col in df_raw.columns if col not in ("Event", "Outcome_PFS")]
    x = run_simple_preprocessing(df_raw, cols_cat=[], cols_num=cols_num)
    x.index = df_raw.index
    y, outcomes = None, None
    if is_train:
        y = df_raw.Outcome_PFS.values
        outcomes = df_raw.Event.values
    return x, y, outcomes


def load_snmmi_challenge_pca(is_train=True):
    scaler, reducer = StandardScaler(), PCA(n_components=0.999, random_state=42)
    x, y, outcomes = load_snmmi_challenge(is_train=True)
    x_scaled = scaler.fit_transform(x.values)
    reducer.fit(x_scaled)

    if not is_train:
        x, y, outcomes = load_snmmi_challenge(is_train=is_train)

    x_reduced = reducer.transform(scaler.transform(x.values))
    x_reduced = pd.DataFrame(
        x_reduced, index=x.index, columns=[f"PC{i}" for i in range(x_reduced.shape[1])]
    )
    return x_reduced, y, outcomes


def load_snmmi_challenge_gauss(is_train=True):
    x, y, outcomes = load_snmmi_challenge(is_train=is_train)

    def _values_to_gauss(values: np.ndarray) -> np.ndarray:
        ranks = values.argsort().argsort()
        gauss = np.random.normal(size=len(ranks))
        gauss.sort()
        return gauss[ranks]

    x = x.apply(_values_to_gauss, axis=0)
    return x, y, outcomes


def load_framingham_baseline():
    """Patient's data from the moment their observation started."""
    cols_cat = [
        "SEX",
        "CURSMOKE",
        "DIABETES",
        "BPMEDS",
        "educ",
        "PREVCHD",
        "PREVAP",
        "PREVMI",
        "PREVSTRK",
        "PREVHYP",
    ]
    cols_num = ["TOTCHOL", "AGE", "SYSBP", "DIABP", "CIGPDAY", "BMI", "HEARTRTE", "GLUCOSE"]
    df_raw = pd.read_csv(DATA_DIR / "framingham.csv")
    df_raw = df_raw.loc[df_raw.PERIOD == 1]
    x = run_simple_preprocessing(df_raw, cols_cat, cols_num)
    y = df_raw["TIMEDTH"].values
    outcomes = df_raw.DEATH.values

    return x, y, outcomes


def load_pbc_baseline():
    """Patient's data from the moment their observation started."""
    cols_cat = ["drug", "sex", "ascites", "hepatomegaly", "spiders", "edema", "histologic"]
    cols_num = [
        "serBilir",
        "serChol",
        "albumin",
        "alkaline",
        "SGOT",
        "platelets",
        "prothrombin",
        "age",
    ]

    df_raw = pd.read_csv(DATA_DIR / "pbc2.csv")
    df_raw = df_raw.loc[df_raw.year == 0]
    x = run_simple_preprocessing(df_raw, cols_cat, cols_num)
    y = df_raw.years.values
    outcomes = df_raw.status2.values

    return x, y, outcomes


def load_support_baseline():
    """Patient's data from the moment their observation started.

    This dataset does not contain followup datapoints.
    """
    cols_cat = ["sex", "dzgroup", "dzclass", "income", "race", "ca"]
    cols_num = [
        "age",
        "num.co",
        "meanbp",
        "wblc",
        "hrt",
        "resp",
        "temp",
        "pafi",
        "alb",
        "bili",
        "crea",
        "sod",
        "ph",
        "glucose",
        "bun",
        "urine",
        "adlp",
        "adls",
    ]
    df_raw = pd.read_csv(DATA_DIR / "support2.csv")
    x = run_simple_preprocessing(df_raw, cols_cat, cols_num)
    y = df_raw["d.time"].values
    outcomes = df_raw.death.values

    return x, y, outcomes


def load_pycox_metabric():
    cols_cat = ["x4", "x5", "x6", "x7"]
    cols_num = ["x0", "x1", "x2", "x3", "x8"]
    df_raw = metabric.read_df()
    x = run_simple_preprocessing(df_raw, cols_cat, cols_num)
    y = df_raw["duration"].values
    outcomes = df_raw["event"].values

    return x, y, outcomes


def run_simple_preprocessing(df_raw, cols_cat, cols_num) -> pd.DataFrame:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", drop="if_binary")
    simple_imputer = SimpleImputer(strategy="median")

    preprocessor = ColumnTransformer(
        transformers=[("cat", one_hot_encoder, cols_cat), ("num", simple_imputer, cols_num)],
        verbose_feature_names_out=False,
    )

    df = pd.DataFrame(preprocessor.fit_transform(df_raw))
    df.columns = preprocessor.get_feature_names_out()
    return df
