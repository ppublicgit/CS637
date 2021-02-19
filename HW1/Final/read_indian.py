import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale(df, scaler="MinMax"):
    """Scale the dataframe using the scaler specified and return the newly scaled dataframe
    """
    inputs = df.iloc[:, :-1].to_numpy()
    if scaler == "MinMax":
        scale = MinMaxScaler()
    elif scaler == "Standard":
        scale = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler {scaler}")
    scale.fit(inputs.astype(float))
    inputs = scale.transform(inputs)

    scaled_df = pd.DataFrame(data=inputs)

    scaled_df["target"] = df["target"]

    return scaled_df

def shuffle(df):
    return df.sample(frac=1).reset_index(drop=True)


def ohe(arr, percent=1):
    size = len(np.unique(arr))
    ohe = np.zeros((len(arr), size), dtype=int)
    for i, val in enumerate(arr):
        idx = val - 1
        ohe[i, idx ] = 1
    if percent < 1 and percent > 0:
        subindex = int(percent * ohe.shape[0])
        ohe = ohe[:subindex, :]
    elif percent == 1:
        pass
    else:
        raise ValueError(f"Percent must be set to a value in range (0, 1], not {percent}")
    return ohe


def split(df):
    num = len(df)
    test_split = int(0.75*num)
    val_split = int(0.75*test_split)
    inputs = df.iloc[:, :-1].to_numpy()
    targets = np.array(df["target"].values)
    targets = ohe(targets)
    train = inputs[:test_split]
    train_labels = targets[:test_split]
    val = train[val_split:]
    val_labels = train_labels[val_split:]
    train = train[:val_split]
    train_labels = train_labels[:val_split]
    test = inputs[test_split:]
    test_labels = targets[test_split:]
    return train, train_labels, val, val_labels, test, test_labels


def read_indian():
    indian = loadmat("indian/indianR.mat")
    data = np.array(indian["X"]).T
    targets = np.array(indian["gth"])[0]
    indian_df = pd.DataFrame(data=data)
    indian_df["target"] = targets
    np.unique(indian_df["target"], return_counts=True)
    indexDrop = indian_df[indian_df["target"] == 0].index
    indian_df.drop(indexDrop, inplace=True)
    indian_df.reset_index(inplace=True)
    indian_scaled = scale(indian_df, scaler="MinMax")
    indian_shuffled = shuffle(indian_scaled)
    return split(indian_shuffled)
