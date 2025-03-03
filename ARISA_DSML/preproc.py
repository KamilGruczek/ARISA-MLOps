import re
import pandas as pd
import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import joblib


def preprocess_data():

    api = KaggleApi()
    api.authenticate()

    dataset = "titanic"
    dataset_test = "wesleyhowe/titanic-labelled-test-set"
    download_folder = Path("data/titanic")
    zip_path = download_folder / "titanic.zip"
    download_folder.mkdir(parents=True, exist_ok=True)

    api.competition_download_files(dataset, path=str(download_folder))
    api.dataset_download_files(dataset_test, path=str(download_folder), unzip=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(str(download_folder))

    os.remove(zip_path)

    df_train = pd.read_csv(download_folder / "train.csv")

    df_train = df_train.drop(columns=["Ticket"])
    df_train["Title"] = df_train["Name"].apply(extract_title)

    pattern = r'([A-Za-z]+)(\d+)'
    matches = df_train['Cabin'].str.extractall(pattern)
    matches.reset_index(inplace=True)
    result = matches.pivot(index='level_0', columns='match', values=[0, 1])
    result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]
    df_train = df_train.join(result[["0_0", "1_0"]])

    df_train["1_0"] = df_train["1_0"].astype(float)
    df_train = df_train.fillna({"0_0": "N", "1_0": df_train["1_0"].mean()})
    df_train["1_0"] = df_train["1_0"].astype(int)
    df_train = df_train.rename(columns={"0_0": "Deck", "1_0": "CabinNumber"})
    df_train.drop(columns=["Cabin", "Name"], axis=1, inplace=True)
    df_train = df_train.fillna({"Embarked": "N", "Age": df_train["Age"].mean()})

    categorical = ["Pclass", "Sex", "Embarked", "Deck", "Title"]
    y_train = df_train.pop("Survived")
    X_train = df_train

    categorical_indices = [X_train.columns.get_loc(col) for col in categorical if col in X_train.columns]

    # Save preprocessed data
    joblib.dump((X_train, y_train, categorical_indices), download_folder / "preprocessed_data.pkl")


def extract_title(name):
    match = re.search(r',\s*([\w\s]+)\.', name)
    return match.group(1) if match else None


def main():
    preprocess_data()


if __name__ == "__main__":
    main()
