import chardet
import pandas as pd
from colorama import Fore, Style
from sklearn.model_selection import train_test_split


def get_encoding(File_path):
    with open(File_path, "rb") as File:
        result = chardet.detect(File.read())
    return result["encoding"]


def split_dataset(dataset_path, test_size=0.2, random_state=42):
    encoding = get_encoding(dataset_path)
    print(Fore.CYAN + Style.BRIGHT + "Detected encoding: " + encoding + Style.RESET_ALL)
    dataset = pd.read_csv(dataset_path, encoding=encoding)
    column_names = dataset.columns.tolist()
    target_label_column_name = column_names[-1]
    input_feature_column_names = column_names[:-1]
    x = dataset[input_feature_column_names].values
    y = dataset[target_label_column_name].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return x_train, y_train, x_test, y_test


dataset_path = "corpdata/csv/IMDB_Dataset.csv"
X_train, Y_train, X_test, Y_test = split_dataset(dataset_path)

print(Fore.GREEN + Style.BRIGHT + "X_train: " + str(X_train))
print(Fore.GREEN + Style.BRIGHT + "Y_train: " + str(Y_train))
print(Fore.YELLOW + Style.BRIGHT + "X_test: " + str(X_test))
print(Fore.YELLOW + Style.BRIGHT + "Y_test: " + str(Y_test))
print(Style.RESET_ALL)
