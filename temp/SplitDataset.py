import pandas as pd
from colorama import Fore, Style
from sklearn.model_selection import train_test_split


def split_dataset(dataset_path, test_size=0.2, random_state=42):
    dataset = pd.read_csv(dataset_path)
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


print(Fore.GREEN + "X_train: " + X_train)
print(Fore.GREEN + "Y_train: " + Y_train)
print(Fore.GREEN + "X_test: " + X_test)
print(Fore.GREEN + "Y_test: " + Y_test)
print(Style.RESET_ALL)
