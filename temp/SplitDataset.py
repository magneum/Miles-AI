import chardet
import pandas as pd
from colorama import Fore, Style
from sklearn.model_selection import train_test_split


def get_encoding(file_path):
    print(Fore.CYAN + Style.BRIGHT + "Dataset Path: " + file_path)
    print(Style.RESET_ALL)
    try:
        with open(file_path, "rb") as file:
            result = chardet.detect(file.read())
            print(Fore.CYAN + Style.BRIGHT + "Detected encoding: " + result["encoding"])
            print(Style.RESET_ALL)
            return result["encoding"]
    except Exception as e:
        print(
            Fore.RED
            + Style.BRIGHT
            + "Error occurred while detecting encoding: "
            + str(e)
        )
        print(Style.RESET_ALL)
        return None


def split_dataset(dataset_path, test_size=0.2, random_state=42):
    encoding = get_encoding(dataset_path)
    if encoding is None:
        return None, None, None, None

    try:
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
    except Exception as e:
        print(
            Fore.RED + Style.BRIGHT + "Error occurred while reading dataset: " + str(e)
        )
        print(Style.RESET_ALL)
        return None, None, None, None


dataset_path = "corpdata/csv/chatgpt_paraphrases.csv"
X_train, Y_train, X_test, Y_test = split_dataset(dataset_path)

if (
    X_train is not None
    and Y_train is not None
    and X_test is not None
    and Y_test is not None
):
    print(Fore.GREEN + Style.BRIGHT + "X_train: " + str(X_train))
    print(Fore.GREEN + Style.BRIGHT + "Y_train: " + str(Y_train))
    print(Fore.YELLOW + Style.BRIGHT + "X_test: " + str(X_test))
    print(Fore.YELLOW + Style.BRIGHT + "Y_test: " + str(Y_test))
else:
    print(
        Fore.RED
        + Style.BRIGHT
        + "Error occurred while splitting dataset. Please check file path and encoding."
    )

print(Style.RESET_ALL)
