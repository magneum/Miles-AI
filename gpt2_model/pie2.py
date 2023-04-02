import pandas as pd
from colorama import Fore as Fr
from colorama import Style as Sr
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


for dataset in [
    "large-762M",
    "large-762M-k40",
    "medium-345M-k40",
    "medium-345M",
    "small-117M-k40",
    "small-117M",
    "xl-1542M-k40",
    "xl-1542M",
]:
    print(f"{Fr.GREEN}DATASET INFO: {dataset} {Sr.RESET_ALL}read_csv()")
    train_data = pd.read_csv(f"corpdata/gpt/{dataset}.train.csv")
    valid_data = pd.read_csv(f"corpdata/gpt/{dataset}.valid.csv")
    test_data = pd.read_csv(f"corpdata/gpt/{dataset}.test.csv")

    print(f"{Fr.GREEN}DATASET INFO: {dataset} {Sr.RESET_ALL}astype(str)")
    train_data["text"] = train_data["text"].astype(str)
    valid_data["text"] = valid_data["text"].astype(str)
    test_data["text"] = test_data["text"].astype(str)

    print(f"{Fr.GREEN}DATASET INFO: {dataset} {Sr.RESET_ALL}Tokenizer()")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data["text"].values)

    print(f"{Fr.GREEN}DATASET INFO: {dataset} {Sr.RESET_ALL}texts_to_sequences()")
    train_sequences = tokenizer.texts_to_sequences(train_data["text"].values)
    valid_sequences = tokenizer.texts_to_sequences(valid_data["text"].values)
    test_sequences = tokenizer.texts_to_sequences(test_data["text"].values)

    print(f"{Fr.GREEN}DATASET INFO: {dataset} {Sr.RESET_ALL}pad_sequences()")
    max_length = 128
    train_inputs = pad_sequences(train_sequences, maxlen=max_length, padding="post")
    valid_inputs = pad_sequences(valid_sequences, maxlen=max_length, padding="post")
    test_inputs = pad_sequences(test_sequences, maxlen=max_length, padding="post")

    train_outputs = train_data["ended"].values
    valid_outputs = valid_data["ended"].values
    test_outputs = test_data["ended"].values

    print(f"{Fr.BLUE}{dataset} TRAIN INPUTS SHAPE:{Sr.RESET_ALL}", train_inputs.shape)
    print(f"{Fr.BLUE}{dataset} VALID INPUTS SHAPE:{Sr.RESET_ALL}", valid_inputs.shape)
    print(f"{Fr.BLUE}{dataset} TEST INPUTS SHAPE:{Sr.RESET_ALL}", test_inputs.shape)
    print(f"{Fr.BLUE}{dataset} TRAIN OUTPUTS SHAPE:{Sr.RESET_ALL}", train_outputs.shape)
    print(f"{Fr.BLUE}{dataset} VALID OUTPUTS SHAPE:{Sr.RESET_ALL}", valid_outputs.shape)
    print(f"{Fr.BLUE}{dataset} TEST OUTPUTS SHAPE:{Sr.RESET_ALL}", test_outputs.shape)
