# train_data = pd.read_csv("corpdata/gpt/large-762M-k40.train.csv")
# valid_data = pd.read_csv("corpdata/gpt/large-762M-k40.valid.csv")
# test_data = pd.read_csv("corpdata/gpt/large-762M-k40.test.csv")


# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(train_data["text"].values)

# train_sequences = tokenizer.texts_to_sequences(train_data["text"].values)
# valid_sequences = tokenizer.texts_to_sequences(valid_data["text"].values)
# test_sequences = tokenizer.texts_to_sequences(test_data["text"].values)

# max_length = 128
# train_inputs = pad_sequences(train_sequences, maxlen=max_length, padding="post")
# train_outputs = train_data["ended"].values
# valid_inputs = pad_sequences(valid_sequences, maxlen=max_length, padding="post")
# valid_outputs = valid_data["ended"].values
# test_inputs = pad_sequences(test_sequences, maxlen=max_length, padding="post")
# test_outputs = test_data["ended"].values


# for dataset in datasets:
#     train_data = pd.read_csv(f"corpdata/gpt/{dataset}.train.csv")
#     valid_data = pd.read_csv(f"corpdata/gpt/{dataset}.valid.csv")
#     test_data = pd.read_csv(f"corpdata/gpt/{dataset}.test.csv")
#     print(f"{dataset} train data columns:", train_data.columns)
#     print(f"{dataset} valid data columns:", valid_data.columns)
#     print(f"{dataset} test data columns:", test_data.columns)

#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(train_data["text"].values)

#     train_sequences = tokenizer.texts_to_sequences(train_data["text"].values)
#     valid_sequences = tokenizer.texts_to_sequences(valid_data["text"].values)
#     test_sequences = tokenizer.texts_to_sequences(test_data["text"].values)
#     print(f"{dataset} train sequences:", train_sequences)
#     print(f"{dataset} valid sequences:", valid_sequences)
#     print(f"{dataset} test sequences:", test_sequences)

#     max_length = 128
#     train_inputs = pad_sequences(train_sequences, maxlen=max_length, padding="post")
#     valid_inputs = pad_sequences(valid_sequences, maxlen=max_length, padding="post")
#     test_inputs = pad_sequences(test_sequences, maxlen=max_length, padding="post")
#     print(f"{dataset} valid inputs shape:", valid_inputs.shape)
#     print(f"{dataset} train inputs shape:", train_inputs.shape)
#     print(f"{dataset} test inputs shape:", test_inputs.shape)

#     train_outputs = train_data["ended"].values
#     valid_outputs = valid_data["ended"].values
#     test_outputs = test_data["ended"].values
#     print(f"{dataset} train outputs shape:", train_outputs.shape)
#     print(f"{dataset} valid outputs shape:", valid_outputs.shape)
#     print(f"{dataset} test outputs shape:", test_outputs.shape)
