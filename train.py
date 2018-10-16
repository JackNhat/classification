import pandas as pd
import numpy as np
from keras.layers import Dense, Embedding, SpatialDropout1D
from keras.models import Sequential
from keras.layers import LSTM
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from collections import Counter

from underthesea import word_tokenize

from util.model_evaluation import get_metrics


def normalize(text):
    output = text.replace("\n", "")
    output = output.lower().strip()
    return output


if __name__ == '__main__':
    # DATA
    train = pd.read_csv("data/corpus/train.csv")
    X_train = np.array(train["text"])
    y_train = np.array(train["label"])
    labels = list(set(y_train.tolist()))

    test = pd.read_csv("data/corpus/test.csv")
    X_test = np.array(test["text"])
    y_test = np.array(test["label"])

    norm_train_texts = [normalize(i) for i in X_train]
    norm_test_texts = [normalize(i) for i in X_test]

    tokenized_train = [word_tokenize(text) for text in norm_train_texts]
    tokenized_test = [word_tokenize(text) for text in norm_test_texts]

    # FEATURE ENGINEERING
    le = LabelEncoder()
    num_classes = 2
    max_len = np.max([len(review) for review in tokenized_train])

    token_counter = Counter([token for review in tokenized_train for token in review])
    vocab_map = {item[0]: index + 1 for index, item in enumerate(dict(token_counter).items())}
    max_index = np.max(list(vocab_map.values()))
    vocab_map["PAD_INDEX"] = 0
    vocab_map["NOT_FOUND_INDEX"] = max_index + 1
    vocab_size = len(vocab_map)

    # Train reviews data corpus
    train_X = [[vocab_map[token] for token in tokenized_review]
               for tokenized_review in tokenized_train]
    train_X = sequence.pad_sequences(train_X, maxlen=max_len)

    # Train prediction class labels
    train_y = le.fit_transform(y_train)

    # Test reviews data corpus
    test_X = [[vocab_map[token] if vocab_map.get(token) else vocab_map["NOT_FOUND_INDEX"]
               for token in tokenized_review]
              for tokenized_review in tokenized_test]
    test_X = sequence.pad_sequences(test_X, maxlen=max_len)
    # Test prediction class labels
    # Convert text sentiments labels to binary encoding
    test_y = le.fit_transform(y_test)

    # TRAINING MODEL
    EMBEDDING_DIM = 128
    LSTM_DIM = 64

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
                        input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(LSTM_DIM, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(len(labels), activation="sigmoid"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    batch_size = 128
    model.fit(train_X, train_y, epochs=5, batch_size=batch_size,
              shuffle=True, validation_split=0.1, verbose=1)

    #  EVALUATE
    pred_test = model.predict_classes(test_X)
    predictions = le.inverse_transform(pred_test)
    get_metrics(test_y, pred_test)

    # SAVE MODEL
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as m:
        m.write(model_yaml)
