import pickle

from keras_preprocessing import sequence
from keras.models import model_from_yaml


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def classifier(text):
    vocab_map = load_obj("models/dictionary.pkl")
    y_transformer = load_obj("models/y_transformer.pkl")
    with open("models/model.yaml") as f:
        model = model_from_yaml(f.read())
    X = [[vocab_map[token] if vocab_map.get(token) else vocab_map["NOT_FOUND_INDEX"]
          for token in tokens] for tokens in [x.split() for x in text]]
    X = sequence.pad_sequences(X, maxlen=model.input_shape[1])
    pred = model.predict_classes(X)
    return y_transformer.inverse_transform(pred)
