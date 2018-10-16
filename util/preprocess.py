import csv
import os
import string
from os.path import join, dirname
from random import shuffle


def normalize_text(text):
    text = " ".join(i for i in text.split())
    table = str.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    return text.lower()


def load_data(folder):
    data = []
    label = folder.split("/")[-1].lower().replace(" ", "_")
    files = [join(folder, x) for x in os.listdir(folder)][:5]
    for file in files:
        with open(file, "rb") as f:
            content = f.read()
            content = content.decode('utf-16')
        data.append({"text": normalize_text(content), "label": label})
    return data


def convert_to_corpus(name, rows):
    corpus_path = join(path, "corpus", name)
    columns = ["text", "label"]
    with open(corpus_path, "w") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    path = join(dirname(dirname(__file__)), 'data')
    train_folder = [join(path, "raw", "train", i) for i in os.listdir(join(path, "raw", "train"))]
    test_folder = [join(path, "raw", "test", i) for i in os.listdir(join(path, "raw", "test"))]
    train = [i for x in train_folder for i in load_data(x)]
    test = [i for x in test_folder for i in load_data(x)]
    shuffle(train)
    shuffle(test)
    convert_to_corpus("train.csv", train)
    convert_to_corpus("test.csv", test)
