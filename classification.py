import argparse

from models import classifier

parser = argparse.ArgumentParser("classification.py")
text = parser.add_argument_group("The following arguments are mandatory for text option")
text.add_argument("text", metavar="TEXT", help="text to predict", nargs="?")
args = parser.parse_args()

if args.text:
    text = args.text
    predict = classifier(text)[0]
    label = predict.replace("_", " ").capitalize()
    print(label)
