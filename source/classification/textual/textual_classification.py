# Use PlaidML for GPU acceleration on the MacBook Pro with AMD GPU (OpenCL)
# These lines need to go before the first keras import
from typing import Optional

import plaidml.keras
plaidml.keras.install_backend()

import sklearn
import glob
import itertools
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential, Model
from keras.callbacks import TensorBoard, Callback, ReduceLROnPlateau, LambdaCallback
from keras.layers import Dense, Dropout, concatenate
from keras_preprocessing import text
from keras.optimizers import Adam

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from spacy.lang.de import GermanDefaults
from spacy.lang.en import EnglishDefaults
from spacy.lang.fr import FrenchDefaults
from spacy.lang.it import ItalianDefaults

lerb_specific = True

tensorboard_dir = ".tensorboard"
data_dir = ".data/" + ("classified" if not lerb_specific else "lerb_classified")
out_dir = ".out"
validation_frac = .2
batch_size = 32
num_epochs = 45
verbose = 0


#
# Class that represents the text content of an invoice
#
class Text:
    def __init__(self, t, clazz, filename):
        self.text = t
        self.clazz = clazz
        self.filename = filename


# Pattern that matches any text with 3 or more characters. This pattern is used to filter invoices where there was a
# very bad OCR result
threeCharsPat = re.compile("[a-zéàèöäüç]{3}", re.IGNORECASE)


#
# Function that filters words with less than three chars and stopwords (de, en, fr, it)
#
def word_filter(word: str):
    # require at least 3 characters
    if not bool(threeCharsPat.search(word)):
        return False

    if word in GermanDefaults.stop_words \
            or word in EnglishDefaults.stop_words \
            or word in FrenchDefaults.stop_words \
            or word in ItalianDefaults.stop_words:
        return False

    return True


#
# Read a CSV file produced by tesseract OCR
#
def read_file(path):
    words = pd.read_csv(
        filepath_or_buffer=path,
        sep="\t",
        quotechar="\0"
    )['text']
    words = words[~pd.isnull(words)]
    words = filter(word_filter, words)

    return " ".join(words)


#
# Read all texts produced by tesseract OCR
#
def read_texts():
    for clazz in os.listdir(data_dir):
        if os.path.isdir(data_dir + "/" + clazz):
            print("Loading texts for class " + clazz)

            for file in os.listdir(data_dir + "/" + clazz):
                if file.endswith(".tsv") and os.path.getsize(data_dir + "/" + clazz + "/" + file) > 0:
                    t = read_file(data_dir + "/" + clazz + "/" + file)

                    if t.isspace() or len(t) == 0:
                        print("Skipping %s as it contains no text..." % (data_dir + "/" + clazz + "/" + file,))
                        continue

                    if len(t) < 25:
                        print("%s has length %d..." % (data_dir + "/" + clazz + "/" + file, len(t)))
                        print(">> %s" % (t,))

                    yield Text(t, clazz, data_dir + "/" + clazz + "/" + file)


#
# Keras Callback to print a confusion matrix
#
class ConfusionMatrixCallback(Callback):
    def __init__(self, log_dir, x_val, y_val, label_encoder, classes, cmap=plt.cm.Greens, normalize=False):
        super().__init__()
        self.log_dir = log_dir
        self.X_val = x_val
        self.Y_val = y_val
        self.label_encoder = label_encoder
        self.classes = classes
        self.cmap = cmap
        self.normalize = normalize

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        super(ConfusionMatrixCallback, self).on_epoch_end(epoch, logs)

        plt.clf()
        pred = self.model.predict(self.X_val)
        cnf_mat = confusion_matrix(self.Y_val, self.label_encoder.inverse_transform(pred))

        if self.normalize:
            cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

        thresh = cnf_mat.max() / 2.
        for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
            plt.text(j, i, cnf_mat[i, j],
                     horizontalalignment="center",
                     color="white" if cnf_mat[i, j] > thresh else "black")

        plt.imshow(cnf_mat, interpolation='nearest', cmap=self.cmap)

        # Labels
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        plt.colorbar()

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.plot()

        plt.savefig("%s/cm_%d.png" % (self.log_dir, epoch))


#
# Load existing texts from .data/ocr_texts.py or assemble them from the invoices
#
def load_data():
    text_file = data_dir + "/ocr_texts.npy"

    if os.path.exists(text_file):
        print("text_file detected. Loading from disk...")
        data_list = np.load(text_file)
    else:
        data_list = [x for x in map(lambda t: [t.text, t.clazz, t.filename], read_texts())]
        np.save(text_file, data_list)

    print("Test data set count:")
    print(len(data_list))

    return shuffle(pd.DataFrame(data_list, columns=['text', 'clazz', 'filename']))


#
# Build the text classification model
#
def build_model(vocab_size, dropout, num_classes, learning_rate):
    model = Sequential()
    model.add(Dense(
        512,  # 512 neurons
        input_shape=(vocab_size,),
        activation="relu"
    ))

    if dropout is not None:
        model.add(Dropout(dropout))

    model.add(Dense(
        num_classes,
        activation="softmax"
    ))

    optimizer = Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


#
# Run the model with a specific configuration
#
def run(vocab_size, dropout=None, print_cm=False, learning_rate=1e-3):
    data = load_data()

    run_with_data(data, vocab_size, dropout, print_cm, learning_rate)


#
# Run n folds of the model as an ensemble
#
def run_with_folds(vocab_size, dropout, learning_rate=1e-3, print_cm=False, n_folds: Optional[int]=5, tensorboard_report_folds=False):
    data = load_data()

    ensemble_meta_frac = 0.3
    ensemble_cv_size = int(len(data['text']) * (1 - ensemble_meta_frac))
    # reindex is needed as otherwise the list will keep weird indexes and split will yield NaNs
    ensemble_cv_texts = np.array(data['text'][:ensemble_cv_size])
    ensemble_cv_classes = np.array(data['clazz'][:ensemble_cv_size])
    ensemble_meta_texts = np.array(data['text'][ensemble_cv_size:])
    ensemble_meta_classes = np.array(data['clazz'][ensemble_cv_size:])

    models = []

    label_encoder = LabelBinarizer()
    label_encoder.fit(ensemble_cv_classes)

    tokenize = text.Tokenizer(num_words=vocab_size)
    tokenize.fit_on_texts(ensemble_cv_texts)

    skf = StratifiedKFold(n_folds)
    fold = 1
    for train_index, test_index in skf.split(ensemble_cv_texts, ensemble_cv_classes):
        fold += 1
        model = run_with_train_and_test_data_label_encoder(
            ensemble_cv_texts[train_index],
            ensemble_cv_classes[train_index],
            ensemble_cv_texts[test_index],
            ensemble_cv_classes[test_index],
            tokenize=tokenize,
            vocab_size=vocab_size,
            label_encoder=label_encoder,
            dropout=dropout,
            print_cm=print_cm,
            fold=fold,
            n_folds=n_folds,
            learning_rate=learning_rate,
            tensorboard_report=tensorboard_report_folds
        )

        models.append(model)

    for i in range(len(models)):
        model = models[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name

    # define multi-headed input
    ensemble_inputs = [model.input for model in models]
    ensemble_outputs = [model.output for model in models]

    dropouts = (0.3, 0.5, 0.75, None)  # (0.97, 0.95, 0.92, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.3, 0.2, None)
    for ensemble_dropout1 in dropouts:  # (0.97, 0.95, 0.92, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.3, 0.2, None):
        for ensemble_dropout2 in dropouts:  # (0.97, 0.95, 0.92, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.3, 0.2, None):
            merge = concatenate(ensemble_outputs)
            if ensemble_dropout1 is not None:
                merge = Dropout(ensemble_dropout1)(merge)
            hidden = Dense(
                int((n_folds * len(label_encoder.classes_) + len(label_encoder.classes_)) / 1.5),
                activation='relu',
            )(merge)
            if ensemble_dropout2 is not None:
                hidden = Dropout(ensemble_dropout2)(hidden)
            # TODO: Experiment with hidden layer depth
            output = Dense(len(label_encoder.classes_), activation='softmax')(hidden)
            ensemble = Model(inputs=ensemble_inputs, outputs=output)
            ensemble.name = 'Ensemble'
            optimizer = Adam(lr=learning_rate)
            ensemble.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            # plot_model(ensemble, to_file=".out_new/ensemble_%d_of_%r_%r.png" % (n_folds, vocab_size, dropout),
            # show_shapes=True)

            # Train the ensemble
            train_size = int(len(ensemble_meta_texts) * (1 - validation_frac))
            train_texts = ensemble_meta_texts[:train_size]
            train_classes = ensemble_meta_classes[:train_size]
            test_texts = ensemble_meta_texts[train_size:]
            test_classes = ensemble_meta_classes[train_size:]
            x_train = tokenize.texts_to_matrix(train_texts)
            y_train = label_encoder.transform(train_classes)
            x_test = tokenize.texts_to_matrix(test_texts)
            y_test = label_encoder.transform(test_classes)
            class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', label_encoder.classes_,
                                                                           train_classes)
            callbacks = [
                # EarlyStopping(monitor='val_acc', patience=8),
                # ModelCheckpoint(out_dir + "/models/" + ensemble.name + "_acc{val_acc:.2f}.h5", save_best_only=True),
                TensorBoard(
                    log_dir="%s___WordBag ensemble of %d vocab_size=%d/dropout=%r e1=%r e2=%r lr=%f" % (
                        tensorboard_dir, n_folds, vocab_size, dropout, ensemble_dropout1, ensemble_dropout2,
                        learning_rate
                    ),
                ),
            ]
            x = [x_train for _ in range(len(ensemble.input))]
            ensemble.fit(x, y_train,
                         batch_size=batch_size,
                         epochs=num_epochs,
                         verbose=verbose,
                         validation_data=([x_test for _ in range(len(ensemble.input))], y_test),
                         class_weight=class_weight,
                         callbacks=callbacks)


#
# Low level function to run the model with assembled data
#
def run_with_data(data, vocab_size, dropout=None, print_cm=False, learning_rate=1e-3):
    texts = data['text']
    labels = data['clazz']
    filenames = data['filename']
    train_size = int(len(texts) * (1 - validation_frac))
    train_texts = texts[:train_size]
    train_classes = labels[:train_size]
    train_filenames = filenames[:train_size]
    test_texts = texts[train_size:]
    test_classes = labels[train_size:]
    test_filenames = filenames[train_size:]

    run_with_train_and_test_data(train_texts, train_classes, test_texts, test_classes, vocab_size, dropout, print_cm,
                                 learning_rate, test_filenames, train_filenames)


#
# Low level function to run the model with assembled data split into train and test
#
def run_with_train_and_test_data(train_texts, train_classes, test_texts, test_classes, vocab_size, dropout=None,
                                 print_cm=False, learning_rate=1e-3, test_filenames=None, train_filenames=None):
    tokenize = text.Tokenizer(num_words=vocab_size)
    tokenize.fit_on_texts(train_texts)

    label_encoder = LabelBinarizer()
    label_encoder.fit(train_classes)

    tokenize = text.Tokenizer(num_words=vocab_size)
    tokenize.fit_on_texts(train_texts)
    # print(tokenize.index_word)
    # print(tokenize.texts_to_matrix(["123"]))
    # print(len(tokenize.texts_to_matrix(["123"])[0]))
    #
    # with open("vocab_%d.csv" % (vocab_size,), 'w') as f:
    #     for key in tokenize.word_index.keys():
    #         f.write("%s,%s\n" % (key, tokenize.word_index[key]))

    return run_with_train_and_test_data_label_encoder(train_texts=train_texts,
                                                      train_classes=train_classes,
                                                      test_texts=test_texts,
                                                      test_classes=test_classes,
                                                      vocab_size=vocab_size,
                                                      tokenize=tokenize,
                                                      label_encoder=label_encoder,
                                                      dropout=dropout,
                                                      print_cm=print_cm,
                                                      learning_rate=learning_rate,
                                                      test_filenames=test_filenames,
                                                      train_filenames=train_filenames)


#
# Low level function to run the model with assembled data split into train and test and a trained label encoder
#
def run_with_train_and_test_data_label_encoder(train_texts, train_classes, test_texts, test_classes, vocab_size,
                                               tokenize, label_encoder, dropout=None, print_cm=False,
                                               learning_rate=1e-3,
                                               tensorboard_report=True, fold=1, n_folds=1, test_filenames=None,
                                               train_filenames=None):
    model = build_model(vocab_size=vocab_size, dropout=dropout, num_classes=len(label_encoder.classes_),
                        learning_rate=learning_rate)

    # model.summary()

    x_train = tokenize.texts_to_matrix(train_texts)
    y_train = label_encoder.transform(train_classes)
    x_test = tokenize.texts_to_matrix(test_texts)
    y_test = label_encoder.transform(test_classes)

    class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', label_encoder.classes_, train_classes)

    log_dir = "%s/WordBag vocab_size=%d dropout=%r fold=%d_%d lr=%f" % (tensorboard_dir, vocab_size, dropout, fold,
                                                                        n_folds, learning_rate)

    def on_epoch_end(epoch, logs):
        analysis_threshold = 0  # 0.95
        # print(logs)
        # print(epoch)
        predictions = model.predict(x_test)
        predictions_label = label_encoder.inverse_transform(predictions)
        for num in range(len(predictions)):
            prediction_label = predictions_label[num]
            prediction_encoded = predictions[num]
            actual = test_classes.values[num]
            filename = test_filenames.values[num]

            if prediction_label != actual:
                # if prediction_label != "unknown":
                if any(list(map(lambda x: x > analysis_threshold, prediction_encoded))):
                    # known_issues.append(filename)
                    print("Epoch %2d: Expected %10s but predicted %10s for %s / %r" % (epoch, actual, prediction_label,
                                                                                       filename, prediction_encoded))
                    # print(sum(prediction_encoded), any(list(map(lambda x: x > 0.5, prediction_encoded))))
                    # print("./reclass.sh %s %s" % (filename.replace(".png.tsv", ""), prediction_label))

        predictions = model.predict(x_train)
        predictions_label = label_encoder.inverse_transform(predictions)
        for num in range(len(predictions)):
            prediction_label = predictions_label[num]
            prediction_encoded = predictions[num]
            actual = train_classes.values[num]
            filename = train_filenames.values[num]

            if prediction_label != actual:
                # if prediction_label != "unknown":
                if any(list(map(lambda x: x > analysis_threshold, prediction_encoded))):
                    # known_issues.append(filename)
                    print("Epoch %2d: Expected %10s but predicted %10s for %s / %r" % (epoch, actual, prediction_label,
                                                                                       filename, prediction_encoded))

    os.makedirs(out_dir + "/models/", exist_ok=True)
    callbacks = [
        ReduceLROnPlateau(monitor="val_los", patience=1, min_delta=1e-3),
        LambdaCallback(
            on_epoch_end=on_epoch_end
        )
        # EarlyStopping(monitor='val_acc', patience=8),
        # ModelCheckpoint(out_dir + "/models/" + model.name + "_acc{val_acc:.2f}.h5", save_best_only=True),
    ]

    if tensorboard_report:
        callbacks.append(
            TensorBoard(
                log_dir=log_dir,
            ),
        )

    if print_cm:
        callbacks.append(ConfusionMatrixCallback(
            log_dir=log_dir,
            x_val=x_test,
            y_val=test_classes,
            classes=label_encoder.classes_,
            label_encoder=label_encoder,
        ))

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              verbose=verbose,
              validation_data=(x_test, y_test),
              class_weight=class_weight,
              callbacks=callbacks)

    # x_test = tokenize.texts_to_matrix(test_texts)
    # y_test = label_encoder.transform(test_classes)

    # score = model.evaluate(x_test, y_test,
    #                       batch_size=batch_size, verbose=1)

    # print(model.metrics_names)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

    #
    #
    # print(read_file(".data_new/classified/fitness/5a6f13de0cf2ab51c3ec3cfd.pdf.png.tsv"))
    # print(read_file(".data_new/classified/optician/5a6b334a0cf2ab51c3ec3c99.pdf.png.tsv"))
    # print(read_file(".data_new/classified/sportsclub/5a5f2df60cf2ab51c3ec2ff5.pdf.png.tsv"))
    # print(read_file(".data_new/classified/unknown/5bd41deb0cf239efb23e8b01.pdf.png.tsv"))

    # text_matrices = tokenize.texts_to_matrix([
    #     read_file(".data_new/classified/fitness/5a6f13de0cf2ab51c3ec3cfd.pdf.png.tsv"),
    #     read_file(".data_new/classified/optician/5a6b334a0cf2ab51c3ec3c99.pdf.png.tsv"),
    #     read_file(".data_new/classified/sportsclub/5a5f2df60cf2ab51c3ec2ff5.pdf.png.tsv"),
    #     read_file(".data_new/classified/unknown/5bd41deb0cf239efb23e8b01.pdf.png.tsv"),
    # ])
    # prediction = model.predict(text_matrices)

    # print(prediction)
    # print(label_encoder.inverse_transform(prediction))

    return model


delete_old_boards = False

if delete_old_boards:
    for d in glob.glob(tensorboard_dir + "/WordBag*"):
        for root, dirs, files in os.walk(d, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(d)

# run_with_folds(
#     vocab_size=1000,
#     dropout=0.5,
#     n_folds=2,
# )
# run_with_folds(
#     vocab_size=1500,
#     dropout=0.5,
#     n_folds=2,
# )
# run_with_folds(
#     vocab_size=1500,
#     dropout=0.5,
#     n_folds=5,
# )
# run_with_folds(
#     vocab_size=1500,
#     dropout=0.8,
#     n_folds=5,
# )
# run_with_folds(
#     vocab_size=1500,
#     dropout=0.8,
#     n_folds=5,
# )

print_confusion_matrix = False

# run_with_folds(
#    vocab_size=5000,
#    dropout=0.95,
#    n_folds=10,
#    print_cm=print_confusion_matrix,
#    learning_rate=0.0005,
#    ensemble_dropout=(0.7, 0.7),
# )
# run(
#    vocab_size=5000,
#    dropout=0.95,
#    print_cm=print_confusion_matrix,
#    learning_rate=0.0005
# )
#
#
# exit(1)


# model = build_model(100, num_classes=4, dropout=0.5, learning_rate=1e-3)
#
# f = open(".out_new/textual_classifier.svg", "wb")
# f.write(model_to_dot(model).create(prog='dot', format='svg'))
#
# exit(1)


run(
    vocab_size=5000,
    dropout=0.92,
    learning_rate=1e-4,
    print_cm=True
)

print("done")

exit(0)


def run_proc(vocab_size):
    # (None, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9):
    for dropout in (0.97, 0.95, 0.92, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.3, 0.2, None):
        for n_folds in (None,):  # , 5, 10, 15, 20):  # (None, 5, 10, 15, 20):
            if n_folds is None:
                print("run %r %r" % (vocab_size, dropout))
                run(
                    vocab_size=vocab_size,
                    dropout=dropout,
                    print_cm=print_confusion_matrix,
                )
            else:
                print("run with folds %r %r %r" % (vocab_size, dropout, n_folds))
                run_with_folds(
                    vocab_size=vocab_size,
                    dropout=dropout,
                    n_folds=n_folds,
                    print_cm=print_confusion_matrix,
                )

procs=[]

for vs in (200, 300, 500, 750, 1000, 1500, 2000, 3000, 3500, 5000):
    run_proc(vocab_size=vs)
    # proc = Process(target=run_proc, args=(vocab_size,))
    # proc.start()
    # procs.append(proc)

for proc in procs:
    proc.join()

