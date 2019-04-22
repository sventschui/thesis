import plaidml.keras

plaidml.keras.install_backend()

import glob
import itertools
import tensorflow as tf
import tensorflow_hub as hub
import sklearn.utils
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow.keras.backend as K

from keras.utils.vis_utils import model_to_dot
from keras.applications import inception_resnet_v2, resnet50
from keras import layers
from keras import optimizers, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

image_size = (400, 300)  # height, width
input_size = (image_size[0], image_size[1], 3)

val_frac = 0.2
batch_size = 32
num_epochs = 100

seed = 13371

print_cm = True

lerb_specific = True
tensorboard_dir = ".tensorboard"
data_dir = ".data/" + ("classified" if not lerb_specific else "lerb_classified")
out_dir = ".out"


#
# Keras Callback to print confusion matrix
#
class ConfusionMatrixCallback(Callback):
    def __init__(self, log_dir, X_val, Y_val, label_encoder, classes, cmap=plt.cm.Greens, normalize=False):
        super().__init__()
        self.log_dir = log_dir
        self.X_val = X_val
        self.Y_val = Y_val
        self.label_encoder = label_encoder
        self.classes = classes
        self.cmap = cmap
        self.normalize = normalize

    def on_epoch_end(self, epoch, logs={}):
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


def create_image_generator_flow(datagen, subset=None, target_size=image_size):
    return datagen.flow_from_directory(
        data_dir,
        shuffle=False,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset
    )


def resnet50_pretrained_model_split(generator, class_labels, steps_per_epoch):
    feature_file = "%s/resnet50_%d_%d.npz" % (data_dir, image_size[0], image_size[1])

    if os.path.exists(feature_file):
        print("ConvNet features detected. Loading from disk... %s" % feature_file)
        conv_features = np.load(feature_file)['features']
    else:
        conv_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_size)
        for layer in conv_model.layers:
            layer.trainable = False

        print("ConvNet features NOT detected. Computing. This may take a while...")
        conv_features = conv_model.predict_generator(generator, steps=steps_per_epoch, verbose=True)
        print("Saving features to disk")
        np.savez(feature_file, features=conv_features)

    print("DONE. Shape: ", conv_features.shape)

    return train_test_split(conv_features, class_labels, test_size=val_frac, random_state=seed)


def inceptionresnetv2_pretrained_model_split(generator, class_labels, steps_per_epoch):
    feature_file = data_dir + "/inceptionresnetv2.npz"

    conv_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_size)
    for layer in conv_model.layers:
        layer.trainable = False

    if os.path.exists(feature_file):
        print("ConvNet features detected. Loading from disk...")
        conv_features = np.load(feature_file)['features']
    else:
        print("ConvNet features NOT detected. Computing. This may take a while...")
        conv_features = conv_model.predict_generator(generator, steps=steps_per_epoch, verbose=True)
        print("Saving features to disk")
        np.savez(feature_file, features=conv_features)

    print("DONE. Shape: ", conv_features.shape)

    return train_test_split(conv_features, class_labels, test_size=val_frac, random_state=seed)


def build_model(model_name, num_classes, optimizer, base_model=None, input_shape=None, dilation_rate=1, is1d=False):
    model = Sequential()
    model.name = model_name

    if base_model is not None:
        model.add(base_model)

    if not is1d:
        if input_shape is not None:
            model.add(Conv2D(
                filters=64,
                kernel_size=(1, 1),
                input_shape=input_shape,
                dilation_rate=dilation_rate,
            ))
        else:
            model.add(Conv2D(
                filters=64,
                kernel_size=(1, 1),
                dilation_rate=dilation_rate,
            ))

        model.add(Flatten())

    model.add(Dense(
        units=256,
        activation='relu',
    ))
    model.add(Dropout(
        rate=0.6,
    ))
    model.add(Dense(
        units=32,
        activation='relu',
    ))
    model.add(Dropout(
        rate=0.45,
    ))
    model.add(Dense(
        units=num_classes,
        activation='softmax',
    ))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', 'mae'])

    f = open(out_dir + "/%s.svg" % (model_name,), "wb")
    f.write(model_to_dot(model).create(prog='dot', format='svg'))
    # plot_model(model, show_layer_names=False, to_file=out_dir+"/%s.png" % (model_name,))

    return model


nasnet_feature_extractor_url = "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1"


def nasnet_feature_extractor(x):
    feature_extractor_module = hub.Module(nasnet_feature_extractor_url)
    return feature_extractor_module(x)


def nasnet_pretrained_model_split(generator, class_labels, steps_per_epoch, target_size):
    feature_file = "%s/nasnet_%d_%d.npz" % (data_dir, target_size[0], target_size[1])

    if os.path.exists(feature_file):
        print("ConvNet features detected. Loading from disk... %s" % feature_file)
        conv_features = np.load(feature_file)['features']
    else:
        features_extractor_layer = layers.Lambda(nasnet_feature_extractor, input_shape=image_size + [3])

        features_extractor_layer.trainable = False

        conv_model = Sequential([features_extractor_layer])
        conv_model.summary()

        sess = K.get_session()
        init = tf.global_variables_initializer()

        sess.run(init)

        for layer in conv_model.layers:
            layer.trainable = False

        print("ConvNet features NOT detected. Computing. This may take a while...")
        conv_features = conv_model.predict_generator(generator, steps=steps_per_epoch, verbose=True)
        print("Saving features to disk")
        np.savez(feature_file, features=conv_features)

    print("DONE. Shape: ", conv_features.shape)

    return train_test_split(conv_features, class_labels, test_size=val_frac, random_state=seed)


def traing_nasnet_pretrained_model(dilation_rate=1):
    target_size = hub.get_expected_image_size(hub.Module(nasnet_feature_extractor_url))

    generator = create_image_generator_flow(
        ImageDataGenerator(
        ),
        subset='training',
        target_size=target_size,
    )

    steps_per_epoch = generator.samples / batch_size
    classes = generator.class_indices
    num_classes = len(classes)
    class_labels = generator.classes

    # Calculate class weights to handle unbalanced classes
    class_indices = list(generator.class_indices.values())
    class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', class_indices, class_labels)

    print("target_size %r" % (target_size,))

    (X_train, X_validation, Y_train, Y_validation) = nasnet_pretrained_model_split(
        generator,
        class_labels,
        steps_per_epoch,
        target_size=target_size,
    )

    optimizer = optimizers.Adam(lr=1e-3)
    model = build_model("nasnet-pretrained", num_classes, optimizer, dilation_rate=dilation_rate, is1d=True)

    model.build(input_shape=X_train.shape)
    model.summary()
    exit(0)

    callbacks = [
        # EarlyStopping(monitor='val_acc', patience=8),
        ModelCheckpoint(out_dir + "/models/" + model.name + "_acc{val_acc:.2f}.h5", save_best_only=True),
        TensorBoard(
            log_dir="%s/Img NASNet dilat=%d" % (tensorboard_dir, dilation_rate),
            write_images=True,
        )
    ]

    os.makedirs(out_dir + "/models", exist_ok=True)

    label_encoder = LabelBinarizer()
    label_encoder.fit(class_labels)

    model.fit(
        X_train, label_encoder.transform(Y_train),
        validation_data=(X_validation, label_encoder.transform(Y_validation)),
        class_weight=class_weight,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=True
    )


def train_resnet50_pretrained_model(dilation_rate=1, optimizer: optimizers.Optimizer = optimizers.Adam(lr=1e-3),
                                    model_name=""):
    generator = create_image_generator_flow(
        ImageDataGenerator(
            preprocessing_function=resnet50.preprocess_input,
        ),
        subset='training',
    )

    steps_per_epoch = generator.samples / batch_size
    classes = generator.class_indices
    num_classes = len(classes)
    class_labels = generator.classes

    # Calculate class weights to handle unbalanced classes
    class_indices = list(generator.class_indices.values())
    class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', class_indices, class_labels)

    (X_train, X_validation, Y_train, Y_validation) = resnet50_pretrained_model_split(
        generator,
        class_labels,
        steps_per_epoch,
    )

    model = build_model("resnet50-pretrained%s" % (model_name,), num_classes, optimizer, dilation_rate=dilation_rate)

    print(X_train.shape)
    model.build(X_train.shape)
    model.summary()

    log_dir = "%s/Img ResNet50 dilat=%d %s" % (tensorboard_dir, dilation_rate, model_name)

    callbacks = [
        # EarlyStopping(monitor='val_acc', patience=8),
        ModelCheckpoint(out_dir + "/models/" + model.name + "_acc{val_acc:.2f}.h5", save_best_only=True),
        TensorBoard(
            log_dir=log_dir,
            write_images=True,
        )
    ]

    label_encoder = LabelBinarizer()
    label_encoder.fit(class_labels)

    if print_cm:
        callbacks.append(ConfusionMatrixCallback(
            log_dir=log_dir,
            X_val=X_validation,
            Y_val=Y_validation,
            classes=class_labels,
            label_encoder=label_encoder
        ))

    os.makedirs(out_dir + "/models", exist_ok=True)

    model.fit(
        X_train, label_encoder.transform(Y_train),
        validation_data=(X_validation, label_encoder.transform(Y_validation)),
        class_weight=class_weight,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=True
    )


def train_inceptionresnetv2_pretrained_model(dilation_rate, optimizer: optimizers.Optimizer = optimizers.Adam(lr=1e-3),
                                             model_name=""):
    generator = create_image_generator_flow(ImageDataGenerator(
        preprocessing_function=inception_resnet_v2.preprocess_input
    ), subset='training')

    steps_per_epoch = generator.samples / batch_size
    classes = generator.class_indices
    num_classes = len(classes)
    class_labels = generator.classes

    # Calculate class weights to handle unbalanced classes
    class_indices = list(generator.class_indices.values())
    class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', class_indices, class_labels)

    (X_train, X_validation, Y_train, Y_validation) = inceptionresnetv2_pretrained_model_split(
        generator,
        class_labels,
        steps_per_epoch,
    )

    model = build_model("inceptionresnetv2-pretrained", num_classes, optimizer, dilation_rate=dilation_rate)

    callbacks = [
        # EarlyStopping(monitor='val_acc', patience=8),
        ModelCheckpoint(out_dir + "/models/" + model.name + "_acc{val_acc:.2f}.h5", save_best_only=True),
        TensorBoard(
            log_dir="%s/Img InceptV2 dilat=%d %s More Dropout" % (tensorboard_dir, dilation_rate, model_name),
            write_images=True,
        )
    ]

    os.makedirs(out_dir + "/models", exist_ok=True)

    label_encoder = LabelBinarizer()
    label_encoder.fit(class_labels)

    model.fit(
        X_train, label_encoder.transform(Y_train),
        validation_data=(X_validation, label_encoder.transform(Y_validation)),
        class_weight=class_weight,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=True
    )


def train_inceptionresnetv2_pretrained_model_with_retrain():
    print("Creating generator...")
    datagen = ImageDataGenerator(
        preprocessing_function=inception_resnet_v2.preprocess_input,
        validation_split=val_frac
    )
    generator = create_image_generator_flow(datagen, subset="training")
    validation_generator = create_image_generator_flow(datagen, subset="validation")
    print("DONE")

    steps_per_epoch = generator.samples / batch_size
    classes = generator.class_indices
    num_classes = len(classes)
    class_labels = generator.classes

    # Calculate class weights to handle unbalanced classes
    print("Computing class weights...")
    class_indices = list(generator.class_indices.values())
    class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', class_indices, class_labels)
    print("DONE")

    print("Creating base model (Inception ResNet V2)")
    model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_size)
    print("Appending custom predictor")
    optimizer = optimizers.Adam(lr=1e-3)
    model = build_model("inceptionresnetv2-with-retrain", num_classes, optimizer, base_model=model)
    print("DONE")

    callbacks = [
        EarlyStopping(monitor='val_acc', patience=8),
        ModelCheckpoint(out_dir + "/models/" + model.name + "_acc{val_acc:.2f}.h5", save_best_only=True)
    ]

    os.makedirs(out_dir + "/models", exist_ok=True)

    model.fit_generator(
        generator=generator,
        validation_data=validation_generator,
        validation_steps=(validation_generator.samples / batch_size),
        class_weight=class_weight,
        epochs=num_epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )


def train_raw_model():
    datagen = ImageDataGenerator(
        validation_split=val_frac
    )
    train_generator = create_image_generator_flow(datagen, subset="training")
    validation_generator = create_image_generator_flow(datagen, subset="validation")

    steps_per_epoch = train_generator.samples / batch_size
    classes = train_generator.class_indices
    num_classes = len(classes)
    class_labels = train_generator.classes

    # Calculate class weights to handle unbalanced classes
    class_indices = list(train_generator.class_indices.values())
    class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', class_indices, class_labels)

    optimizer = optimizers.Adam(lr=1e-3)
    model = build_model("raw", num_classes, optimizer, input_shape=input_size)

    callbacks = [
        EarlyStopping(monitor='val_acc', patience=8),
        ModelCheckpoint(out_dir + "/models/" + model.name + "_acc{val_acc:.2f}.h5", save_best_only=True)
    ]

    os.makedirs(out_dir + "/models", exist_ok=True)

    model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        validation_steps=(validation_generator.samples / batch_size),
        class_weight=class_weight,
        epochs=num_epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        verbose=1
    )


clean_tensorboard = False

if clean_tensorboard:
    for d in glob.glob(tensorboard_dir + "/Img*"):
        for root, dirs, files in os.walk(d, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(d)

# exit(1)
train_resnet50_pretrained_model(dilation_rate=1, optimizer=optimizers.SGD(lr=0.003), model_name="SGD-0_003")
exit(0)
train_inceptionresnetv2_pretrained_model(dilation_rate=1, optimizer=optimizers.SGD(lr=0.003), model_name="SGD-0_003")
exit(0)
traing_nasnet_pretrained_model(dilation_rate=1)
exit(0)
# train_inceptionresnetv2_pretrained_model_with_retrain()
train_resnet50_pretrained_model(dilation_rate=1, optimizer=optimizers.SGD(lr=0.002), model_name="SGD-0_002")
exit(0)
train_resnet50_pretrained_model(dilation_rate=1, optimizer=optimizers.SGD(lr=0.001), model_name="SGD-0_001")
train_inceptionresnetv2_pretrained_model(dilation_rate=1, optimizer=optimizers.SGD(lr=0.001), model_name="SGD-0_001")
train_inceptionresnetv2_pretrained_model(dilation_rate=1, optimizer=optimizers.SGD(lr=0.002), model_name="SGD-0_002")
exit(0)
train_resnet50_pretrained_model(dilation_rate=1)
train_resnet50_pretrained_model(dilation_rate=2)
train_resnet50_pretrained_model(dilation_rate=4)
train_resnet50_pretrained_model(dilation_rate=8)
train_inceptionresnetv2_pretrained_model(dilation_rate=1)
train_inceptionresnetv2_pretrained_model(dilation_rate=2)
train_inceptionresnetv2_pretrained_model(dilation_rate=4)
train_inceptionresnetv2_pretrained_model(dilation_rate=8)
traing_nasnet_pretrained_model(dilation_rate=2)
traing_nasnet_pretrained_model(dilation_rate=4)
traing_nasnet_pretrained_model(dilation_rate=8)
