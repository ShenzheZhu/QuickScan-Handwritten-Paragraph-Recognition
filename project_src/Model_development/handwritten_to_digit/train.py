#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 20:33
# @Author  : FywOo02
# @FileName: Convert_model.py
# @Software: PyCharm
import stow as stow
import os

import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau,ModelCheckpoint
from mltu.tensorflow.callbacks import TrainLogger, Model2onnx

try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in
     tf.config.experimental.list_physical_devices("GPU")]
except:
    pass

from mltu import CVImage
from mltu.augmentors import RandomBrightness, RandomErodeDilate, \
    RandomSharpen, RandomRotate
from mltu.tensorflow.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from tqdm import tqdm


from keras import layers
from keras.models import Model
from mltu.tensorflow.model_utils import residual_block

from model_config import ModelConfigs

dataset, vocab, max_len = [], set(), 0


def data_preprocessing():
    global dataset
    global vocab
    global max_len

    sentences_txt_path = stow.join('original_dataset', 'ascii',
                                   'sentences.txt')
    sentences_folder_path = stow.join('original_dataset', 'sentences')

    words = open(sentences_txt_path, "r").readlines()
    for line in tqdm(words):
        if line.startswith("#"):
            continue

        line_split = line.split(" ")
        if line_split[2] == "err":
            continue

        folder1 = line_split[0][:3]
        folder2 = line_split[0][:8]
        file_name = line_split[0] + ".png"
        label = line_split[-1].rstrip('\n')

        # recplace '|' with ' ' in label
        label = label.replace('|', ' ')

        rel_path = stow.join(sentences_folder_path, folder1, folder2, file_name)
        if not stow.exists(rel_path):
            continue

        dataset.append([rel_path, label])
        vocab.update(list(label))
        max_len = max(max_len, len(label))


def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")

    # normalize images here instead in preprocessing step
    input = layers.Lambda(lambda x: x / 255)(inputs)

    x1 = residual_block(input, 32, activation=activation, skip_conv=True,
                        strides=1, dropout=dropout)

    x2 = residual_block(x1, 32, activation=activation, skip_conv=True,
                        strides=2, dropout=dropout)
    x3 = residual_block(x2, 32, activation=activation, skip_conv=False,
                        strides=1, dropout=dropout)

    x4 = residual_block(x3, 64, activation=activation, skip_conv=True,
                        strides=2, dropout=dropout)
    x5 = residual_block(x4, 64, activation=activation, skip_conv=False,
                        strides=1, dropout=dropout)

    x6 = residual_block(x5, 128, activation=activation, skip_conv=True,
                        strides=2, dropout=dropout)
    x7 = residual_block(x6, 128, activation=activation, skip_conv=True,
                        strides=1, dropout=dropout)

    x8 = residual_block(x7, 128, activation=activation, skip_conv=True,
                        strides=2, dropout=dropout)
    x9 = residual_block(x8, 128, activation=activation, skip_conv=False,
                        strides=1, dropout=dropout)

    squeezed = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    blstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(
        squeezed)
    blstm = layers.Dropout(dropout)(blstm)

    blstm = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(blstm)
    blstm = layers.Dropout(dropout)(blstm)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(
        blstm)

    model = Model(inputs=inputs, outputs=output)
    return model


if __name__ == "__main__":
    data_preprocessing()
    # print(original_dataset)
    # print(vocab)
    # print(max_len)

    # update model config
    configs = ModelConfigs()
    configs.vocab = "".join(vocab)
    configs.max_text_length = max_len
    configs.save()

    # create data_provider object
    data_provider = DataProvider(
        dataset=dataset,
        skip_validation=True,
        batch_size=configs.batch_size,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
            LabelIndexer(configs.vocab),
            LabelPadding(max_word_length=configs.max_text_length,
                         padding_value=len(configs.vocab)),
        ],
    )

    # split train/validation data set
    train_data_provider, val_data_provider = data_provider.split(split=0.8)


    # enhance data
    train_data_provider.augmentors = [
        RandomBrightness(),
        RandomErodeDilate(),
        RandomSharpen(),
    ]

    # Creating TensorFlow model architecture
    model = train_model(
        input_dim=(configs.height, configs.width, 3),
        output_dim=len(configs.vocab),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
        loss=CTCloss(),
        metrics=[
            CERMetric(vocabulary=configs.vocab),
            WERMetric(vocabulary=configs.vocab)
        ],
        run_eagerly=False
    )
    model.summary(line_length=110)

    # Define callbacks
    earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1,mode="min")
    checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5",monitor="val_CER", verbose=1,save_best_only=True, mode="min")
    trainLogger = TrainLogger(configs.model_path)
    tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
    reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9,min_delta=1e-10, patience=5, verbose=1,mode="auto")
    model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

    # Train the model
    model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=configs.train_epochs,
        callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
        workers=configs.train_workers
    )

    # Save training and validation datasets as csv files
    train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
    val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
