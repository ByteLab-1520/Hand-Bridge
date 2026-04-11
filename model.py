"""
LSTM-based sign language recognition model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from config import (
    SEQUENCE_LENGTH, NUM_FEATURES,
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, EARLY_STOPPING_PATIENCE,
    MODEL_PATH,
)


def build_model(num_classes: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),

        layers.LSTM(LSTM_UNITS_1, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),

        layers.LSTM(LSTM_UNITS_2, return_sequences=False),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(DENSE_UNITS, activation='relu'),
        layers.Dropout(DROPOUT_RATE / 2),

        layers.Dense(num_classes, activation='softmax'),
    ], name='sign_lstm')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def get_callbacks() -> list:
    return [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]
