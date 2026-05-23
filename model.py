"""
LSTM-based sign language recognition model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from config import (
    SEQUENCE_LENGTH, NUM_FEATURES,
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, EARLY_STOPPING_PATIENCE,
    MODEL_PATH, L2_REGULARIZATION,
)


def build_model(num_classes: int) -> tf.keras.Model:
    l2_reg = regularizers.l2(L2_REGULARIZATION)  # L2 정규화
    
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),

        layers.LSTM(LSTM_UNITS_1, return_sequences=True, kernel_regularizer=l2_reg),
        layers.Dropout(DROPOUT_RATE),

        layers.LSTM(LSTM_UNITS_2, return_sequences=False, kernel_regularizer=l2_reg),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(DENSE_UNITS, activation='relu', kernel_regularizer=l2_reg),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2_reg),
    ], name='sign_lstm')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),  # 학습률 약간 감소
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def get_callbacks() -> list:
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            min_delta=1e-4,
            patience=10,
            min_lr=1e-6,
            verbose=0,
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=0,
        ),
    ]
