import tensorflow as tf
from models import CustomSchedule,transformer
import data
from params import Params as P
import os

EN_MAX_LENGTH = P.EN_MAX_LENGTH
DE_MAX_LENGTH = P.DE_MAX_LENGTH
D_MODEL = P.D_MODEL
NUM_LAYERS = P.NUM_LAYERS
NUM_HEADS = P.NUM_HEADS
UNITS = P.UNITS
DROPOUT = P.DROPOUT
BATCH_SIZE = P.BATCH_SIZE

def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1,DE_MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, DE_MAX_LENGTH - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

def train():
    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    Data = data.Data(EN_MAX_LENGTH,DE_MAX_LENGTH,BATCH_SIZE)

    tf.keras.backend.clear_session()

    # Hyper-parameters
    model = transformer(
        vocab_size=Data.VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
    if os.path.exists(P.MODEL_PATH):
        model.load_weights(P.MODEL_PATH)
    EPOCHS = P.EPOCHS
    '''
    model.fit(Data.dataIter(),epochs=EPOCHS)
model.save_weights('chat_bot.h5')
    '''
    for i in range(EPOCHS):
        model.fit(Data.dataIter(), epochs=i+1,initial_epoch=i)
        model.save_weights(P.MODEL_PATH)

if __name__ == '__main__':
    train()