import tensorflow as tf
from models import transformer
from  data import Data

from params import Params as P

EN_MAX_LENGTH = P.EN_MAX_LENGTH
DE_MAX_LENGTH = P.DE_MAX_LENGTH
D_MODEL = P.D_MODEL
NUM_LAYERS = P.NUM_LAYERS
NUM_HEADS = P.NUM_HEADS
UNITS = P.UNITS
DROPOUT = P.DROPOUT
BATCH_SIZE = P.BATCH_SIZE

Data = Data(EN_MAX_LENGTH,DE_MAX_LENGTH,BATCH_SIZE,training=False)

model = transformer(
    vocab_size=Data.VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

model.load_weights('chat_bot.h5')

def evaluate(inputs,batch=None):
    '''
    对输入预测其输出，batch表示批处理的大小，默认为1
    '''
    output = tf.expand_dims(Data.START_TOKEN, 0)
    if batch is None:
        sentence = tf.expand_dims(
        Data.START_TOKEN + inputs + Data.END_TOKEN, axis=0)
    else:
        sentence = inputs
        output = tf.tile(output,[batch,1])

    for i in range(Data.DE_MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.reduce_all(tf.equal(predicted_id, Data.END_TOKEN[0])):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)
    if batch is not None:
        return output
    return tf.squeeze(output, axis=0)


def predict():
    inputs = []
    while True:
        sentence = input('your:')
        if sentence == 'ESC':break
        inputs = Data.tokenizer.encode(sentence)
        prediction = evaluate(inputs)
        #inputs += [i for i in prediction if i < Data.tokenizer.vocab_size]
        predicted_sentence = Data.tokenizer.decode(
          [i for i in prediction if i < Data.tokenizer.vocab_size])
        print('ant\'s: {}'.format(predicted_sentence))

if __name__ == '__main__':
    print('没有app界面我也很绝望啊！')
    print('输入ESC退出')
    predict()