import tensorflow as tf
from models import transformer
import data
import math
from tqdm import trange
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

from params import Params as P

EN_MAX_LENGTH = P.EN_MAX_LENGTH
DE_MAX_LENGTH = P.DE_MAX_LENGTH
D_MODEL = P.D_MODEL
NUM_LAYERS = P.NUM_LAYERS
NUM_HEADS = P.NUM_HEADS
UNITS = P.UNITS
DROPOUT = P.DROPOUT
BATCH_SIZE = P.BATCH_SIZE

Data = data.Data(EN_MAX_LENGTH,DE_MAX_LENGTH,BATCH_SIZE)
Data.language_model()

model = transformer(
    vocab_size=Data.VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

model.load_weights(P.MODEL_PATH)

def perplexity(sentence):
    '''
    sentence 是列表
    这里的sentence 是模型输出的序列，或者是语言编码后的序列
    '''
    p = 1.
    for i in range(len(sentence)-1):
        x = sentence[i]*Data.mask+sentence[i+1]
        k = 0.5
        p += math.log((Data.bigam.get(x,0)+k)/(Data.keys.get(sentence[i],0)+k*Data.VOCAB_SIZE))

    return pow(2,-1./len(sentence)*p)

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


def cut(out):
    out2 = []
    for o in out:
        C = []
        for i, x in enumerate(o):
            if x == Data.END_TOKEN[0]: break
            if x < Data.START_TOKEN and x > 0:
                C.append(x)
        out2.append(C)
    return out2

S_per,S_b1,S_b2,S_b3,S_b4=0,0,0,0,0
N = 0
data = iter(Data.dataIter())
smooth = SmoothingFunction()
for i in trange(len(Data.questions)//BATCH_SIZE):
    d = next(data)
    q = d[0]['inputs']
    out = evaluate(q,BATCH_SIZE)
    out = cut(out.numpy())
    for q in out:
        S_per += perplexity(q)
    N += len(out)

    out2 = cut(d[1]['outputs'].numpy())
    for ref, cad in zip(out2, out):
        S_b1 += sentence_bleu([ref], cad, weights=(1, 0, 0, 0),smoothing_function=smooth.method1)
        S_b2 += sentence_bleu([ref], cad, weights=(0, 1, 0, 0),smoothing_function=smooth.method1)
        S_b3 += sentence_bleu([ref], cad, weights=(0, 0, 1, 0),smoothing_function=smooth.method1)
        S_b4 += sentence_bleu([ref], cad, weights=(0, 0, 0, 1),smoothing_function=smooth.method1)

print('perplexity:',S_per/N)
print('BELU_1_gram:',S_b1/N)
print('BELU_2_gram:',S_b2/N)
print('BELU_3_gram:',S_b3/N)
print('BELU_4_gram:',S_b4/N)
