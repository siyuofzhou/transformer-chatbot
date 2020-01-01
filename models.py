import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    '''
    query:shape=（Batch_size,n,q_len,vec)
    key:shape =(Batch_size,n,k_len,vec)
    value:shape = (Batch_size,n,v_len,vec)
    mask:shape=(Batch_size,1,1,k_len)
    其中：k_len == v_len , n表示注意力头的数量，mask是掩码矩阵
    功能：计算attention值
    返回结果：output =softmax( query x key^T /sqrt(vec) * mask) * value
    '''
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output

class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    多头注意力计算
    d_model:表示输出的特征维度大小
    num_heads:表示注意力头的数量
    保证d_model被num_heads整除
    '''
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        '''
        :param inputs:字典，有如下关键字
        query: shape (Batch_size,q_len,d_model)
        key:shape(Batch_size,k_len,d_model)
        value:shape(Batch_size,v_len,d_model)
        mask :shape(Batch_size,1,1,k_len)
        其中：bacth_size表示批大小，k_len == v_len
        一般key和value使同一个变量矩阵,mask是掩码矩阵
        功能：先对query,value,key的最后一维进行分割，分成num_head份，num_head表示注意力头数量
        然后在第二维进行拼接，拼接后计算注意力值，然后再变换回来。
        :return: output shape(Batch_size,q_len,d_model)
        '''
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs

def create_padding_mask(x):
    '''
    :param x:x是一个二维矩阵
    先根据x创建一个二维掩码矩阵，x中元素维0的地方掩码矩阵取1，其他地方取0
    之后对掩码矩阵进行扩展，添加二三维，变成4维矩阵
    :return: 一个4维掩码矩阵
    '''
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
    '''
    :param x:x是一个2维矩阵
    :return: 一个4维矩阵，3，4维所构成的矩阵是一个上三角矩阵，并且，如果x[i,j]=0，那么掩码矩阵
    mask[i,:,:j] = 1
    '''
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

class PositionalEncoding(tf.keras.layers.Layer):
    '''
    位置向量生成
    '''
    def __init__(self, position, d_model):
        '''
        :param position:位置的数量
        :param d_model: 词向量的维度
        '''
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        '''
        :param inputs:shape=(Batch_size,in_len,d_model)
        :return: 加上位置嵌入的向量
        '''
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    '''
    编码层
    :param units:隐藏层大小
    :param d_model: 特征维度大小
    :param num_heads: 注意力头数量
    :param dropout: dropout值
    :return: 编码层模型
    '''
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    '''
    :param vocab_size:字典大小
    :param num_layers: 编码层的数量
    :param units: 隐藏层的大小
    :param d_model: 特征维度的大小
    :param num_heads: 注意力头的数量
    :return: 完整的编码器，编码吗器输入（Batch_size,len)形状的输入，形状为（Batch_size,1,1,None)
    的掩码层
    '''
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    '''
    :param units:隐藏层大小
    :param d_model: 特征大小
    :param num_heads: 注意力头数量
    :return: 解码层模型。形状为（Batch_size,len)的回复序列，形状为（Batch_size,len,d_model)的编码
    结果，形状为(Batch_size,1,None,None)的自编码时的掩码look_ahead_mask,形状为（Batch_size,1,1,None)的正常
    上三角掩码矩阵。
    '''
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
        })
    attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    '''
    :param vocab_size:字典大小
    :param num_layers: 解码层的大小
    :param units: 隐藏层大小
    :param d_model: 特征维度大小
    :param num_heads: 注意力头数量
    :return: 解码器模型。形状为（Batch_size,len)的回复序列，形状为（Batch_size,len,d_model)的编码
    结果，形状为(Batch_size,1,None,None)的自编码时的掩码look_ahead_mask,形状为（Batch_size,1,1,None)的正常
    上三角掩码矩阵。
    '''
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
    '''
    :param vocab_size:字典大小
    :param num_layers: 解码器和编码器层的数量
    :param units: 隐藏层大小
    :param d_model: 特征维度的大小
    :param num_heads: 注意力头的数量
    :return: transformer模型。形状为（Batch_size,de_len)的回复序列，形状为（Batch_size,en_len)的编码
    输入，形状为(Batch_size,1,None,None)的自编码时的掩码look_ahead_mask,形状为（Batch_size,1,1,None)的正常
    上三角掩码矩阵。
    '''
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(
          create_padding_mask, output_shape=(1, 1, None),
          name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
          create_look_ahead_mask,
          output_shape=(1, None, None),
          name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
          create_padding_mask, output_shape=(1, 1, None),
          name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
          vocab_size=vocab_size,
          num_layers=num_layers,
          units=units,
          d_model=d_model,
          num_heads=num_heads,
          dropout=dropout,
      )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
          vocab_size=vocab_size,
          num_layers=num_layers,
          units=units,
          d_model=d_model,
          num_heads=num_heads,
          dropout=dropout,
      )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)
    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''
    warm up 启动器
    '''
    def __init__(self, d_model, warmup_steps=4000):
        '''
        :param d_model:特征维度大小
        :param warmup_steps: 预热的步数，预热时学习率上升，预热结束，学习率下降
        '''
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



