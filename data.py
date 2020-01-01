import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

class Data:
    def __init__(self,EN_MAX_LENGTH,DE_MAX_LENGTH,BATCH_SIZE,training=True):
        self.EN_MAX_LENGTH = EN_MAX_LENGTH
        self.DE_MAX_LENGTH = DE_MAX_LENGTH
        self.single_questions = []
        self.single_answers = []
        self.load_data()
        self.questions,self.answers = self.concat()
        self.build_Voca()
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = 20000
        if training == True:
            self.tokenized_inputs, self.tokenized_outputs = self.tokenize_and_filter()

    def load_data(self):
        f_q = open('questions.txt', 'r', encoding='utf-8')
        f_a = open('answers.txt', 'r', encoding='utf-8')
        flag = True
        while flag:
            single_question = []
            single_answer = []
            while True:
                answer = f_a.readline()
                question = f_q.readline()
                if answer == '\n':break
                elif len(answer) == 0:
                    flag = False
                    break
                single_answer.append(answer.replace('\n',''))
                single_question.append(question.replace('\n',''))
            self.single_questions.append(single_question)
            self.single_answers.append(single_answer)
        f_q.close()
        f_a.close()

    def concat(self,concat = False):
        multi_turn_q = []
        multi_turn_a = []
        for questions,answers in tqdm(zip(self.single_questions,self.single_answers)):
            for i,answer in enumerate(answers):
                multi_turn_a.append(answer)
                if concat == True:
                    q = ''
                    for j in range(i,-1,-1):
                        if len(questions[j] +'</S> '+q) > self.EN_MAX_LENGTH*2:
                            break
                        q = questions[j] +'</S> '+q
                    multi_turn_q.append(q)
                else:
                    multi_turn_q.append(questions[i])
        return multi_turn_q,multi_turn_a

    def build_Voca(self):
        tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            self.questions + self.answers, target_vocab_size=2 ** 13)
        # Define start and end token to indicate the start and end of a sentence
        self.START_TOKEN, self.END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
        # Vocabulary size plus start and end token
        self.VOCAB_SIZE = tokenizer.vocab_size + 2
        self.tokenizer = tokenizer

    def tokenize_and_filter(self):
        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in tqdm(zip(self.questions,self.answers)):
            # tokenize sentence
            sentence1 = self.START_TOKEN + self.tokenizer.encode(sentence1) + self.END_TOKEN
            sentence2 = self.START_TOKEN + self.tokenizer.encode(sentence2) + self.END_TOKEN
            # check tokenized sentence max length
            if len(sentence1) <= self.EN_MAX_LENGTH and len(sentence2) <= self.DE_MAX_LENGTH:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)

        # pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=self.EN_MAX_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=self.DE_MAX_LENGTH, padding='post')

        return tokenized_inputs, tokenized_outputs

    def dataIter(self):

        # decoder inputs use the previous target as input
        # remove START_TOKEN from targets
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': self.tokenized_inputs,
                'dec_inputs': self.tokenized_outputs[:, :-1]
            },
            {
                'outputs': self.tokenized_outputs[:, 1:]
            },
        ))

        dataset = dataset.cache()
        dataset = dataset.shuffle(self.BUFFER_SIZE)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def language_model(self):
        self.keys = {}
        self.bigam = {}
        k_n = .0
        b_n = .0
        self.mask = 1000000.
        for Qs,As in tqdm(zip(self.single_questions,self.single_answers)):
            for q in Qs+As[-1:]:
                indexs =  self.tokenizer.encode(q)
                pre = None
                for i in indexs:
                    if i in self.keys:self.keys[i] +=1
                    else:self.keys[i] = 1
                    k_n += 1.
                    if pre is not None:
                        x = pre*self.mask + i
                        if x in self.bigam:self.bigam[x] += 1
                        else: self.bigam[x] = 1
                        b_n += 1.
                    pre = i

