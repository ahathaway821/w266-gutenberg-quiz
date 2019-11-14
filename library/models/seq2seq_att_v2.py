from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, LSTM, Dense, Embedding, Dropout, add, RepeatVector, GRU, Concatenate, TimeDistributed, Bidirectional
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from library.utility import text_utils
from library.utility.qa_data_utils import Seq2SeqTripleSamples
import numpy as np
import nltk
import os
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ..layers.attention import AttentionLayer
from ..layers.b_attention import BahdanauAttention
from ..layers.encoder import Encoder
from ..layers.decoder import Decoder


def generate_batch(ds, input_data, output_data, batch_size):
    num_batches = len(input_data) // batch_size
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            encoder_input_paragraph_data_batch = []
            encoder_input_question_data_batch = []
            for input_paragraph_data, input_question_data in input_data[start:end]:
                encoder_input_paragraph_data_batch.append(input_paragraph_data)
                encoder_input_question_data_batch.append(input_question_data)
            encoder_input_paragraph_data_batch = pad_sequences(encoder_input_paragraph_data_batch,
                                                               ds.input_paragraph_max_seq_length)
            encoder_input_question_data_batch = pad_sequences(encoder_input_question_data_batch,
                                                              ds.input_question_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(batch_size, ds.target_max_seq_length, ds.num_target_tokens))
            decoder_input_data_batch = np.zeros(shape=(batch_size, ds.target_max_seq_length, ds.num_target_tokens))
            for lineIdx, target_wid_list in enumerate(output_data[start:end]):
                for idx, wid in enumerate(target_wid_list):
                    if wid == 0:  # UNKNOWN
                        continue
                    decoder_input_data_batch[lineIdx, idx, wid] = 1
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, wid] = 1
            yield [encoder_input_paragraph_data_batch, encoder_input_question_data_batch,
                   decoder_input_data_batch], decoder_target_data_batch


class Seq2SeqV2QA(object):
    model_name = 'seq2seq-qa-v2-att'

    def __init__(self):
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.input_paragraph_word2idx = None
        self.input_paragraph_idx2word = None
        self.input_question_word2idx = None
        self.input_question_idx2word = None
        self.target_word2idx = None
        self.target_idx2word = None
        self.max_encoder_paragraph_seq_length = None
        self.max_encoder_question_seq_length = None
        self.max_decoder_seq_length = None
        self.num_encoder_paragraph_tokens = None
        self.num_encoder_question_tokens = None
        self.num_decoder_tokens = None

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return os.path.join(model_dir_path, Seq2SeqV2QA.model_name + '-architecture.json')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, Seq2SeqV2QA.model_name + '-weights.h5')

    def load_model(self, model_dir_path):
        self.input_paragraph_word2idx = np.load(
            model_dir_path + '/' + self.model_name + '-input-paragraph-word2idx.npy').item()
        self.input_paragraph_idx2word = np.load(
            model_dir_path + '/' + self.model_name + '-input-paragraph-idx2word.npy').item()
        self.input_question_word2idx = np.load(
            model_dir_path + '/' + self.model_name + '-input-question-word2idx.npy').item()
        self.input_question_idx2word = np.load(
            model_dir_path + '/' + self.model_name + '-input-question-idx2word.npy').item()
        self.target_word2idx = np.load(model_dir_path + '/' + self.model_name + '-target-word2idx.npy').item()
        self.target_idx2word = np.load(model_dir_path + '/' + self.model_name + '-target-idx2word.npy').item()
        context = np.load(model_dir_path + '/' + self.model_name + '-config.npy').item()
        self.max_encoder_paragraph_seq_length = context['input_paragraph_max_seq_length']
        self.max_encoder_question_seq_length = context['input_question_max_seq_length']
        self.max_decoder_seq_length = context['target_max_seq_length']
        self.num_encoder_paragraph_tokens = context['num_input_paragraph_tokens']
        self.num_encoder_question_tokens = context['num_input_question_tokens']
        self.num_decoder_tokens = context['num_target_tokens']

        print(self.max_encoder_paragraph_seq_length)
        print(self.max_encoder_question_seq_length)
        print(self.max_decoder_seq_length)
        print(self.num_encoder_paragraph_tokens)
        print(self.num_encoder_question_tokens)
        print(self.num_decoder_tokens)

        self.create_model()
        weight_file_path = self.get_weight_file_path(model_dir_path)
        self.model.load_weights(weight_file_path)

    def create_model(self):
        return

    def reply(self, paragraph, question):
        return

    def test_run(self, ds, index=None):
        if index is None:
            index = 0
        paragraph, question, actual_answer = ds.get_data(index)
        predicted_answer = self.reply(paragraph, question)
        # print({'context': paragraph, 'question': question})
        print({'predict': predicted_answer, 'actual': actual_answer})

    def loss_function(self, real, pred, loss_object):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, q, c, a, q_enc_hidden, c_enc_hidden, question_encoder, context_encoder, answers, optimizer, decoder, loss_object):
        loss = 0
        BATCH_SIZE=64

        with tf.GradientTape() as tape:
            q_enc_output, q_enc_hidden = question_encoder(q, q_enc_hidden)
            c_enc_output, c_enc_hidden = context_encoder(c, c_enc_hidden)

            dec_hidden = tf.concat(q_enc_hidden, c_enc_hidden, axis=-1)
            enc_output = tf.concat(q_enc_output, c_enc_output, axis=-1)

            dec_input = tf.expand_dims([answers.word_index['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, a.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += self.loss_function(a[:, t], predictions, loss_object)

                # using teacher forcing
                dec_input = tf.expand_dims(a[:, t], 1)

        batch_loss = (loss / int(a.shape[1]))

        variables = question_encoder.trainable_variables + context_encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss



    def fit(self, contexts, questions, answers, model_dir_path, epochs=None, batch_size=None, test_size=None, random_state=None,
            save_best_only=False, max_input_vocab_size=None, max_target_vocab_size=None, max_num_examples=None):
        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 100
        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = 42
        if max_input_vocab_size is None:
            max_input_vocab_size = 5000
        if max_target_vocab_size is None:
            max_target_vocab_size = 5000

        question_tensor, question_tokenizer = text_utils.tokenize(questions)
        context_tensor, context_tokenizer = text_utils.tokenize(contexts)
        answer_tensor, answer_tokenizer = text_utils.tokenize(answers)

        max_length_question, max_length_context, max_length_answer = text_utils.max_length(question_tensor), text_utils.max_length(context_tensor), text_utils.max_length(answer_tensor)

        question_tensor_train, question_tensor_val, context_tensor_train, context_tensor_val, answer_tensor_train, answer_tensor_val = train_test_split(question_tensor, context_tensor, answer_tensor, test_size=test_size, random_state=random_state)

        #Create tf.data dataset
        buffer_size = len(question_tensor_train)

        steps_per_epoch = len(question_tensor_train) //  batch_size
        embedding_dim = 256
        units = 1024
        vocab_question_size = len(questions.word_index) + 1
        vocab_context_size = len(contexts.word_index) + 1
        vocab_answer_size = len(answers.word_index) + 1

        dataset = tf.data.Dataset.from_tensor_slices((question_tensor_train, context_tensor_train, answer_tensor_train)).shuffle(buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        # Encoding
        question_encoder = Encoder(vocab_question_size, embedding_dim, units, batch_size)
        context_encoder = Encoder(vocab_context_size, embedding_dim, units, batch_size)

        # Decoder
        decoder = Decoder(vocab_answer_size, embedding_dim, units, batch_size)

        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                        question_encoder=question_encoder,
                                        context_encoder=context_encoder,
                                        decoder=decoder)

        for epoch in range(epochs):
            start = time.time()

            question_enc_hidden = question_encoder.initialize_hidden_state()
            context_enc_hidden = context_encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (q, c, a)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(q, c, a, question_enc_hidden, context_enc_hidden, question_encoder, context_encoder, answers, optimizer, decoder, loss_object)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                batch,
                                                                batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

