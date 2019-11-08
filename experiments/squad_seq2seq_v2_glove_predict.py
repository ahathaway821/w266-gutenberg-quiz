from library.models.seq2seq_v2_glove import Seq2SeqV2GloveQA
from library.utility.squad import SquADDataSet


def main():
    qa = Seq2SeqV2GloveQA()
    qa.load_glove_model('../embeddings')
    qa.load_model(model_dir_path='./models')
    data_set = SquADDataSet(data_path='../data/SQuAD/train-v1.1.json')
    for i in range(20):
        index = i * 10
        paragraph, question, actual_answer = data_set.get_data(index)
        predicted_answer = qa.reply(paragraph, question)
        print('context: ', paragraph)
        print('question: ', question)
        print({'guessed_answer': predicted_answer, 'actual_answer': actual_answer})


if __name__ == '__main__':
    main()