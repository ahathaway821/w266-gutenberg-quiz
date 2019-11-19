from ..library.models.seq2seq_v2 import Seq2SeqV2QA
from ..library.utility.squad import SquADDataSet
import pathlib
import nltk

def main():
    qa = Seq2SeqV2QA()
    model_dir_path = str(pathlib.Path(__file__).parent / 'models')
    qa.load_model(model_dir_path=model_dir_path)

    fn = pathlib.Path(__file__).parent.parent / 'data/SQuAD/dev-v1.1.json'

    data_set = SquADDataSet(data_path=fn)
    scores_1 = 0
    scores_4 = 0
    num_examples = 50
    for i in range(50):
        index = i * 10
        paragraph, question, actual_answer = data_set.get_data(index)
        predicted_answer = qa.reply(paragraph, question)
        BLEU1score = nltk.translate.bleu_score.sentence_bleu([actual_answer], predicted_answer, weights=([1]))
        BLEU4score = nltk.translate.bleu_score.sentence_bleu([actual_answer], predicted_answer, weights=(.25, .25, .25, .25))
        print('context: ', paragraph)
        print('question: ', question)
        print({'guessed_answer': predicted_answer, 'actual_answer': actual_answer})
        print(f'BLEU4 Score: {BLEU4score}')
        scores_1 = scores_1 + BLEU1score
        scores_4 = scores_4 + BLEU4score
    print('----------')
    print(f'Avg Score BLEU1: {scores_1/num_examples}')
    print(f'Avg Score BLEU4: {scores_4/num_examples}')

if __name__ == '__main__':
    main()
