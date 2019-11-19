import json
import nltk 

from .qa_data_utils import QADataSet
from .text_utils import in_white_list, preprocess_sentence

def load_squad(contexts, questions, answers, data_path, max_data_count=None,
               max_context_seq_length=None,
               max_question_seq_length=None,
               max_target_seq_length=None):
    if data_path is None:
        return

    with open(data_path) as file:
        json_data = json.load(file)

        for instance in json_data['data']:
            for paragraph in instance['paragraphs']:
                context = paragraph['context']
                qas = paragraph['qas']
                for qas_instance in qas:
                    question = qas_instance['question']
                    answers_list = qas_instance['answers']
                    for answer in answers_list:
                        ans = answer['text']
                        contexts.append(preprocess_sentence(context))
                        questions.append(preprocess_sentence(question))
                        answers.append(preprocess_sentence(ans))
                        break #only take one answer

                if max_data_count != None and len(contexts) >= max_data_count:
                    break

                break


class SquADDataSetV2(QADataSet):

    contexts = []
    questions = []
    answers = []

    def __init__(self, data_path, max_data_count=None,
                 max_context_seq_length=None,
                 max_question_seq_length=None,
                 max_target_seq_length=None):
        super(SquADDataSetV2, self).__init__()

        load_squad(self.contexts, self.questions, self.answers, data_path=data_path,
                   max_data_count=max_data_count,
                   max_context_seq_length=max_context_seq_length,
                   max_question_seq_length=max_question_seq_length,
                   max_target_seq_length=max_target_seq_length)

    def load_model(self, data_path, max_data_count=None,
                   max_context_seq_length=None,
                   max_question_seq_length=None,
                   max_target_seq_length=None):
        load_squad(self.contexts, self.questions, self.answers, data_path,
                   max_data_count=max_data_count,
                   max_context_seq_length=max_context_seq_length,
                   max_question_seq_length=max_question_seq_length,
                   max_target_seq_length=max_target_seq_length
                   )
