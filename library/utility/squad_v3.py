import json
import nltk 

from .qa_data_utils import QADataSet
from .text_utils import in_white_list, preprocess_sentence

def load_squad(contexts, questions, answers, examples, data_path, max_data_count=None,
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

                        c = preprocess_sentence(context, False)
                        q = preprocess_sentence(question, False)
                        a = preprocess_sentence(ans, False)
                        contexts.append(c)
                        questions.append(q)
                        answers.append(a)

                        examples.append((q,c,a))
                        break #only take one answer

class SquADDataSetV3(QADataSet):

    contexts = []
    questions = []
    answers = []

    examples = []

    def __init__(self, data_path, max_data_count=None,
                 max_context_seq_length=None,
                 max_question_seq_length=None,
                 max_target_seq_length=None):
        super(SquADDataSetV3, self).__init__()

        load_squad(self.contexts, self.questions, self.answers, self.examples, data_path=data_path,
                   max_data_count=max_data_count,
                   max_context_seq_length=max_context_seq_length,
                   max_question_seq_length=max_question_seq_length,
                   max_target_seq_length=max_target_seq_length)

    def load_model(self, data_path, max_data_count=None,
                   max_context_seq_length=None,
                   max_question_seq_length=None,
                   max_target_seq_length=None):
        load_squad(self.contexts, self.questions, self.answers, self.examples, data_path,
                   max_data_count=max_data_count,
                   max_context_seq_length=max_context_seq_length,
                   max_question_seq_length=max_question_seq_length,
                   max_target_seq_length=max_target_seq_length
                   )
