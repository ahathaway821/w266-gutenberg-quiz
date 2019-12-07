import json
import nltk 

from .qa_data_utils import QADataSet
from .text_utils import in_white_list, preprocess_sentence

def load_squad(contexts, questions, answers, examples, mc_answers_a, mc_answers_b, mc_answers_c, mc_answers_d, data_path, max_data_count=None,
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
                    answer_a = qas_instance['answer_a']
                    answer_b = qas_instance['answer_b']
                    answer_c = qas_instance['answer_c']
                    answer_d = qas_instance['answer_d']
                    for answer in answers_list:
                        ans = answer['text']

                        c = preprocess_sentence(context, False)
                        q = preprocess_sentence(question, False)
                        a = preprocess_sentence(ans, False)
                        contexts.append(c)
                        questions.append(q)
                        answers.append(a)

                        examples.append((q,c,a))

                        mc_answers_a.append(preprocess_sentence(answer_a, False))
                        mc_answers_b.append(preprocess_sentence(answer_b, False))
                        mc_answers_c.append(preprocess_sentence(answer_c, False))
                        mc_answers_d.append(preprocess_sentence(answer_d, False))

                        break #only take one answer from original squad answer list

class SquADDataSetSparkNotes(QADataSet):


    def __init__(self, data_path, max_data_count=None,
                 max_context_seq_length=None,
                 max_question_seq_length=None,
                 max_target_seq_length=None):
        super(SquADDataSetSparkNotes, self).__init__()

        self.contexts = []
        self.questions = []
        self.answers = []
        self.examples = []
        self.mc_answers_a = []
        self.mc_answers_b = []
        self.mc_answers_c = []
        self.mc_answers_d = []

        load_squad(self.contexts, self.questions, self.answers, self.examples, 
                   self.mc_answers_a, 
                   self.mc_answers_b,
                   self.mc_answers_c,
                   self.mc_answers_d,
                   data_path=data_path,
                   max_data_count=max_data_count,
                   max_context_seq_length=max_context_seq_length,
                   max_question_seq_length=max_question_seq_length,
                   max_target_seq_length=max_target_seq_length)

    def load_model(self, data_path, max_data_count=None,
                   max_context_seq_length=None,
                   max_question_seq_length=None,
                   max_target_seq_length=None):
        load_squad(self.contexts, self.questions, self.answers, self.examples, 
                   self.mc_answers_a, 
                   self.mc_answers_b,
                   self.mc_answers_c,
                   self.mc_answers_d,
                   data_path,
                   max_data_count=max_data_count,
                   max_context_seq_length=max_context_seq_length,
                   max_question_seq_length=max_question_seq_length,
                   max_target_seq_length=max_target_seq_length
                   )
