import nltk
import nltk.data
from nltk.corpus import stopwords
import uuid
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import traceback
import csv
import tfidf_functions
from html.parser import HTMLParser

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
narrative_qa_repo_filepath = "../../../narrativeqa"
data_directory_filepath = "../../data"

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

# Read in an entire story based on document id
def get_document_text(document_id):
    try:
        text=""
        try:
            f=open(f"{narrative_qa_repo_filepath}/tmp/{document_id}.content", "r", encoding="utf-16")
            text = f.read()
            f.close()
        except:
            f=open(f"{narrative_qa_repo_filepath}/tmp/{document_id}.content", "r", encoding="ISO-8859-1")
            text = f.read()
            f.close()
            
        #text = strip_tags(text)
        return text
    except Exception as e:
        print(f"Error getting document {document_id}")
        print(f"Exception: {e}")      



# Split text into passages to serve as documents in tfidf
def split_document_and_tfidf_vectorize_paragraphs(text, document_id, num_characters=1500):
    paragraphs = text.split('\n\n')
    
    if len(paragraphs) < 10:
        print(text)
        print(paragraphs)
        print('heeeelp')
        print(f'doc id: {document_id}')
        return
    
    passages = []
    passage_text = ""
    
    for p in paragraphs:
        passage_text = passage_text + p.replace(" ", "")
        if len(passage_text) > num_characters - 100:
            no_tags = strip_tags(passage_text)
            passages.append(no_tags)
            passage_text = ""

    vectorizer = TfidfVectorizer(stop_words=set(stopwords.words("english")))
    try:
        tfidf = vectorizer.fit_transform(passages)
        return passages, tfidf, vectorizer
    except Exception:
        traceback.print_exc()
        print(passages)
        
# Split text into passages to serve as documents in tfidf
def split_document_and_tfidf_vectorize_characters(text, num_characters=1500):
    passages = [text[i:i+num_characters] for i in range(0, len(text), num_characters)]

    #passages = text.split('\n\n')

    #passages = list(filter(None, passages))
    #print(len(passages))

    vectorizer = TfidfVectorizer(stop_words=set(stopwords.words("english")))
    try:
        tfidf = vectorizer.fit_transform(passages)
        return passages, tfidf, vectorizer
    except Exception:
        traceback.print_exc()
        print(passages)
        

    
# Split text into passages to serve as documents in tfidf
def split_document_and_tfidf_vectorize_sentences(text, document_id, num_characters=1500):

    text = strip_tags(text)
    sentences = sent_detector.tokenize(text.strip())
    
    if len(sentences) < 10:
        print(text)
        print(sentences)
        print('heeeelp')
        print(f'doc id: {document_id}')
        return
    
    passages = []
    sentence_text = ""
    
    for s in sentences:
        sentence_text = sentence_text + " " + " ".join(s.split())
        if len(sentence_text) > num_characters - 100:
            passages.append(sentence_text)
            sentence_text = ""

    vectorizer = TfidfVectorizer(stop_words=set(stopwords.words("english")))
    try:
        tfidf = vectorizer.fit_transform(passages)
        return passages, tfidf, vectorizer
    except Exception:
        traceback.print_exc()
        print(passages)

def split_document_and_tfidf_vectorize(text, document_id):
    #return split_document_and_tfidf_vectorize_paragraphs(text, document_id)
    return split_document_and_tfidf_vectorize_sentences(text, document_id)
        
class QAPair:
    passages = []
    
    def __init__(self, document_id, question, answer1, answer2, set_split):
        self.document_id = document_id
        self.question = question
        self.answer1 = answer1
        self.answer2 = answer2
        self.id = uuid.uuid4()
        self.set_split = set_split
        
# Load all question answer pairs for the available documents
#document_id, set, question, answer1, answer2, question_tokenized, answer1_tokenized, answer2_tokenized.
def get_question_answer_pairs():
    document_questions = {}
    with open(f'{narrative_qa_repo_filepath}/qaps.csv', newline='') as csvfile:
        rows = csv.DictReader(csvfile, delimiter=',')
        for qpair in rows:
            document_id = qpair['document_id']

            if document_id not in document_questions.keys():
                document_questions[document_id] = []

            document_questions[document_id].append(QAPair(document_id, qpair['question_tokenized'], qpair['answer1'], qpair['answer2'], qpair['set']))

    return document_questions



# Get the top n passage indices in regards to a query within a document
# Based on cosine simliarity
def get_related_passage_indices(question, vectorizer, tfidf, num_passages_to_return=5):
    q_vec = vectorizer.transform([question])
    cosine_similarities = linear_kernel(q_vec, tfidf).flatten()

    related_docs_indices_a = cosine_similarities.argsort()[:-num_passages_to_return:-1]
    related_docs_indices = []
    for index in related_docs_indices_a:
        if abs(cosine_similarities[index]) > 0:
            related_docs_indices.append(index)
            
    #if len(related_docs_indices) == 0:
    #    print(f'empty question: {question}')
    #    print(f'empty qvec: {q_vec}')
    
    return related_docs_indices

# Get the top n passages in regards to a query
def get_related_passages(passages, related_docs_indices):
    related_passages = []
    for i in related_docs_indices:
        related_passages.append(passages[i])
    
    return related_passages
    
    
#   {
# .   title:
# .   document_id:
# .   paragraphs:[
#{                    "qas": [
#                        {
#                            "question": "In what country is Normandy located?",
#                            "id": "56ddde6b9a695914005b9628",
#                            "answers": [
#                                {
#                                    "text": "France",
#                                    "answer_start": 159
#                                },
#                                {
#                                    "text": "France",
#                                    "answer_start": 159
#                                },
#
#                            ],
#                            "is_impossible": false
#                        },}
#                       #context:

def convert_question_pair_to_squad_format(qa_pair):
    data = {
        "qas": [
            {
                "question": qa_pair.question,
                "id": str(qa_pair.id),
                "answers": [
                    {
                        "text": qa_pair.answer1
                    },
                    {
                        "text": qa_pair.answer2
                    }
                ]
            }
        ],
        "context": qa_pair.passages[0]
    }
    
    return data

def get_doc_start(text, start_search):
    doc_start = text.find(start_search)
    if doc_start == -1:
        start_search = "*** START "
        doc_start = text.find(start_search, 0)
        if doc_start == -1:
            start_search = "***START "
            doc_start = text.find(start_search, 0)
            if doc_start == -1:
                start_search = "<pre>"
                doc_start = text.find(start_search)
    return doc_start, start_search

def get_doc_end(text, end_search):
    doc_end = text.rfind(end_search)
    if doc_end == -1:
        end_search = "*** END"
        doc_end = text.rfind(end_search)
        if doc_end == -1:
            end_search = "***END"
            doc_end = text.rfind(end_search)
            if doc_end == -1:
                end_search = "</pre>"
                doc_end = text.rfind(end_search)
    return doc_end, end_search

def document_is_on_skip_list(document_id):
    skip_list = ['09355a61a4d84807f9533f31d3263809cc486b6b',
                 "492f2d56eba93816e7d0958e2ba62d36d93bc97e", 
                 "bd14fef15878fdac1e9c2d2dbe52df0951f38aad",
                "1aae28477e771b3af008ec59ce29086a1bc66776",
                "3747036f950fe8f79cdaa0eb713104b9eb8af6c5",
                "5283fa0a6ea69f4b4224d12018bbf985a2b80543"]
    if document_id in skip_list:
        return True
    return False

def set_data_split(data, book_data, data_split, train_data, valid_data, test_data):
    data['data'].append(book_data)
    
    if data_split == "train":
        train_data['data'].append(book_data)
    elif data_split == "valid":
        valid_data['data'].append(book_data)
    elif data_split == "test":
        test_data['data'].append(book_data) 
        
def save_train_valid_test_data(data, train_data, valid_data, test_data):
    target_directory = f"{data_directory_filepath}/nqa_squad_document_qa_passages"
        
    mini_train_data = {}
    mini_train_data['version']=version
    mini_train_data['data'] = []
    mini_train_data['data'] = train_data['data'][0:1000]
    
    squad_json_format_qa = json.dumps(mini_train_data)
    fq = open(f"{target_directory}/mini_train.data", "w")
    fq.write(squad_json_format_qa)
    fq.close()
    
    squad_json_format_qa = json.dumps(data)
    fq = open(f"{target_directory}/all.data", "w")
    fq.write(squad_json_format_qa)
    fq.close()
    
    squad_json_format_qa = json.dumps(train_data)
    fq = open(f"{target_directory}/train.data", "w")
    fq.write(squad_json_format_qa)
    fq.close()
    
    squad_json_format_qa = json.dumps(valid_data)
    fq = open(f"{target_directory}/valid.data", "w")
    fq.write(squad_json_format_qa)
    fq.close()
    
    squad_json_format_qa = json.dumps(test_data)
    fq = open(f"{target_directory}/test.data", "w")
    fq.write(squad_json_format_qa)
    fq.close()
    
    return train_data, valid_data, test_data, mini_train_data
    
# Loop through available documents, retrieve the top passages for each question/answer pair
# Write the returned passages to document_qa_passages directory for later use
# Pairs written as directionary of form {q: [passage, passage, etc]}

#document_id,set,kind,story_url,story_file_size,wiki_url,wiki_title,story_word_count,story_start,story_end

# {
# version:
# data: [

#}       
#]
#}
#]
#}
def get_and_write_qa_passages_as_squad(document_questions, max_stories=-1):
    with open(f'{narrative_qa_repo_filepath}/documents.csv') as f1:
        rows = csv.DictReader(f1, delimiter=',')
        i = 0 # document iterator index
        s = 0 # number of skipped question pairs
        q = 0 # number of total question pairs saved
        document_id=""
        version = "1.0"

        data = {}
        data['version'] = version
        data['data'] = []
        
        train_data = {}
        train_data['version'] = version
        train_data['data'] = []
        
        valid_data = {}
        valid_data['version'] = version
        valid_data['data'] = []
        
        test_data = {}
        test_data['version'] = version
        test_data['data'] = []
        
        for doc in rows:
            try:
                i = i + 1
                if i == 1:
                    continue

                book_data = {}
                document_id = doc['document_id']
                
                if document_is_on_skip_list(document_id):
                    continue

                book_data['title'] = doc['wiki_title']
                book_data['document_id'] = document_id
                book_data['paragraphs'] = []


                text = get_document_text(document_id)
                text = text.replace('</b><b>', '\n\n')
                #text = ' '.join(text.split())

                doc_start, start_search = get_doc_start(text, doc['story_start'])
                doc_end, end_search = get_doc_end(text, doc['story_end'])

                if doc_start != -1:
                    text = text[int(doc_start):int(doc_end)]

                passages, tfidf, vectorizer = split_document_and_tfidf_vectorize(text, document_id)
                passages_to_write = {}
                data_split = ""
                
                for qa_pair in document_questions[document_id]:
                    q = q + 1
                    related_indices = get_related_passage_indices(qa_pair.question, vectorizer, tfidf, num_passages_to_return=5)
                    related_passages = get_related_passages(passages, related_indices)
                    qa_pair.passages = related_passages

                    if len(qa_pair.passages) == 0:
                        #print(f'skipped pair: {document_id}')
                        s = s + 1
                        continue

                    book_data['paragraphs'].append(convert_question_pair_to_squad_format(qa_pair))
                    passages_to_write[qa_pair.question] = related_passages
                    data_split = qa_pair.set_split

                json_question_pairs = json.dumps(passages_to_write)
                fq = open(f"{data_directory_filepath}/nqa_document_qa_passages/{document_id}.q_passages", "w")
                fq.write(json_question_pairs)
                fq.close()

                set_data_split(data, book_data, data_split, train_data, valid_data, test_data)
                
                if max_stories != -1 and i > max_stories:
                    break

            except Exception:
                traceback.print_exc()
                print(f"Error processing qa pairs and passages for document {document_id}. Story no {i}")
                break

        save_train_valid_test_data(data, train_data, valid_data, test_data)
        
        print(f'total question pairs: {q}')
        print(f'skipped pairs: {s}')
