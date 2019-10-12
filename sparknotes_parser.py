import requests
from bs4 import BeautifulSoup
import re

def get_some_soup(url):
    headers = {'content-type': 'text/html', 'Accept-Charset': 'UTF-8'}
    r = requests.get(url,  headers=headers)
    
    if (r == None):
        print(f"Unable to find url {url}")
    
    content = BeautifulSoup(r.content, 'html.parser')
    return content

class QuizQuestion:
    def __init__(self, question, answers, correct_answer):
        self.question = question
        self.answers = answers
        self.correct_answer = correct_answer
        
def clean_string(string):
    return string.strip().replace('\n', '')

def find_question(question_tag):
    question = question_tag.find('h3')
    
    question_text = question.getText()
    question_text = re.sub(r'[1-9][0-9]*\.', '', question_text)
    question_text = re.sub(r'[1-9][0-9]* of [1-9][0-9]*', '', question_text)
    return clean_string(question_text)
        
def find_answers(question_tag):
    answers = []
    correct_answer = ""
    for answer in question_tag.findAll("li"):
        if 'true-answer' in answer.get("class"):
            correct_answer = clean_string(answer.getText())
        answers.append(clean_string(answer.getText()))
        
    return answers, correct_answer

def find_questions_and_answers(soup_content):
    
    questions = []
    
    for question_tag in soup_content.findAll("div", {"class": "quick-quiz-question"}):
        question = find_question(question_tag)
        answers, correct_answer = find_answers(question_tag)
        questions.append(QuizQuestion(question, answers, correct_answer))
    
    return questions

def get_quiz_for_book(url):
    soup_content = get_some_soup(url)
    return find_questions_and_answers(soup_content)
    
    