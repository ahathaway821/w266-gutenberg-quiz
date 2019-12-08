## w266-gutenberg-quiz

### Abstract
Reading comprehension for texts with lengths longer than a few paragraphs, such as novels, plays, and movie scripts, presents a difficult challenge. Over the course of tens and hundreds of pages, a myriad of plot events and character interactions creates a rich text to draw insights and conclusions from. Typically, evaluating reading comprehension can be accomplished with question answering tasks. In an attempt to measure reading comprehension levels against a typical middle or high school student, we draw upon multiple choice quiz questions from the popular literature study guide website SparkNotes. We ask young students to consume classic texts by the likes of Shakespeare and Dickens, and so we can attempt to teach machines how to consume these same texts in an effort to emulate human understanding. Drawing from quiz questions available on SparkNotes, I present a new multiple choice dataset to be used as an evaluation technique for reading comprehension on longer narrative texts. I also introduce a transformer pointer generator model for abstractive question answering on this task.

### About
The goal of this project is to scrape quiz questions from SparkNotes, and then in combination with the NarrativeQA dataset, retrieve relevant passages for question answer pairs. Using these combinations of question/answer pairs and relevant passages, I use a couple different models in to answer these questions correctly.

This project includes a basic seq2seq lstm model for question answering along with a transformer pointer generator model. 

For consistency, all question/answer/passage sets are stored in the SQuAD data format so they can be consumed in the same fashion.

### Data Scraping
In the library/scraping folder you will find scripts to run through the SparkNotes website and pull down quiz questions for a given list of books, while avoiding any quizzes that require outside world context. The main web scraping functions are in sparknotes_parser.py, and can be run within the sparknotes_playground.ipynb notebook.

### Information Retrieval
In the library/retrieval folder, you will find scripts to pull back relevant text passages based on a query using tfidf. The majority of this functionality lives within tfidf_passage_retrieval.py, and can be run with the passage_retrieval.ipynb notebook.

### Models
The primary transformer-pointer-generator model can be found in model_notebooks at transformer_pointer_generator.ipynb.

Previous attempts at a basic LSTM based seq2seq model and a neural machine translation with attention base, halway through conversion to a question answering model, are included as well.


Code References:
https://github.com/thushv89/attention_keras/blob/master/layers/attention.py
https://github.com/chen0040/keras-question-and-answering-web-api
https://www.tensorflow.org/tutorials/text/transformer
https://github.com/ParikhKadam/bidaf-keras
https://github.com/abisee/pointer-generator,
