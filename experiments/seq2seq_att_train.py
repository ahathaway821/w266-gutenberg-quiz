from library.models.seq2seq_att import Seq2SeqAtt
from library.utility.squad import SquADDataSet
import numpy as np


def main():
    random_state = 42
    output_dir_path = './models'

    np.random.seed(random_state)
    data_set = SquADDataSet(data_path='../data/SQuAD/train-v1.1.json')

    qa = Seq2SeqAtt()
    qa.load_glove_model('../embeddings')
    batch_size = 64
    epochs = 200
    history = qa.fit(data_set, model_dir_path=output_dir_path,
                     batch_size=batch_size, epochs=epochs,
                     random_state=random_state)


if __name__ == '__main__':
    main()
