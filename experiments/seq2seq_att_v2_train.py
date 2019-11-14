from ..library.models.seq2seq_att_v2 import Seq2SeqAttV2
from ..library.utility.squad_v2 import SquADDataSetV2
import numpy as np
import pathlib


def main():
    random_state = 42
    output_dir_path = str(pathlib.Path(__file__).parent / 'models')

    np.random.seed(random_state)
    fn = pathlib.Path(__file__).parent.parent / 'data/SQuAD/train-v1.1.json'
    data_set = SquADDataSetV2(data_path=fn)

    qa = Seq2SeqAttV2()
    batch_size = 64
    epochs = 20
    history = qa.fit(data_set, model_dir_path=output_dir_path,
                     batch_size=batch_size, epochs=epochs,
                     random_state=random_state)


if __name__ == '__main__':
    main()
