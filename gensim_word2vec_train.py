from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import logging
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import os
import multiprocessing

logger = logging.getLogger(__name__)
CORPUS_FILE = 'YOUR_OWN_CORPUS_FILENAME'
MODEL_DIR = './models'


def build_model_name(options):
    size = options['size']
    window = options['window']
    arch = 'sg' if options['sg'] == 1 else 'cbow'
    algorithm = 'hs' if options['hs'] == 1 else 'negative'
    mincount = options['min_count']
    epoch = options['iter']
    negative = options['negative'] if algorithm != 'hs' else ''

    return '{}_size{}_window{}_{}{}_mincount{}_epoch{}'.format(
        arch, size, window, algorithm, negative, mincount, epoch)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    os.makedirs(MODEL_DIR, exist_ok=True)
    cpu_count = multiprocessing.cpu_count()

    all_params = {
        'size': [100, 200, 300],
        'window': [5],
        'negative': [10, 15, 20],
        'sample': [0.001],
        'sg': [0],
        'hs': [1],
        'min_count': [7],
        'workers': [cpu_count],
        'iter': [5]
    }
    sentences = LineSentence(CORPUS_FILE)

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params: {}'.format(params))
        logger.info(build_model_name(params))
        model = Word2Vec(sentences, **params)
        model.wv.save_word2vec_format(
            os.path.join(MODEL_DIR, 'word2vec_{}.txt'.format(
                build_model_name(params))),
            binary=False)
