import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = [{word: model.score(X, length) for word, model in models}
                     for X, length in test.get_all_Xlengths()]

    d = np.empty((2, len(test_set.get_all_Xlengths())*len(models)),
                 dtype=[('word', np.unicode_),
                        ('logL', np.float64)])
    prob_df = pd.DataFrame(d)
    for i, probability in enumerate(probabilities):
        for word, logL in probability:
            prob_df[i] = {'word': word, 'logL': logL}

    guesses = prob_df.groupby('word', set_index=False).max()['word'].tolist()
    return probabilities, guesses
