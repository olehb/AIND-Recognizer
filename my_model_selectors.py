import math
import statistics
import warnings
import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

from sklearn.model_selection import KFold

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False,
                 features = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.features = features

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        d = np.empty((0,), dtype=[('num_hidden_states', np.uint8), ('bic_score', np.float64)])
        results = pd.DataFrame(d)
        i = 0

        bic_score_term = len(self.features)*math.log(len(self.sequences))

        for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_hidden_states).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                bic_score = -2*logL + bic_score_term
                results = results.append([{'num_hidden_states': num_hidden_states, 'bic_score': bic_score}])
                i += 1
                if self.verbose:
                    print(f"{i} hidden states: {num_hidden_states}, BIC: {bic_score}, word: {self.this_word}")
            except:
                logging.exception(f"{i} hidden states: {num_hidden_states}, word: {self.this_word}")


        best_num_hidden_states = results.loc[results['bic_score'] == results['bic_score'].min()]['num_hidden_states'].tolist()[0]
        best_bic_score = results['bic_score'].min()
        if self.verbose:
            print(f"Best model for {self.this_word}: num_hidden_states: {best_num_hidden_states}, BIC: {best_bic_score}")

        return self.base_model(best_num_hidden_states)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        d = np.empty((0,), dtype=[('num_hidden_states', np.uint8), ('dic_score', np.float64)])
        results = pd.DataFrame(d)
        i = 0

        for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(num_hidden_states).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                
                not_this_words = [self.hwords[word] for word in self.hwords if word != self.this_word]
                antiLogL = sum([model.score(X, length) for X, length in not_this_words])
                dic_score = logL - antiLogL/len(not_this_words)

                results = results.append([{'num_hidden_states': num_hidden_states, 'dic_score': dic_score}])
            except:
                logging.exception(f"{i} hidden states: {num_hidden_states}, word: {self.this_word}")

        best_num_hidden_states = results.loc[results['dic_score'] == results['dic_score'].max()]['num_hidden_states'].tolist()[0]
        best_dic_score = results['dic_score'].max()
        if self.verbose:
            print(f"Best model for {self.this_word}: num_hidden_states: {best_num_hidden_states}, DIC: {best_dic_score}")

        return self.base_model(best_num_hidden_states)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        d = np.empty((0,), dtype=[('num_hidden_states', np.uint8), ('log_likelihood', np.float64)])
        results = pd.DataFrame(d)
        i = 0
        if len(self.sequences) < 2:
            print(f"Not enough sequences for word {self.this_word}")
            return None

        split_method = KFold(2) if len(self.sequences) == 2 else KFold()

        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
            X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
            for num_hidden_states in range(self.min_n_components, self.max_n_components+1):
                try:
                    model = self.base_model(num_hidden_states).fit(X_train, lengths_train)
                    logL = model.score(X_test, lengths_test)
                    results = results.append([{'num_hidden_states': num_hidden_states, 'log_likelihood': logL}])
                    i += 1
                    if self.verbose:
                        print(f"{i} hidden states: {num_hidden_states}, log likelihood: {logL}, word: {self.this_word}")
                except:
                    logging.exception(f"{i} hidden states: {num_hidden_states}, word: {self.this_word}")


        means = results.groupby('num_hidden_states').mean()
        best_num_hidden_states = means.loc[means['log_likelihood'].idxmax()].name
        best_log_likelihood_mean = means['log_likelihood'].max()
        if self.verbose:
            print(f"Best model for {self.this_word}: num_hidden_states: {best_num_hidden_states}, log_likelihood_mean = {best_log_likelihood_mean}")

        return self.base_model(best_num_hidden_states)
