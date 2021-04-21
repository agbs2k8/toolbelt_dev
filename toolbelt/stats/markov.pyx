#!python
#cython: language_level=3
import numbers
import numpy as np
import warnings
from ..utils import window


def get_unique_items(sequences, dtype=None):
    reduced_set = set([item for sublist in sequences for item in sublist])
    if dtype:
        return {cast_type(x, dtype) for x in reduced_set}
    else:
        return reduced_set


def _push_sequence(sequence, new_value):
    sequence = sequence.copy()
    if isinstance(sequence, list):
        sequence.insert(0, new_value)
        return sequence
    elif isinstance(sequence, np.ndarray):
        return np.insert(sequence, 0, new_value)


def _extend_sequence(sequence, new_value):
    sequence = sequence.copy()
    if isinstance(sequence, list):
        sequence.append(new_value)
        return sequence
    elif isinstance(sequence, np.ndarray):
        return np.append(sequence, new_value)


def _add_zeros_row(matrix):
    matrix = matrix.copy()
    new_row = np.zeros((1, matrix.shape[1]), dtype=matrix.dtype)
    return np.append(matrix, new_row, axis=0)


def cast_type(value, dtype):
    return dtype(value)


class MarkovChain:
    """
    Parameters
    ----------
    start_of_sequence_identifier : any object (default = 0)
        item to insert before sequences to denote the start of the sequence.
        It should not be anything that could possibly be part of any sequence.

    end_of_sequence_identifier : any object (default = -1)
        item to append to sequences to denote the end of the sequence.
        It should not be anything that could possibly be part of any sequence.

    Attributes
    ----------
    objects_ : dict
        A mapping of all of the sequence objects and it's index in the matrix

    probability_matrix_ : np.ndarray
        A transitional probabilities matrix where each row index represents the given state, and the
        columns represent the probably subsequent states

    instance_counts_ : np.ndarray
        the raw number observations that were used in generating the transitional probabilities matrix,
        preserved for use in updating the transitional probabilities matrix

    Examples
    --------
    >>> from toolbelt import MarkovChain
    >>> host_log_list = [
    ...     ['4107', '4107', '4107', '7610'],
    ...     ['7610', '7610', '7610', '7610'],
    ...     ['4107', '7610'],
    ...     ['7610'],
    ... ]
    >>> chain = MarkovChain()
    >>> _ = chain.fit(host_log_list)
    >>> print(chain.objects_)
    {0: '4107', 1:'7610'}
    >>> print(chain.instance_counts_)
    [[2 2]
     [0 3]]
    """
    # TODO: Add dtype like implemented for the SequenceModel

    def __init__(self, start_of_sequence_identifier=0, end_of_sequence_identifier=-1):
        self.sos = start_of_sequence_identifier
        self.eos = end_of_sequence_identifier
        #self._mapping = {self.sos: 0, self.eos: -1}  # item: index {'item1': 1, ...}


    def fit(self, sequences):
        """Learn transitional probabilities matrix from given list of sequences.

        Parameters
        ----------
        sequences : iterable
            a list of sequences to use in creating the transitional probabilities matrix

        Returns
        -------
        self : MarkovChain
        """
        _unique = get_unique_items(sequences)
        _mapping = {item: idx+1 for idx, item in enumerate(_unique)}
        n = len(_mapping)
        _sos = self.sos
        _eos = self.eos
        _mapping[_eos] = -1
        _mapping[_sos] = 0

        instance_counts_ = np.zeros((n+1, n+2), dtype=np.int32)

        for sequence in sequences:
            sequence = _push_sequence(_extend_sequence(sequence, _eos), _sos)
            for items in window(sequence, 2):
                instance_counts_[_mapping[items[0]], _mapping[items[1]]] += 1

        self.instance_counts_ = instance_counts_
        self._mapping = _mapping
        self.objects_ = {idx: value for value, idx in _mapping.items()}
        self.probability_matrix_ = instance_counts_/instance_counts_.sum(axis=1, keepdims=True)
        self.index_items = [self.objects_[x] for x in range(1, len(self.objects_)-1)]

        return self

    def extend(self, sequences):
        """Update the transitional probabilities matrix given new sequences

        Parameters
        ----------
        sequences : list of sequences
            a list of sequences to be used in updating the transitional probabilities matrix

        Returns
        -------
        self : MarkovChain
        """
        # Ensure the class instance is fitted
        if not hasattr(self, 'probability_matrix_'):
            raise ValueError("Model not fitted.")
        # if it is only a single sequence, wrap in a list so everything else works fine
        if not isinstance(sequences[0], (list, np.ndarray)):
            sequences = [sequences]

        # Pull out the objects I need to work with:
        _mapping = self._mapping
        index_items = self.index_items
        instance_counts_ = self.instance_counts_
        _sos = self.sos
        _eos = self.eos

        # determine if there are new possible values (so we need to extend the mappings & matrices
        _new = get_unique_items(sequences).difference(set(index_items))
        n_new = len(_new)
        if n_new > 0:
            count_matrix = instance_counts_.copy()
            n_rows, n_cols = count_matrix.shape
            # If there are new, split off the end proba column
            stop_col = count_matrix[:,-1]
            # Add new columns at the end of the matrix & re-append the stop probabilities
            with_new_cols = np.append(count_matrix[:,:-1], np.zeros((n_rows, n_new), dtype=np.int32), axis=1)
            all_cols = np.append(with_new_cols, count_matrix[:,-1].reshape(-1,1), axis=1)
            # Add new rows & assert that the shape is still n, n+1
            instance_counts_ = np.append(all_cols, np.zeros((n_new, n_cols+n_new), dtype=np.int32), axis=0)
            # check your work!
            n_rows2, n_cols2 = instance_counts_.shape
            if not n_cols2 == n_rows2+1 and n_cols2 == n_cols+n_new:
                raise ValueError('Matrix not extended correctly to accommodate {} new rows'.format(n_new))
            # Update _mapping
            for new_item in _new:
                _mapping[new_item] = n_rows  # to start, the old n_rows variable is equal to the next index value
                n_rows += 1

        # iterate through the new item(s) and update the instance_counts_
        for sequence in sequences:
            sequence = _push_sequence(_extend_sequence(sequence, _eos), _sos)
            for items in window(sequence, 2):
                instance_counts_[_mapping[items[0]], _mapping[items[1]]] += 1

        # do all the final updates like in the fit method
        self.instance_counts_ = instance_counts_
        self._mapping = _mapping
        self.objects_ = {idx: value for value, idx in _mapping.items()}
        self.probability_matrix_ = instance_counts_/instance_counts_.sum(axis=1, keepdims=True)
        self.index_items = [self.objects_[x] for x in range(1, len(self.objects_)-1)]
        return self

    def create_sequence(self, starting_state=None, random_seed=None):
        """Create a random sequence using the markov probabilities.

        Parameters
        ----------
        starting_state : object (default=None)
            a object contained in the existing state mapping to use as the starting state.
            If None, one is randomly selected based on the start_of_sequence transitional
            probabilities previously learned

        random_seed : integer
            random seed for numpy.random.seed()

        Returns
        -------
        sequence : a randomply generated sequence
        """
        # Ensure the class instance is fitted
        if not hasattr(self, 'probability_matrix_'):
            raise ValueError("Model not fitted.")
        # set random_seed
        if random_seed:
            np.random.seed(random_seed)
        # select starting_state if one is not given
        if not starting_state:
            next_item = self.generate_next(self.sos)
        else:
            next_item = starting_state
        sequence = []
        while next_item != self.eos:
            sequence.append(next_item)
            next_item = self.generate_next(next_item)
        return sequence

    def generate_next(self, current_state, random_seed=None):
        """Select the next object for a markov chain based on the current state

        Parameters
        ----------
        current_state : object
            a object contained in the existing state mapping

        random_seed : integer
            random seed for numpy.random.seed()

        Returns
        -------
        next_object : an object selected based on the transitional probabilities from the given current_state
        """
        # Ensure the class instance is fitted
        if not hasattr(self, 'probability_matrix_'):
            raise ValueError("Model not fitted.")
        # set random_seed
        if random_seed:
            np.random.seed(random_seed)
        _mapping = self._mapping
        idx = _mapping[current_state]
        probas = self.probability_matrix_[idx]
        return np.random.choice([self.sos] + self.index_items + [self.eos], p=probas)

    def subsequent_probability(self, current_state):
        """Returns the probabilities for the subsequent steps in the random walk given the current_state

        Parameters
        ----------
        current_state : object
            a object contained in the existing state mapping

        Returns
        -------
        probabilities : a mapping of all possible subsequent states and their probabilities
        """
        # Ensure the class instance is fitted
        if not hasattr(self, 'probability_matrix_'):
            raise ValueError("Model not fitted.")
        _mapping = self._mapping
        idx = _mapping[current_state]
        probas = self.probability_matrix_[idx]
        return {option: proba for option, proba in zip([self.sos] + self.index_items + [self.eos], probas) if proba > 0}

    def stationary_distribution(self):
        """Returns the stationary distribution values for all states not including the start and stop

            * If I started n random walks with these markov probabilities but never allowed any to end/exit
            these values are approximately what % of the random walks would be at that corresponding state
            at any given time *

        Returns
        -------
        stationary_distribution : dict
            a mapping of all all states and their stationary distribution values
        """
        X = self.probability_matrix_[1:, 1:-1]
        eig = np.linalg.eig(X.T)
        res = (eig[1]/sum(eig[1]))[:,0]
        return np.real(res)

    def sequence_probability(self, sequence, return_all_steps=False):
        """Returns the probability of seeing the exact passed sequence given the fitted transitional
        probability matrix
            ex: P([1,2,3]) = P( 1 | start ) * P( 2 | 1 ) * P( 3 | 2 ) * P( end | 3 )

        return_all_steps: If true, (default=False) returns a list of all probabilities for each
            transitional step in the sequence

        Returns
        -------
        sequence_probability : float
            the probability of the specified sequence occurring.

        all_probabilities : np.ndarray
            a list of all probabilities for the steps in the sequence.
            len(all_probabilities)  == len(sequence) + 1
        """
        _mapping = self._mapping
        _sos = self.sos
        _eos = self.eos
        probability_matrix_ = self.probability_matrix_
        all_probabilities = np.array([], dtype=np.float64)
        for step in window(_push_sequence(_extend_sequence(sequence, _eos), _sos), 2):
            step_probability = probability_matrix_[step[0], step[1]]
            all_probabilities = _extend_sequence(all_probabilities, step_probability)
        sequence_probability = np.cumprod(all_probabilities, dtype=np.float64)[-1]
        if return_all_steps:
            return sequence_probability, all_probabilities
        else:
            return sequence_probability


class SequenceModel:
    """
    Parameters
    ----------
    order : int (default = 2, max = 5)
        integer of the number of prior states to consider when calculating the probability of the current or subsequent
        state.  Ex: given sequence X = [ 1, 2, 3, 4, 5 ]
            if n = 2, P(X) = P(2 | [start, 1]) * P(3 | [1, 2]) * P(4 | [2, 3]) * P(5 | [3, 4]) * P(end | [4, 5])
        *Warning - a higher order results in exponentially more possible prior state conditions that need to be tracked

    dtype : data-type *required*

    end_of_sequence_identifier : any object (default = -1)
        item to append to sequences to denote the end of the sequence.
        It should not be anything that could possibly be part of any sequence.

    Attributes
    ----------
    states_ : set
        All possible states for the sequence model

    Examples
    --------
    >>> from toolbelt import SequenceModel
    >>> host_log_list = [
    ...     ['4107', '4107', '4107', '7610'],
    ...     ['7610', '7610', '7610', '7610'],
    ...     ['4107', '4107','7610'],
    ... ]
    >>> model = SequenceModel(order=2, dtype=str, end_of_sequence_identifier=-1)
    >>> _ = model.fit(host_log_list)
    >>> print(model.states_)
    {'7610', '4107', '-1'}
    >>> print(model.subsequent_probability(['4107','4107']))
    {'7610': 0.6666666666666666, '4107': 0.3333333333333333}
    """


    def __init__(self, order=2, dtype=str, end_of_sequence_identifier=-1):
        self.order = order
        if not isinstance(order, numbers.Integral) or not order <= 5:
            raise ValueError('Order must be an integer <=5.')
        self.dtype = dtype
        self.ref_type = None
        self.eos = cast_type(end_of_sequence_identifier, self.dtype)
        self.states_ = {self.eos}
        self._start_probas = dict()  # self._start_probas = {initial_state: [count, proba]}
        self._col_mapping = {self.eos: -1}  # self._col_mapping = {'state_id': idx, ...}
        self._row_mapping = dict()  # self._row_mapping = {'prior_set' : idx, ...}
        self._count_matrix = None  # np.array( n_prior_state_sets, n_individual_states )


    def _valid_sequence_length(self, sequence):
        if len(sequence) > self.order:
            return True
        else:
            return False

    def _update_starts(self, start_sequence, update_probas=True):
        _start_probas = self._start_probas
        if start_sequence is not None and not len(start_sequence) == self.order:
            raise ValueError('starting sequence order differs from specified order.')
        # update the count values
        if start_sequence is not None:
            start_sequence = self._sequence_to_key(start_sequence)

        if start_sequence is not None and start_sequence in _start_probas.keys():
            _start_probas[start_sequence][0] += 1
        elif start_sequence is not None:
            _start_probas[start_sequence] = [1, 0.0]
        # convert counts to percentages
        if update_probas:
            total_count = np.sum([value[0] for value in _start_probas.values()], dtype=np.int64)
            for key, value in _start_probas.items():
                _start_probas[key][1] = float(value[0]) / total_count
        # update class instance
        self._start_probas = _start_probas

    def _sequence_to_key(self, sequence):
        sequence = np.array(sequence, dtype=self.dtype)
        if isinstance(sequence[0], str):
            self.ref_type = 'string'
            return '|'.join(sequence)
        elif isinstance(sequence[0], numbers.Integral):
            self.ref_type = 'int'
            return '|'.join([str(x) for x in sequence])
        elif isinstance(sequence[0], numbers.Number):
            self.ref_type = 'float'
            return '|'.join([str(round(x,2)) for x in sequence])
        else:
            raise ValueError('Dtype for state identifier not supported')

    def _key_to_sequence(self, key):
        ref_type = self.ref_type
        if ref_type == 'string':
            return key.split('|')
        elif ref_type == 'int':
            return [int(x) for x in key.split('|')]
        elif ref_type == 'float':
            return [float(x) for x in key.split('|')]
        else:
            raise ValueError('Cannot re-create sequence from key {}'.format(key))


    def fit(self, sequences):
        """Learn transitional probabilities matrix from given list of sequences.

        Parameters
        ----------
        sequences : iterable
            a list of sequences to use in creating the transitional probabilities matrix

        Returns
        -------
        self : SequenceModel
        """
        order = self.order
        states_ = get_unique_items(sequences)
        _col_mapping = self._col_mapping  # map single states to index in probability rows
        for idx, state in enumerate(states_):
            _col_mapping[cast_type(state, self.dtype)] = idx
        n_states = len(states_)
        end = self.eos

        _row_mapping = dict()  # map prior-state-sets to their row index in the master table
        _count_matrix = self._count_matrix

        for sequence in sequences:
            sequence = np.array(_extend_sequence(sequence, end), dtype=self.dtype)
            # start proba - Update all start probas as the end rather than each time
            self._update_starts(start_sequence = sequence[:order], update_probas=False)
            # iterate through windows of order+1 size
            for subseq in window(sequence, order+1):
                prior_state = self._sequence_to_key(subseq[:order])
                col_idx = _col_mapping[subseq[-1]]

                if _count_matrix is None:  # for the very first iteration
                    _count_matrix = np.zeros((1, n_states+1), dtype=np.int64)  # create zeros matrix (one row)
                    _row_mapping[prior_state] = 0  # assign to index 0
                    # Set the correct entry in the matrix to 1
                    _count_matrix[0, col_idx] = 1

                elif prior_state not in _row_mapping.keys():  # this is a new instance of the prior
                    # add to _row_mapping
                    new_row_idx = _count_matrix.shape[0]
                    _row_mapping[prior_state] = new_row_idx
                    # add a new row to the count matrix:
                    _count_matrix = _add_zeros_row(_count_matrix)
                    # append zeros row to _count_matrix
                    _count_matrix[new_row_idx, col_idx] = 1
                else:
                    _row_idx = _row_mapping[prior_state]
                    _count_matrix[_row_idx, col_idx] += 1

        # Update start probabilities now that I've run through them all
        self._update_starts(start_sequence=None, update_probas=True)
        # make sure all of the instance data is updated with what I've used here
        self._count_matrix = _count_matrix
        self._proba_matrix = _count_matrix/_count_matrix.sum(axis=1, keepdims=True)
        self._row_mapping = _row_mapping
        self._col_mapping = _col_mapping
        self.states_ = set(_col_mapping.keys())
        return self

    def extend(self, sequences):
        """Update the transitional probabilities matrix given new sequences

        Parameters
        ----------
        sequences : list of sequences
            a list of sequences to be used in updating the transitional probabilities matrix

        Returns
        -------
        self : SequenceModel
        """
        # Ensure the class instance is fitted
        if not hasattr(self, '_proba_matrix'):
            raise ValueError("Model not fitted.")
        # if it is only a single sequence, wrap in a list so everything else works fine
        if not isinstance(sequences[0], (list, np.ndarray)):
            sequences = [sequences]

        order = self.order
        _col_mapping = self._col_mapping  # map single states to column index in probability rows
        end = self.eos
        _row_mapping = self._row_mapping  # map prior-state-sets to their row index in the master table
        _count_matrix = self._count_matrix

        # determine if there are new possible values (so we need to extend the mappings & matrices
        _new = get_unique_items(sequences, dtype=self.dtype).difference(set(_col_mapping.keys()))
        n_new = len(_new)
        if n_new > 0:
            temp_matrix = _count_matrix.copy()
            n_rows, n_cols = temp_matrix.shape
            # If there are new, split off the end proba column
            stop_col = temp_matrix[:,-1]
            # Add new columns at the end of the matrix & re-append the stop probabilities
            with_new_cols = np.append(temp_matrix[:,:-1], np.zeros((n_rows, n_new), dtype=np.int32), axis=1)
            _count_matrix = np.append(with_new_cols, stop_col.reshape(-1,1), axis=1)

            for new_state in _new:
                new_idx = max(_col_mapping.values()) + 1
                _col_mapping[cast_type(new_state, self.dtype)] = new_idx

        for sequence in sequences:
            sequence = np.array(_extend_sequence(sequence, end), dtype=self.dtype)
            # start proba - Update all start probas as the end rather than each time
            self._update_starts(start_sequence = sequence[:order], update_probas=False)
            # iterate through windows of order+1 size
            for subseq in window(sequence, order+1):
                prior_state = self._sequence_to_key(subseq[:order])
                col_idx = _col_mapping[subseq[-1]]

                if prior_state not in _row_mapping.keys():  # this is a new instance of the priors
                    # add to _row_mapping
                    new_row_idx = _count_matrix.shape[0]
                    _row_mapping[prior_state] = new_row_idx
                    # add a new row to the count matrix:
                    _count_matrix = _add_zeros_row(_count_matrix)
                    # append zeros row to _count_matrix
                    _count_matrix[new_row_idx, col_idx] = 1
                else:
                    _row_idx = _row_mapping[prior_state]
                    _count_matrix[_row_idx, col_idx] += 1

        # Update start probabilities now that I've run through them all
        self._update_starts(start_sequence=None, update_probas=True)
        # make sure all of the instance data is updated with what I've used here
        self._count_matrix = _count_matrix
        self._proba_matrix = _count_matrix/_count_matrix.sum(axis=1, keepdims=True)
        self._row_mapping = _row_mapping
        self._col_mapping = _col_mapping
        self.states_ = set(_col_mapping.keys())
        return self

    def generate_next(self, current_state, random_seed=None):
        """Randomly Select the next state for a sequence model based on the current state

        Parameters
        ----------
        current_state : object
            a object contained in the existing state mapping

        random_seed : integer
            random seed for numpy.random.seed()

        Returns
        -------
        next_object : an object selected based on the transitional probabilities from the given current_state
        """
        # Ensure the class instance is fitted
        if not hasattr(self, '_proba_matrix'):
            raise ValueError("Model not fitted.")
        # set random_seed
        if random_seed:
            np.random.seed(random_seed)

        if not isinstance(current_state, str):
            current_state = self._sequence_to_key(current_state)

        row_idx = self._row_mapping[current_state]
        probas = self._proba_matrix[row_idx]
        rev_cols_mapping = {value:key for key, value in self._col_mapping.items()}
        options = [rev_cols_mapping[x] for x in range(len(rev_cols_mapping)-1)] + [self.eos]
        return np.random.choice(options, p=probas)

    def create_sequence(self, starting_state=None, random_seed=None):
        """Create a random sequence using the trained transitional probability matrix.

        Parameters
        ----------
        starting_state : object (default=None)
            a object contained in the existing state mapping to use as the starting state.
            If None, one is randomly selected from the self._start_probas dictionary

        random_seed : integer
            random seed for numpy.random.seed()

        Returns
        -------
        sequence : a randomly generated sequence
        """
        # Ensure the class instance is fitted
        if not hasattr(self, '_proba_matrix'):
            raise ValueError("Model not fitted.")
        # set random_seed
        if random_seed:
            np.random.seed(random_seed)

        # select starting_state if one is not given
        if not starting_state:
            starts = []
            probas = []
            for key, val in self._start_probas.items():
                starts.append(key)
                probas.append(val[1])
            sequence = self._key_to_sequence(np.random.choice(starts, p=probas))
        else:
            if not isinstance(starting_state, str):
                starting_state = self._sequence_to_key(starting_state)
            sequence = self._key_to_sequence(starting_state)

        next_item = None
        while next_item != self.eos:
            if next_item is not None:
                sequence.append(next_item)
            prior = self._sequence_to_key(sequence[- self.order:])
            next_item = self.generate_next(prior)
        return sequence

    def subsequent_probability(self, current_state):
        """Returns the probabilities for the possible subsequent states given the current_state

        Parameters
        ----------
        current_state : object
            a sequence or single object contained in the existing state mapping

        Returns
        -------
        probabilities : a mapping of all possible subsequent states and their probabilities
        """
        # Ensure the class instance is fitted
        if not hasattr(self, '_proba_matrix'):
            raise ValueError("Model not fitted.")

        if not isinstance(current_state, str):
            current_state = self._sequence_to_key(current_state)

        row_idx = self._row_mapping[current_state]
        probas = self._proba_matrix[row_idx]
        rev_cols_mapping = {value:key for key, value in self._col_mapping.items()}
        options = [rev_cols_mapping[x] for x in range(len(rev_cols_mapping)-1)] + [self.eos]

        return {option: proba for option, proba in zip(options, probas) if proba > 0}

    def sequence_probability(self, sequence, return_all_steps=False, include_start_proba=True):
        """Returns the probability of seeing the exact passed sequence given the fitted transitional
        probability matrix

        return_all_steps: If true, (default=False) returns a list of all probabilities for each
            transitional step in the sequence

        include_start_proba : if true, (default=True) include the probability of seeing the starting sequence,
            otherwise start from the probability of the order+1 state in the sequence

        Returns
        -------
        sequence_probability : float
            the probability of the specified sequence occurring.

        all_probabilities : np.ndarray
            a list of all probabilities for the steps in the sequence.
        """
        _proba_matrix = self._proba_matrix
        _row_mapping = self._row_mapping
        _col_mapping = self._col_mapping
        sequence = np.array(sequence, dtype=self.dtype)

        if include_start_proba:
            _start_probas = self._start_probas  #self._start_probas = {initial_state: [count, proba]}
            start_key = self._sequence_to_key(sequence[:self.order])
            try:
                all_probabilities = np.array([_start_probas[start_key][1]], dtype=np.float64)
            except KeyError:
                all_probabilities = np.array([0.0], dtype=np.float64)
                warnings.warn('The sequence start was not found and therefore the probability is 0')
        else:
            all_probabilities = np.array([], dtype=np.float64)

        for step in window(_extend_sequence(sequence, self.eos), self.order+1):
            try:
                prior_state = self._sequence_to_key(step[:self.order])
                col_idx = _col_mapping[step[-1]]
                _row_idx = _row_mapping[prior_state]
                all_probabilities = _extend_sequence(all_probabilities, _proba_matrix[_row_idx, col_idx])
            except KeyError:
                all_probabilities = _extend_sequence(all_probabilities, 0.0)
                warnings.warn('The following was unable to be found: P({0} | {1})'.format(step[-1], prior_state))

        sequence_probability = np.cumprod(all_probabilities, dtype=np.float64)[-1]
        if return_all_steps:
            return sequence_probability, all_probabilities
        else:
            return sequence_probability

