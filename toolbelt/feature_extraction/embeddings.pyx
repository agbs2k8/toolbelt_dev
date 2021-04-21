#!python
#cython: language_level=3
import numbers
import numpy as np
from ..stats import get_unique_items


class EmbeddingVectorizer:
    """Convert a collection of log id lists to a either a padded 2D matrix of encodings
    -or-
    a 3D matrix of one-hot sequence embeddings (one-hot encoding applied to the 2D matrix)

    Parameters
    ----------
    embedding_len : int (default=None)
        When None, it creates embeddings based on the total number of unique items.  If specified, it will pad out
        each sequence to the provided length. (Only necessary when dealing with 3D output)

    sequence_len : int (default=None)
        When None, it each sequence embeddings is padded out to match the length of the longest sequence.  If otherwise
        specified, it will pad out to the indicated length.

    include_constant: bool (default=False)
        If true, will add an additional feature to every embedding, which will always be 1. (Only available in 3D data)


    Attributes
    ----------
    mapping_ : dict
        A mapping of log-types to index in the encoded logs.

    Examples
    --------
    >>> from toolbelt import EmbeddingVectorizer
    >>> host_log_list = [
    ...     ['4107', '4107', '4107', '4107', '4107', '4107'],
    ...     ['7610', '7610', '7610', '7610'],
    ...     ['4107', '4107'],
    ...     ['7610'],
    ... ]
    >>> vectorizer = EmbeddingVectorizer()
    >>> X = vectorizer.fit_transform(host_log_list)
    >>> print(vectorizer.logs_)
    ['4107', '7610']
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    [[6 0]
     [0 4]
     [2 0]
     [0 1]]
    """
    # TODO: DECODE - Turn the embeddings back into sequences
    def __init__(self, embedding_len=None, sequence_len=None, include_constant=False):
        if embedding_len is not None:
            if not isinstance(embedding_len, numbers.Integral):
                raise ValueError('embedding_len must be integer or None')
        self.embedding_len = embedding_len

        if sequence_len is not None:
            if not isinstance(sequence_len, numbers.Integral):
                raise ValueError('sequence_len must be integer or None')
        self.sequence_len = sequence_len
        self.include_constant = include_constant

    def fit(self, sequences):
        """Learn the mapping of sequence item to array position, calculate the embedding_len and sequence_len

        Parameters
        ----------
        sequences : iterable
            A list of sequences

        Returns
        -------
        self
        """
        embedding_len = self.embedding_len
        sequence_len = self.sequence_len

        # find unique items & count them
        unique_items = get_unique_items(sequences)
        if embedding_len is not None and len(unique_items) > embedding_len:
            raise ValueError('Provided embedding length is less than the number of unique items.')
        elif embedding_len is None:
            embedding_len = len(unique_items)
        if self.include_constant:
            embedding_len += 1

        # find sequence length
        longest_sequence = max([len(x) for x in sequences])
        if sequence_len is not None and longest_sequence > sequence_len:
            raise ValueError('Provided sequence length is shorter than the longest sequence.')
        elif sequence_len is None:
            sequence_len = longest_sequence

        # create embedding / mapping dict
        mapping_ = {log_id:idx for idx, log_id in enumerate(unique_items)}

        # write all to self. something
        self.embedding_len = embedding_len
        self.sequence_len = sequence_len
        self.mapping_ = mapping_
        return self


    def transform(self, sequences, output_dim=3):
        """Transform lists of sequences to 3D array of padded sequence embeddings based on parameters
        determined during self.fit()

        Parameters
        ----------
        sequences : iterable
            A list of sequences

        output_dim : 2 or 3
            2 returns a 2D matrix where the rows are padded to sequence_len but consists of the integer values from
                mapping_ rather that the original input.
            3 returns a 3D matrix the second dimension is padded to sequence_len, the 3rd dimension is size of
                len(embedding_len), and the 3rd dimension contains one-hot encodings where the position is equal to the
                integer found in the 2D output/mapping_.

        Returns
        -------
        X : array, [n_samples, sequence_len, embedding_len]
        """
        # Make sure fit has run (there must be a mapping)
        if not hasattr(self, 'mapping_'):
            raise ValueError('Must fit before attempting to transform data.')
        # make sure we have a valid dimension for output:
        if not output_dim in (2,3):
            raise ValueError("output_dim can only be [2,3]. Higher/lower dimensional output not supported.")
        # get all of the goodies we created in fit
        embedding_len = self.embedding_len
        sequence_len = self.sequence_len
        mapping_ = self.mapping_
        all_encodings = []
        for sequence in sequences:
            sequence_list = []
            # Build Encodings
            for log in sequence:
                if output_dim == 3:
                    log_vector = np.zeros(embedding_len, dtype=np.int8)
                    log_vector[mapping_[log]] = 1
                    if self.include_constant:
                        log_vector[-1] = 1
                    sequence_list.append(log_vector)
                elif output_dim == 2:
                    sequence_list.append(mapping_[log])
            # pad with zeros as needed
            while len(sequence_list) < sequence_len:
                if output_dim == 3:
                    log_vector = np.zeros(embedding_len, dtype=np.int8)
                    if self.include_constant:
                        log_vector[-1] = 1
                    sequence_list.insert(0, log_vector)
                elif output_dim == 2:
                    sequence_list.insert(0, 0)
            all_encodings.append(sequence_list)
        return np.array(all_encodings)

    def fit_transform(self, sequences, output_dim=3):
        """Learn the mapping of sequence item to array position, calculate the embedding_len and sequence_len.
        Then transform lists of sequences to 3D array of padded sequence embeddings based on those values.

        Parameters
        ----------
        sequences : iterable
            A list of sequences

        output_dim : 2 or 3
            2 returns a 2D matrix where the rows are padded to sequence_len but consists of the integer values from
                mapping_ rather that the original input.
            3 returns a 3D matrix the second dimension is padded to sequence_len, the 3rd dimension is size of
                len(embedding_len), and the 3rd dimension contains one-hot encodings where the position is equal to the
                integer found in the 2D output/mapping_.

        Returns
        -------
        X : array, [n_samples, sequence_len, embedding_len]
        """
        self.fit(sequences)
        return self.transform(sequences, output_dim)
