#!python
#cython: language_level=3

import numbers
import itertools
import numpy as np
cimport numpy as np
from ..utils import quicksort


class AprioriVectorizer:
    def __init__(self, max_set_size=5, min_set_size=2, support_threshold=0.25):
        self.max_set_size = max_set_size
        if not isinstance(max_set_size, numbers.Integral) or max_set_size < 2:
            raise ValueError("max_set_size must be an integer >= 2")
        self.min_set_size = min_set_size
        if not isinstance(min_set_size, numbers.Integral) or min_set_size < 2:
            raise ValueError("min_set_size must be an integer >= 2")
        self.support_threshold = support_threshold

    @staticmethod
    def _set_to_string(s):
        return '|'.join([str(x) for x in quicksort(list(s))])

    @staticmethod
    def _lists_to_sets(list_of_lists):
        return [set(l) for l in list_of_lists]

    @staticmethod
    def _get_unique_items(list_of_sets):
        return set([item for sublist in list_of_sets for item in sublist])

    @staticmethod
    def _meets_threshold(item, all_sets, min_instances):
        instance_count = 0
        if isinstance(item, set):
            for master_set in all_sets:
                if item.issubset(master_set):
                    instance_count += 1
                if instance_count >= min_instances:
                    return True
            return False
        else:
            for master_set in all_sets:
                if item in master_set:
                    instance_count += 1
                if instance_count >= min_instances:
                    return True

            return False

    @staticmethod
    def _make_initial_pairs_list(valid_items):
        return [set(x) for x in itertools.combinations(valid_items, 2)]

    @staticmethod
    def _add_to_set(_set, item):
        _set = _set.copy()
        _set.add(item)
        return _set

    @staticmethod
    def _index_to_set_mapping(ap_sets):
        mapping = dict()
        for idx, _set in enumerate(ap_sets):
            mapping[idx] = _set
        return mapping

    @staticmethod
    def _encode_from_mapping(data_sets, mapping):
        matrix = np.zeros(shape=(len(data_sets), len(mapping)))
        for c_idx in range(len(mapping)):
            subset = mapping[c_idx]
            for r_idx, row_set in enumerate(data_sets):
                if subset.issubset(row_set):
                    matrix[r_idx, c_idx] = 1
        return matrix

    def _reduce_singles(self, single_items, master_sets, min_instances):
        usable_items = set()
        for item in single_items:
            if self._meets_threshold(item, master_sets, min_instances):
                usable_items.add(item)
        return usable_items

    def _make_next_batch(self, last_batch, new_batch_item_len, items_to_add):
        new_sets = []
        for old_set in last_batch:
            for item in items_to_add:
                _set = self._add_to_set(old_set, item)
                if len(_set) == new_batch_item_len and _set not in new_sets:
                    new_sets.append(_set)
        return new_sets

    def _make_apriori_sets(self, data_sets, min_set_size, max_set_size, support_threshold):
        # initial calculations & object creation
        min_instances = int(len(data_sets) * support_threshold)
        items_over_threshold = []

        # step 1: reduce list of lists to list of sets
        # data_sets = _lists_to_sets(data)

        # setp 2: find unique items
        unique_items = self._get_unique_items(data_sets)

        # step 3: find unique items that meet the threshold
        unique_items_to_use = self._reduce_singles(unique_items, data_sets, min_instances)

        # step 4: build initial pairs
        initial_pairs = self._make_initial_pairs_list(unique_items_to_use)

        # step 5: find pairs that meet threshold and add to a list of things to keep
        valid_items = [pair for pair in initial_pairs if self._meets_threshold(pair, data_sets, min_instances)]

        # step 6: if pairs are included in the output, add them to the overall output
        if min_set_size == 2:
            items_over_threshold += valid_items

        last_batch = valid_items.copy()
        # Until I've checked the max set length or no longer have sets to check:
        for batch_length in range(3, max_set_size + 1):
            # append new items to prior run's valid sets to create new sets to check
            working_batch = self._make_next_batch(last_batch, batch_length, unique_items_to_use)

            # check sets and keep those which meet the threshold
            valid_items = [_set for _set in working_batch if self._meets_threshold(_set, data_sets, min_instances)]

            if max_set_size >= batch_length and len(valid_items) > 0:
                items_over_threshold += valid_items
            elif len(valid_items) == 0:
                break

            # use the valid sets for the next run
            last_batch = valid_items.copy()
        return items_over_threshold

    def _make_str_set_idx_mapping(self, idx_set_mapping):
        return {self._set_to_string(value): key for key, value in idx_set_mapping.items()}

    def fit(self, list_of_lists, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.
        Parameters
        ----------
        list_of_lists : iterable
            An iterable of iterables that can be reduced to sets.
        Returns
        -------
        self
        """
        self.fit_transform(list_of_lists)
        return self

    def fit_transform(self, list_of_lists, y=None):
        """Learn the Apriori Set Mapping and return vectorizer matrix.
        ----------
        list_of_lists : iterable
            An iterable of iterables that can be reduced to sets.
        Returns
        -------
        X : array, [n_samples, n_features]
        """

        # self._validate_params()
        # self._validate_vocabulary()
        max_set_size = self.max_set_size
        min_set_size = self.min_set_size
        support_threshold = self.support_threshold

        data_sets = self._lists_to_sets(list_of_lists)
        ap_sets = self._make_apriori_sets(data_sets, min_set_size, max_set_size, support_threshold)
        mapping = self._index_to_set_mapping(ap_sets)
        self.mapping_ = mapping
        return self._encode_from_mapping(data_sets, mapping)

    def transform(self, list_of_lists):
        """Transform data using pre-fitted mapping.
        Parameters
        ----------
        list_of_lists : iterable
            An iterable of iterables that can be reduced to sets.
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
        """

        if not hasattr(self, 'mapping_'):
            raise ValueError("Vectorizer not fitted")
            # self._validate_vocabulary()

        # self._check_vocabulary()
        data_sets = self._lists_to_sets(list_of_lists)
        mapping = self.mapping_
        return self._encode_from_mapping(data_sets, mapping)

    def get_mapping(self):
        if not hasattr(self, 'mapping_'):
            raise ValueError("Vectorizer not fitted")
        return self._make_str_set_idx_mapping(self.mapping_)
