#!python
#cython: language_level=3
# Standard Library
import numbers
import array
from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations
import warnings

# Outside Packages
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
from sklearn.preprocessing import normalize
import scipy
import scipy.sparse as sp
import numpy as np

from ..utils import window


# Direct copy -- from sklearn.utils.fixes import _astype_copy_false
def _parse_version(version_string):
    # Direct copy -- from sklearn.utils.fixes import _astype_copy_false
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)
sp_version = _parse_version(scipy.__version__)
def _astype_copy_false(X):
    # Direct copy -- from sklearn.utils.fixes import _astype_copy_false
    """Returns the copy=False parameter for
    {ndarray, csr_matrix, csc_matrix}.astype when possible,
    otherwise don't specify
    """
    if sp_version >= (1, 1) or not sp.issparse(X):
        return {'copy': False}
    else:
        return {}


def _host_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)


def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))


class CountVectorizer(BaseEstimator):
    """Convert a collection of log id lists to a matrix of log counts

    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    Parameters
    ----------
    max_hf : float in range [0.0, 1.0] or int (default=1.0)
        When building the log ID set ignore logs that have a host
        frequency strictly higher than the given threshold.

        If float, the parameter represents a proportion of hosts, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_hf : float in range [0.0, 1.0] or int (default=1)
        When building the log ID set ignore logs that have a host
        frequency strictly lower than the given threshold.

        If float, the parameter represents a proportion of hosts, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int or None (default=None)
        If not None, build a log set that only consider the top
        max_features ordered by log frequency across the hosts.

        This parameter is ignored if vocabulary is not None.

    logs : Mapping or iterable, optional (default=None)
        Either a Mapping (e.g., a dict) where keys are log IDs and values are
        indices in the feature matrix, or an iterable over log IDs. If not
        given, a vocabulary is determined from the input documents.

    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    ngram_value : integer (default=1) the number of logs, taken in order, to combine into a feature

    Attributes
    ----------
    logs_ : dict
        A mapping of terms to feature indices.

    unused_logs_: set
        Log IDs that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).

        This is only available if no logs were given.

    Examples
    --------
    >>> from toolbelt.feature_extraction import CountVectorizer
    >>> host_log_list = [
    ...     ['4107', '4107', '4107', '4107', '4107', '4107'],
    ...     ['7610', '7610', '7610', '7610'],
    ...     ['4107', '4107'],
    ...     ['7610'],
    ... ]
    >>> vectorizer = CountVectorizer()
    >>> X = vectorizer.fit_transform(host_log_list)
    >>> print(vectorizer.logs_)
    ['4107', '7610']
    >>> print(X.toarray())  # doctest: +NORMALIZE_WHITESPACE
    [[6 0]
     [0 4]
     [2 0]
     [0 1]]
    """

    def __init__(self, max_hf=1.0, min_hf=1, max_features=None, logs=None, dtype=np.int64, ngram_value=1):
        self.max_hf = max_hf
        self.min_hf = min_hf
        if max_hf < 0 or min_hf < 0:
            raise ValueError("negative value for max_hf or min_hf")
        self.max_features = max_features
        if max_features is not None:
            # numbers.Integral is the class from which int, numpy.uint, etc is derived
            if not isinstance(max_features, numbers.Integral) or max_features <= 0:
                raise ValueError("max_features={}, neither a positive integer nor None".format(max_features))
        self.logs = logs
        self.dtype = dtype
        self.ngram_value = ngram_value
        if not isinstance(ngram_value, numbers.Integral):
            raise ValueError('ngram value must be an integer.')

    @staticmethod
    def _sort_logs(X, logs):
        """Sort features by name

        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(logs.items())
        map_index = np.empty(len(sorted_features), dtype=X.indices.dtype)
        for new_val, (term, old_val) in enumerate(sorted_features):
            logs[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode='clip')
        return X

    def _validate_logs(self):
        logs = self.logs
        if logs is not None:
            if isinstance(logs, set):
                logs = sorted(logs)
            if not isinstance(logs, Mapping):
                rls = {}
                for i, t in enumerate(logs):
                    if rls.setdefault(t, i) != i:
                        msg = "Duplicate log IDs in log: %r" % t
                        raise ValueError(msg)
                logs = rls
            else:
                indices = set(logs.values())
                if len(indices) != len(logs):
                    raise ValueError("Logs contains repeated IDs.")
                for i in range(len(logs)):
                    if i not in indices:
                        msg = ("Log of size %d doesn't contain index "
                               "%d." % (len(logs), i))
                        raise ValueError(msg)
            if not logs:
                raise ValueError("empty logs passed to fit")
            self.fixed_logs_ = True
            self.logs_ = dict(logs)
        else:
            self.fixed_logs_ = False

    def _check_logs(self):
        """Check if logs mapping is empty or missing (not fit-ed)"""
        msg = "%(name)s - Logs weren't fitted."
        check_is_fitted(self, 'logs_', msg=msg),

        if len(self.logs_) == 0:
            raise ValueError("Logs are empty")

    def _limit_features(self, X, logs, high=None, low=None, limit=None):
        """Remove too rare or too common features.

        Prune features that are non zero in more samples than high or less
        host than low, modifying the log set, and restricting it to
        at most the limit most frequent.

        This does not prune samples with zero features.
        """
        if high is None and low is None and limit is None:
            return X, set()

        # Calculate a mask based on document frequencies
        hfs = _host_frequency(X)
        rfs = np.asarray(X.sum(axis=0)).ravel()
        mask = np.ones(len(hfs), dtype=bool)
        if high is not None:
            mask &= hfs <= high
        if low is not None:
            mask &= hfs >= low
        if limit is not None and mask.sum() > limit:
            mask_inds = (-rfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(hfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for log_id, old_index in list(logs.items()):
            if mask[old_index]:
                logs[log_id] = new_indices[old_index]
            else:
                del logs[log_id]
                removed_terms.add(log_id)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no logs remain. Try a lower"
                             " min_hf or a higher max_hf.")
        return X[:, kept_indices], removed_terms

    def _count_logs(self, log_lists, fixed_logs):
        """Create sparse feature matrix, and logs where fixed_logs=False
        """
        if fixed_logs:
            logs = self.logs_
        else:
            # Add a new value when a new log id is seen
            logs = defaultdict()
            logs.default_factory = logs.__len__

        j_indices = []
        indptr = []

        values = _make_int_array()
        indptr.append(0)
        for host_logs in log_lists:  # Iterate through each host's list of logs
            log_counter = {}
            # make sure they are all strings
            host_logs = [str(x) for x in host_logs]
            for log_id in window(host_logs, self.ngram_value):
                if len(log_id) == self.ngram_value:
                    log_id = '|'.join(log_id)
                    try:
                        log_idx = logs[log_id]
                        if log_idx not in log_counter:
                            log_counter[log_idx] = 1
                        else:
                            log_counter[log_idx] += 1
                    except KeyError:
                        # Ignore out-of-vocabulary items for fixed_vocab=True
                        continue

            j_indices.extend(log_counter.keys())
            values.extend(log_counter.values())
            indptr.append(len(j_indices))

        if not fixed_logs:
            # disable defaultdict behaviour
            logs = dict(logs)
            if not logs:
                raise ValueError("empty logs; perhaps the host had no logs fire")

        if indptr[-1] > 2147483648:  # = 2**31 - 1
            indices_dtype = np.int64

        else:
            indices_dtype = np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(logs)),
                          dtype=self.dtype)
        X.sort_indices()

        if self.ngram_value > 1:  # if using n-grams, add a constant col to avoid div by 0 with LF-IHF
            X = sp.csr_matrix(np.append(X.toarray(), np.ones((X.shape[0], 1)), axis=1), dtype=np.float64)
            logs['const'] = max(logs.values()) + 1

        return logs, X

    def fit_transform(self, log_lists, y=None):
        """Learn the logs dictionary and return log-host matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        log_lists : iterable
            A list of lists of logs

        Returns
        -------
        X : array, [n_samples, n_features]
            Log-host matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # LfihfVectorizer.

        self._validate_logs()
        max_hf = self.max_hf
        min_hf = self.min_hf
        max_features = self.max_features

        logs, X = self._count_logs(log_lists, self.fixed_logs_)

        if not self.fixed_logs_:
            X = self._sort_logs(X, logs)

            n_hosts = X.shape[0]
            max_host_count = (max_hf
                              if isinstance(max_hf, numbers.Integral)
                              else max_hf * n_hosts)
            min_host_count = (min_hf
                              if isinstance(min_hf, numbers.Integral)
                              else min_hf * n_hosts)
            if max_host_count < min_host_count:
                raise ValueError(
                    "max_hf corresponds to < hosts than min_hf")
            X, self.unused_logs_ = self._limit_features(X, logs,
                                                       max_host_count,
                                                       min_host_count,
                                                       max_features)

            self.logs_ = logs

        return X

    def fit(self, log_lists, y=None):
        """Learn the logs dictionary and return log-host matrix.

        Parameters
        ----------
        log_lists : iterable
            A list of lists of logs

        Returns
        -------
        self
        """
        self.fit_transform(log_lists)
        return self

    def transform(self, log_lists):
        """Transform lists of logs to host-log matrix.

        Parameters
        ----------
        log_lists : iterable
            A list of lists of logs

        Returns
        -------
        X : array, [n_samples, n_features]
            Log-host matrix.
        """

        if not hasattr(self, 'logs_'):
            self._validate_logs()

        self._check_logs()

        # use the same matrix-building strategy as fit_transform
        _, X = self._count_logs(log_lists, fixed_logs=True)
        return X


class LfihfTransformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a normalized hf or lf-ihf representation

        Lf means log-frequency while lf-ihf means log-frequency times inverse
        host-frequency. This is adapted from the common TF-IDF weighting scheme in
        information retrieval, that has also found good use in document classification.

        The goal of using lf-ihf instead of the raw frequencies of occurrence of a
        log on a given host is to scale down the impact of logs that occur
        very frequently in a given set of hosts and that are hence empirically less
        informative than logs that occur in a small fraction of the hosts.

        The formula that is used to compute the lf-ihf for a log l of a host h
        in a host set is lf-ihf(l, h) = lf(l, h) * ihf(l), and the ihf is
        computed as ihf(l) = log [ n / lf(l) ] + 1 (if ``smooth_ihf=False``), where
        n is the total number of hosts in the host set and lf(h) is the
        log frequency of l; the host frequency is the number of hosts
        in the host set that contain the log l. The effect of adding "1" to
        the ihf in the equation above is that logs with zero ihf, i.e., logs
        that occur in all hosts in a training set, will not be entirely
        ignored.
        (Note that the ihf formula above differs from the standard textbook
        notation for TF-IDF which defines the IDF as:
        idf(t) = log [ n / (df(t) + 1) ]).

        If ``smooth_ihf=True`` (the default), the constant "1" is added to the
        numerator and denominator of the ihf as if an extra host was seen
        containing every log in the collection exactly once, which prevents
        zero divisions: idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1.

        Furthermore, the formulas used to compute lf and ihf depend
        on parameter settings that correspond to the SMART notation used in IR
        as follows:

        Lf is "n" (natural) by default, "l" (logarithmic) when
        ``sublinear_lf=True``.
        Ihf is "t" when use_ihf is given, "n" (none) otherwise.
        Normalization is "c" (cosine) when ``norm='l2'``, "n" (none)
        when ``norm=None``.

        Parameters
        ----------
        norm : 'l1', 'l2' or None, optional (default='l2')
            Each output row will have unit norm, either:
            * 'l2': Sum of squares of vector elements is 1. The cosine
            similarity between two vectors is their dot product when l2 norm has
            been applied.
            * 'l1': Sum of absolute values of vector elements is 1.

        use_ihf : boolean (default=True)
            Enable inverse-host-frequency reweighting.

        smooth_ihf : boolean (default=True)
            Smooth ihf weights by adding one to host frequencies, as if an
            extra host was seen containing every log in the collection
            exactly once. Prevents zero divisions.

        lf_type : 'natural', 'log', 'bool', 'pct', (default='natural')
            * 'natural' : the actual count is used for lf
            * 'log' : sublinear lf scaling: = 1 + log(lf)
            * 'bool' : 1 if lf > 0 else 0
            * 'pct' : scaled by the length of the host's log list = pct of logs for the host

        Attributes
        ----------
        ihf_ : array, shape (n_features)
            The inverse host frequency (IHF) vector; only defined
            if  ``use_ihf`` is True.
        """

    def __init__(self, norm='l2', use_ihf=True, smooth_ihf=True,
                 lf_type='natural'):
        self.norm = norm
        self.use_ihf = use_ihf
        self.smooth_ihf = smooth_ihf
        self.lf_type = lf_type

    def fit(self, X, y=None):
        """Learn the ihf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of host/log counts
        """
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_ihf:
            n_samples, n_features = X.shape
            hf = _host_frequency(X)
            hf = hf.astype(dtype, **_astype_copy_false(hf))

            # perform ihf smoothing if required
            hf += int(self.smooth_ihf)
            n_samples += int(self.smooth_ihf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            ihf = np.log(n_samples / hf) + 1
            self._ihf_diag = sp.diags(ihf, offsets=0,
                                      shape=(n_features, n_features),
                                      format='csr',
                                      dtype=dtype)
        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a lf or lf-ihf representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of host/log counts

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        n_samples, n_features = X.shape

        # if it is natural, do nothing
        if self.lf_type == 'log':
            np.log(X.data, X.data)
            X.data += 1
        elif self.lf_type == 'bool':
            np.clip(X.data, 0, 1, X.data)
        elif self.lf_type == 'pct':
            X = sp.csr_matrix(np.divide(X.toarray(), X.toarray().sum(axis=1, keepdims=True, dtype=np.float64)),
                              dtype=np.float64)

        if self.use_ihf:
            check_is_fitted(self, '_ihf_diag', 'ihf vector is not fitted')

            expected_n_features = self._ihf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._ihf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def ihf_(self):
        # if _ihf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "ihf_") is False
        return np.ravel(self._ihf_diag.sum(axis=0))

    @ihf_.setter
    def ihf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self._ihf_diag = sp.spdiags(value, diags=0, m=n_features,
                                    n=n_features, format='csr')


class LfihfVectorizer(CountVectorizer):
    """Convert a collection of log IDs to a matrix of Log Frequency - Inverse Host Frequency (LF-IHF) features.

    Parameters
    ----------
    max_hf : float in range [0.0, 1.0] or int (default=1.0)
        When building the log ID set ignore logs that have a host
        frequency strictly higher than the given threshold.

        If float, the parameter represents a proportion of hosts, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_hf : float in range [0.0, 1.0] or int (default=1)
        When building the log ID set ignore logs that have a host
        frequency strictly lower than the given threshold.

        If float, the parameter represents a proportion of hosts, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int or None (default=None)
        If not None, build a log set that only consider the top
        max_features ordered by log frequency across the hosts.

        This parameter is ignored if vocabulary is not None.

    logs : Mapping or iterable, optional (default=None)
        Either a Mapping (e.g., a dict) where keys are log IDs and values are
        indices in the feature matrix, or an iterable over log IDs. If not
        given, a vocabulary is determined from the input documents.

    norm : 'l1', 'l2' or None, optional (default='l2')
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied.
        * 'l1': Sum of absolute values of vector elements is 1.

    use_ihf : boolean (default=True)
        Enable inverse-host-frequency re-weighting.

    smooth_ihf : boolean (default=True)
        Smooth ihf weights by adding one to log frequencies, as if an
        extra host was seen containing every log in the collection
        exactly once. Prevents zero divisions.

    lf_type : 'natural', 'log', 'bool', 'pct', (default='natural')
        How the log frequency will be calculated / normalized
            * 'natural' : the actual count of logs
            * 'log' : sublinear lf scaling: = 1 + log(lf)
            * 'bool' : 1 if lf > 0 else 0
            * 'pct' : scaled by the length of the host's log list = pct of logs for the host

    ngram_value : integer (default=1) the number of logs, taken in order, to combine into a feature

    Attributes
    ----------
    logs_ : dict
        A mapping of terms to feature indices.

    ihf_ : array, shape (n_features)
        The inverse host frequency (IDF) vector; only defined
        if ``use_ihf`` is True.


    Examples
    --------
    >>> from toolbelt.feature_extraction import LfihfVectorizer
    >>> host_log_list = [
    ...     ['4107', '4107', '4107', '4107', '4107', '4107'],
    ...     ['7610', '7610', '7610', '7610'],
    ...     ['4107', '4107'],
    ...     ['7610'],
    ... ]
    >>> vectorizer = LfihfVectorizer()
    >>> X = vectorizer.fit_transform(host_log_list)
    >>> print(vectorizer.logs_)
    ['4107', '7610']
    >>> print(X.shape)
    (4, 2)
    """

    def __init__(self, max_hf=1.0, min_hf=1, max_features=None, logs=None, dtype=np.float64, norm='l2', use_ihf=True,
                 smooth_ihf=True, lf_type='natural', ngram_value=1):
        super().__init__(max_hf=max_hf, min_hf=min_hf, max_features=max_features, logs=logs, dtype=dtype,
                         ngram_value=ngram_value)

        self._lfihf = LfihfTransformer(norm=norm, use_ihf=use_ihf, smooth_ihf=smooth_ihf, lf_type=lf_type)
    # Broadcast the LF-IHF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._lfihf.norm

    @norm.setter
    def norm(self, value):
        self._lfihf.norm = value

    @property
    def use_ihf(self):
        return self._lfihf.use_ihf

    @use_ihf.setter
    def use_ihf(self, value):
        self._lfihf.use_ihf = value

    @property
    def smooth_ihf(self):
        return self._lfihf.smooth_ihf

    @smooth_ihf.setter
    def smooth_ihf(self, value):
        self._lfihf.smooth_ihf = value

    @property
    def lf_type(self):
        return self._lfihf.lf_type

    @lf_type.setter
    def lf_type(self, value):
        self._lfihf.lf_type = value

    @property
    def ihf_(self):
        return self._lfihf.ihf_

    @ihf_.setter
    def ihf_(self, value):
        self._validate_logs()
        if hasattr(self, 'logs_'):
            if len(self.logs_) != len(value):
                raise ValueError("ihf length = %d must be equal "
                                 "to log size = %d" %
                                 (len(value), len(self.logs)))
        self._lfihf.ihf_ = value

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will "
                          "be converted to np.float64."
                          .format(FLOAT_DTYPES, self.dtype),
                          UserWarning)

    def fit(self, log_lists, y=None):
        """Learn logs and ihf from training set.

        Parameters
        ----------
        log_lists : iterable
            a list of lists of logs for each host

        Returns
        -------
        self : LfihfVectorizer
        """
        self._check_params()
        X = super().fit_transform(log_lists)
        self._lfihf.fit(X)
        return self

    def fit_transform(self, log_lists, y=None):
        """Learn logs and ihf, return log-host matrix.

        Parameters
        ----------
        log_lists : iterable
            a list of lists of logs for each host

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Lf-ihf-weighted log-host matrix.
        """
        self._check_params()
        X = super().fit_transform(log_lists)
        self._lfihf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._lfihf.transform(X, copy=False)

    def transform(self, log_lists, copy=True):
        """Transform log_lists to log-host matrix.

        Uses the logs and host frequencies (hf) learned by fit (or
        fit_transform).

        Parameters
        ----------
        log_lists : iterable
            a list of lists of logs for each host

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Lf-ihf-weighted log-host matrix.
        """
        check_is_fitted(self, '_lfihf', 'The lfihf vector is not fitted')

        X = super().transform(log_lists)
        return self._lfihf.transform(X, copy=False)

'''
class AprioriVectorizer(BaseEstimator):
    """Convert a collection of log IDs to a matrix of Apriori features.

    Parameters
    ----------
    max_set_size : int (default=5)
        When building the possible Apriori sets, the max length allowed

    min_set_size : int (default=2)
        When building the possible Apriori sets, the min length required

    support_threshold : float in range [0.0, 1.0] (default=0.75)
        When selecting subsets ignore those with a support level lower than
        the given threshold.  The parameter represents a proportion of hosts,\


    Attributes
    ----------
    ap_sets_ : dict
        A mapping of log-sets to feature indices.


    Examples
    --------
    TODO: Fix Example
    >>> from toolbelt.feature_extraction import AprioriVectorizer
    >>> host_log_list = [
    ...     ['4107', '4107', '4107', '4107', '4107', '7610'],
    ...     ['7610', '7610', '7610', '7610', '4107',],
    ...     ['4107', '4107'],
    ...     ['7610'],
    ... ]
    >>> vectorizer = AprioriVectorizer()
    >>> X = vectorizer.fit_transform(host_log_list)
    >>> print(vectorizer.ap_sets_)
    {'4107|7610': 0}
    >>> print(X.toarray())
    array([ [1],
            [1],
            [0],
            [0] ])
    """

    def __init__(self, max_set_size=5, min_set_size=2, support_threshold=0.10):
        self.max_set_size = max_set_size
        if not isinstance(max_set_size, numbers.Integral) or max_set_size < 0:
            raise ValueError("max_set_size must be an integer > 0")
        self.min_set_size = min_set_size
        if not isinstance(min_set_size, numbers.Integral) or min_set_size < 0:
            raise ValueError("min_set_size must be an integer > 0")
        self.support_threshold = support_threshold

    @staticmethod
    def _lists_to_sets(log_lists):
        return [set(l) for l in log_lists]

    @staticmethod
    def set_to_string(s):
        return '|'.join([x for x in quicksort(list(s))])

    @staticmethod
    def string_to_set(s):
        return set([x for x in s.split('|')])

    @staticmethod
    def all_subsets(s, min_len=2, max_len=5):
        combos = []
        range_max = min(len(s) + 1, max_len + 1)
        for l in range(min_len, range_max):
            combos += [set(x) for x in combinations(s, l)]
        return combos

    def _apriori_sets(self, sets: list):
        min_setlen = self.min_set_size
        max_setlen = self.max_set_size
        min_threshold = self.support_threshold
        num_sets = len(sets)
        results = dict()
        while len(sets) > 0:
            current_set = sets.pop(0)
            current_subsets = self.all_subsets(current_set, min_setlen, max_setlen)
            for _subset in current_subsets:
                _subset_str = self.set_to_string(_subset)
                if _subset_str in results.keys():
                    continue
                else:
                    results[_subset_str] = 1
                    if len(sets) > 0:
                        for other_set in sets:
                            if _subset.issubset(other_set):
                                results[_subset_str] += 1
        result_dict = dict()
        i = 0
        for key, count in results.items():
            if float(count) / num_sets >= min_threshold:
                result_dict[key] = i
                i += 1
        self.ap_sets_ = result_dict
        return result_dict

    def _count_sets(self, sets_list, ap_sets=None):
        if not ap_sets:
            ap_sets = self.ap_sets_
        reverse_mapping = {val: key for key, val in ap_sets.items()}
        cols = {reverse_mapping[x] for x in range(len(reverse_mapping.keys()))}
        rows = []
        for row_set in sets_list:
            row_data = []
            for col_str in cols:
                col_set = self.string_to_set(col_str)
                if col_set.issubset(row_set):
                    row_data.append(1)
                else:
                    row_data.append(0)
            rows.append(row_data)
        return np.array(rows)

    def fit_transform(self, logs_list):
        log_sets = self._lists_to_sets(logs_list)
        ap_sets = self._apriori_sets(log_sets)
        return self._count_sets(log_sets, ap_sets)
'''
