# toolbelt
This is a collection of tools that I have built up over many years.  Some are better than others.  Many are in .pyx
files to take advantage of Cython's speedups. 

## Subsections:

### utils
quicksort(xs) : standard, recrusive quicksort implementation to sort iterable xs
    
batch(iterable, n: int = 1) : yields portions of iterable in len(n) batches.  No overlap, just splits it up 
    
window(sequence, n: int = 5) : steps through sequence and yields subset of size n (overlapping slices)

edit_distance(str_a, str_b) : Levenshtein Distance between the two strings


### stats
visualize_distribution(data) : Generates a 2x2 plot of KDE, ECDF, Violin and Histogram for passed data series

test_stationarity(timeseries, periods=12) : Plots rolling mean and std

bicbic(kmeans, X) : Bayesian Information Criterion for clusters given fitted kmeans clustering object
and numpy array of original X values that were used to fit kmeans

cramer_v(x, y) : Cramer's V coefficient - Symmetrical correlation between two categorical variables

conditional_entropy(x, y) : Calculates the conditional entropy of x given y: S(x|y)

theil_u(x, y): Theil's Uncertainty Coefficient - the uncertainty of x given y: value is on the range of [0,1] 
where 0 means y provides no information about x, and 1 means y provides full information about x

corr_ratio(categories, continuous) : Given a continuous number, how well can you know to which category it belongs to?
Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement, and 1 means
a category can be determined with absolute certainty.

MarkovChain : class with multiple methods.  fit() with a list of sequences.  Preserves the transitional probability 
matrix, can give subsequent probabilities, probability of observing a given sequence, or generate new sequences

SequenceModel : similar to MarkovChain except it will take order parameter to consider more than 1 prior state. Not as 
robust as MarkovChain in what other methods are available.


### trees

Tree, Node, read_tree: base implementation of a tree wherein it has a single starting node which can branch off to 1:n 
possible other nodes. Methods are available for comparison, subtrees, plotting, preservation to disk, etc. 

Master, Host, ProcessTree, Process : extension of the Tree and Node to account for processes calling child processes. 
The Host can keep n process trees seen on that host, and the master keeps a repository of unique ProcessTree instances
seen across the Hosts it represents.  Also has read/write methods:
read_process_tree, read_host, read_master, find_all_matches, build_master_from_hosts

### feature_extraction
These all deal with sequences of logs much like what the MarkovChain & SequenceModel can use to fit, however these 
classes are for aggregating and encoding the information into a 2D format that can be used for ML, wherein each row is 
a given host and the columns are aggregated data about the logs we saw for that host. 
CountVectorizer : It aggregates by counting the raw number of occurrences by log type
LfihfVectorizer : Log Frequency - Inverse Host Frequency. Read the paper if you don't know what it is.
AprioriVectorizer : Counts the occurrences of the most common subsets of logs. 


## Installation
It should pip-install now!  Run the following:
```bash
pip install <path to package>
```
For best results, download the repo and point to the root cysiv-toolbelt folder where setup.py exists with that pip 
install command.