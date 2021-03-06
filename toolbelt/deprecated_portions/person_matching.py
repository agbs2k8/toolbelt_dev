import os
import pickle
import pandas as pd
import numpy as np
import multiprocessing as mp
from keras.models import load_model
from fuzzywuzzy import fuzz
from tqdm import tqdm
from .distribute_processing import multiprocess_function

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

nnet_model = load_model(os.path.join(__location__,'keras_person_model.h5'))
scaler = pickle.load(open(os.path.join(__location__,'scaler.pkl'), 'rb'))


def find_matches(_df):
    """
    :param _df: data frame extract from AKELA to be matched.  Must include the following columns:
        PKPersonID (as int index),
        SKPersonID: int,
        FirstName: str,
        MiddleName: str,
        LastName: str,
        Gender: str,
        DateOfBirth: pandas datetime obj,
        AddressLine1: str,
        AddressLine2: str,
        City: str,
        State: str,
        Zip5: str
    :return: list of tuples of matching pairs of PKPersonID's
    """
    _df['Male'] = _df.Gender.apply(lambda x: True if x == 'M' else (False if x == 'F' else None))
    _df = _df.drop(labels=['Gender'], axis=1).dropna(subset=['Zip5', 'DateOfBirth', 'Male']).drop_duplicates()
    _df = _df[['SKPersonID', 'FirstName', 'MiddleName', 'LastName', 'DateOfBirth', 'AddressLine1', 'AddressLine2',
               'City', 'State', 'Zip5', 'Male']]

    clusters = _df.groupby(['Zip5', 'Male', 'DateOfBirth'])[['SKPersonID']].count().sort_values(by='SKPersonID',
                                                                                                ascending=True)
    clusters = clusters[clusters.SKPersonID > 1]

    x_rows = []
    id_rows = []

    cluster_list = [cluster for cluster in clusters.iterrows()]

    for cluster in tqdm(cluster_list):
        zip5, male, dateofbirth = cluster[0]

        records = list(_df[np.logical_and(_df.Zip5 == zip5, np.logical_and(_df.Male == male,
                        _df.DateOfBirth == dateofbirth))].reset_index().drop_duplicates().as_matrix())
        if len(records) > 1:  # Make sure the same number of rows were retrieved are, at least, greater than 1
            while records:  # iterate through the list the first time
                current_record = records.pop()  # remove last pid and compare it to the others
                if len(
                        records) >= 1:  # make sure I'm not just down to the last record
                    for comparison_record in records:  # iterate through the remaining records

                        id_rows.append((current_record[0], comparison_record[0]))

                        test_data = []  # Build single row of comparison data to feed to the model
                        test_data.append(fuzz.ratio(current_record[4], comparison_record[4]) / 100)
                        test_data.append(len(current_record[4]) - len(comparison_record[4]))
                        test_data.append(fuzz.ratio(current_record[2], comparison_record[2]) / 100)
                        test_data.append(len(current_record[2]) - len(comparison_record[2]))
                        test_data.append(fuzz.ratio(current_record[2:5], comparison_record[2:5]) / 100)
                        test_data.append(fuzz.token_set_ratio(current_record[2:5], comparison_record[2:5]) / 100)
                        test_data.append(fuzz.token_sort_ratio(current_record[2:5], comparison_record[2:5]) / 100)
                        test_data.append(fuzz.token_set_ratio(current_record[6:10], comparison_record[6:10]) / 100)
                        test_data.append(np.absolute((current_record[0] - comparison_record[0])))
                        test_data.append(int(current_record[-1]))
                        test_data.append(int(pd.isnull(current_record[3]) == pd.isnull(comparison_record[3])))

                        x_rows.append(test_data)
    x_rows = scaler.transform(x_rows)
    y_probas = nnet_model.predict(x_rows)
    y_bools = [1 if x >= 0.5 else 0 for x in y_probas]

    return [x[0] for x in zip(id_rows, y_bools) if x[1] == 1]


def dedupe_cluster_table(df, *args, **kwargs):
    proc_name = mp.current_process().name
    print('Deduplcating Cluster for process {}'.format(proc_name))
    nnet_copy = load_model(os.path.join(__location__, 'keras_person_model.h5'))
    scaler_copy = pickle.load(open(os.path.join(__location__, 'scaler.pkl'), 'rb'))
    x_rows = []
    id_rows = []
    records = list(df.reset_index().drop_duplicates().as_matrix())
    if len(records) > 1:
        while records:
            current_record = records.pop()
            if len(records) >= 1:
                for comparison_record in records:
                    id_rows.append((current_record[0], comparison_record[0]))
                    test_data = []
                    test_data.append(fuzz.ratio(current_record[4], comparison_record[4]) / 100)
                    test_data.append(len(current_record[4]) - len(comparison_record[4]))
                    test_data.append(fuzz.ratio(current_record[2], comparison_record[2]) / 100)
                    test_data.append(len(current_record[2]) - len(comparison_record[2]))
                    test_data.append(fuzz.ratio(current_record[2:5], comparison_record[2:5]) / 100)
                    test_data.append(fuzz.token_set_ratio(current_record[2:5], comparison_record[2:5]) / 100)
                    test_data.append(fuzz.token_sort_ratio(current_record[2:5], comparison_record[2:5]) / 100)
                    test_data.append(fuzz.token_set_ratio(current_record[6:10], comparison_record[6:10]) / 100)
                    test_data.append(np.absolute((current_record[0] - comparison_record[0])))
                    test_data.append(int(current_record[-1]))
                    test_data.append(int(pd.isnull(current_record[3]) == pd.isnull(comparison_record[3])))
                    x_rows.append(test_data)

    if len(x_rows) > 0:
        x_rows = scaler_copy.transform(x_rows)
        y_probas = nnet_copy.predict(x_rows)
        y_bools = [1 if x >= 0.5 else 0 for x in y_probas]

        return [x[0] for x in zip(id_rows, y_bools) if x[1] == 1]
    else:
        return None


def dedupe_cluster_list(cluster_list, return_dict=None, *args, **kwargs):
    if return_dict is not None:
        print('Deduplicating Cluster List with Return Dict')
        for idx, cluster in enumerate(cluster_list):
            return_dict[idx] = dedupe_cluster_table(cluster)
    else:
        print('Deduplicating Cluster List without Return Dict')
        deduped_clusters = []
        for cluster in cluster_list:
            deduped_clusters.append(dedupe_cluster_table(cluster))
        return deduped_clusters


def make_cluster_dfs(_df):
    # Processing of clusters, which is what can then be divided up and passed to the multiple processes
    _df['Male'] = _df.Gender.apply(lambda x: True if x == 'M' else (False if x == 'F' else None))
    _df = _df.drop(labels=['Gender'], axis=1).dropna(subset=['Zip5', 'DateOfBirth', 'Male']).drop_duplicates()
    _df = _df[['SKPersonID', 'FirstName', 'MiddleName', 'LastName', 'DateOfBirth', 'AddressLine1', 'AddressLine2',
               'City', 'State', 'Zip5', 'Male']]
    clusters = _df.groupby(['Zip5', 'Male', 'DateOfBirth'])[['SKPersonID']].count()
    clusters = clusters[clusters.SKPersonID > 1]
    cluster_frames = []
    for cluster in clusters.iterrows():
        zip5, male, dateofbirth = cluster[0]
        cluster_frames.append(_df[np.logical_and(_df.Zip5 == zip5,
                                                 np.logical_and(_df.Male == male,
                                                                _df.DateOfBirth == dateofbirth))])
    return cluster_frames


def multiprocess_deduplicate(df, n_jobs=4):
    clusters = make_cluster_dfs(df)
    if len(clusters) > 0:
        print('Clusters Made')
    return multiprocess_function(data_in=clusters, func=dedupe_cluster_list, n_jobs=n_jobs, shared_resources=None)
