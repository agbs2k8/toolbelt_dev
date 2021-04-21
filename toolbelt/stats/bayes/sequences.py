import numpy as np


class DataSource:
    """
    A class for log-ingestion data sources that handles bayesian probabilities for the data source
    log ingestion rates falling to zero for any given number of time period

    Can be initialized with a history, from which we make assumptions to create a true positive rate (TPR)
    and a false positive rate (FPR)

    Can be initialized without a history.  For those instances, we initialize the TPR/FPR based on a pre-assigned
    value for high/low log volumes.
    """
    # TODO: copy from working directory '/Users/ajwilson/OneDrive - Cysiv/ds_analytics_engine_working/ML/Volume_Forecasts/Connector_Stats/june2020'

