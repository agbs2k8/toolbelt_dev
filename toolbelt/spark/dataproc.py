# Python Standard Library
import datetime
import json
from collections import Counter
from collections.abc import Iterable
import re

# Oher common python packages
import shlex
import numpy as np
import pandas as pd

# Google Cloud Utils
from google.cloud import storage
from google.cloud.storage import Blob

# Spark basics
from pyspark.sql.functions import udf
import pyspark.sql.functions as f
from pyspark.sql.types import StructField, StructType, Row, ArrayType, MapType
from pyspark.sql.types import IntegerType, LongType, DoubleType, DecimalType
from pyspark.sql.types import TimestampType, NumericType, IntegralType, FractionalType
from pyspark.sql.types import StringType, BinaryType, DateType, NullType, BooleanType


def filter_blob(blob, customer_id=None, start_time=None, end_time=None):
    time_format = '%Y%m%dT%H%M%S.%fZ'
    parsed_customer_id = blob.name.split('/')[0]
    try:
        parsed_timestamp = datetime.datetime.strptime(blob.name.split('-')[1], time_format)
    except:
        return False
    if customer_id != None:
        if isinstance(customer_id, str):
            if customer_id != parsed_customer_id:
                return False
        elif isinstance(customer_id, list):
            if parsed_customer_id not in customer_id:
                return False
    if start_time != None:
        if start_time > parsed_timestamp:
            return False
    if end_time != None:
        if end_time < parsed_timestamp:
            return False
    return True


def filter_blobs_to_files(blob_lists, bucket_name, customer_id=None, start_time=None, end_time=None):
    files = []
    for blob in blob_lists:
        if filter_blob(blob, customer_id=customer_id, start_time=start_time, end_time=end_time) == True:
            files.append('gs://' + bucket_name + '/' + blob.name)
    return files


def iter_check(x):
    return not isinstance(x, str) and isinstance(x, Iterable)


def flatten(xs):
    result = []
    for x in xs:
        if iter_check(x):
            result += flatten(x)
        else:
            result.append(x)
    return result


def flatten_schema(field, prefix=None):
    if isinstance(field, dict) and "fields" in field.keys():
        if prefix:
            return [y for y in [flatten_schema(field['fields'][x], prefix) for x in range(len(field['fields']))] if
                    y is not None]
        else:
            return [y for y in [flatten_schema(field['fields'][x]) for x in range(len(field['fields']))] if
                    y is not None]
    elif isinstance(field, dict) and "type" in field.keys():
        if field['type'] == 'string':
            if prefix:
                return prefix + '.' + field['name']
            else:
                return field['name']
        elif field['type'] == 'array':
            return prefix
        elif field['type'] is not None:
            if prefix:
                return flatten_schema(field['type'], prefix + '.' + field['name'])
            else:
                return flatten_schema(field['type'], field['name'])


def flatten_df(_df):
    column_names = flatten(flatten_schema(json.loads(_df.schema.json())))
    lowercase_used_names = set()
    final_selects = []
    merges = []
    for col in column_names:
        _rename = col.replace('.', '_')
        if _rename.lower() in lowercase_used_names:
            # find the duplicate I already have:
            already_used = [x for x in final_selects if x[1].lower() == _rename.lower()][0]
            merges.append((already_used[1], _rename + '_x'))
            final_selects.append((col, _rename + '_x'))
        else:
            lowercase_used_names.add(_rename.lower())
            final_selects.append((col, _rename))
    result_df = _df.select([f.col(x[0]).alias(x[1]) for x in final_selects])

    if len(merges) > 0:
        for merge_tuple in merges:
            result_df = result_df.withColumn(merge_tuple[0], f.coalesce(f.col(merge_tuple[0]), f.col(merge_tuple[1])))
            result_df.drop(f.col(merge_tuple[1]))
    return result_df


def count_null_nan(c, dtype):
    if dtype in ['double', 'float']:
        pred = f.col(c).isNull() | f.isnan(c)
        return f.sum(pred.cast("integer")).alias(c)
    else:
        pred = f.col(c).isNull()
        return f.sum(pred.cast("integer")).alias(c)


def flatten_and_select_non_null_cols(_df):
    flat_df = flatten_df(_df)
    null_counts = flat_df.agg(f.count(f.col(_df.columns[0])), *[count_null_nan(c[0], c[1]) for c in flat_df.dtypes]).rdd.collect()[
        0].asDict()
    count_key = [x for x in null_counts.keys() if str.startswith(x, 'count(')][0]
    count = null_counts[count_key]
    good_columns = [key for key, val in null_counts.items() if key != count_key and val != count]
    return flat_df.select([f.col(x) for x in good_columns])
