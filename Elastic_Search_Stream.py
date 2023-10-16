# importing all the necessary packages.
import os
import time
from sklearn.model_selection import train_test_split
from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_io as tfio
# Once the instance has been started, grep for elasticsearch in the processes list to confirm the availability.
import subprocess

ES_NODES = "http://localhost:9200"


def read_data():
    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
    tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                            extract=True, cache_dir='.')
    pf_df = pd.read_csv(csv_file)
    # print(pf_df.head(5))
    # print(pf_df['AdoptionSpeed'])
    # print(pf_df['AdoptionSpeed'].value_counts())
    return pf_df


def preprocess_data(pf_df):
    # In the dataset "4" indicates the pet was not adopted.
    pf_df['target'] = np.where(pf_df['AdoptionSpeed'] == 4, 0, 1)

    # Drop un-used columns.
    pf_df = pf_df.drop(columns=['AdoptionSpeed', 'Description'])

    # print Number of datapoints and columns
    # print(pf_df.shape)

    # Split the dataset into train, validation, and test datasets.
    train_df, test_df = train_test_split(pf_df, test_size=0.3, shuffle=True)
    print("Number of training samples: ", len(train_df))
    print("Number of testing sample: ", len(test_df))

    return train_df, test_df


def prepare_elsatic_search_data(index, doc_type, df):
    """
    This function is designed to prepare data from a DataFrame (typically containing structured data) for
    indexing into an Elasticsearch cluster. It formats the data in a way that is compatible with Elasticsearch's bulk
    indexing API :
    param index:
    param doc_type: :
    param df: :return:
    """
    # Storing the data in the local elasticsearch cluster simulates an environment for continuous remote data
    # retrieval for training and inference purposes
    records = df.to_dict(orient='records')
    es_data = []
    for idx, record in enumerate(records):
        meta = {
            "index": {
                "_index": index,
                "_type": doc_type,
                "_id": idx
            }
        }
        es_data.append(meta)
        es_data.append(record)
    return es_data


def index_elastic_search_data(index, es_data):
    """
    This function is responsible for indexing data into an Elasticsearch cluster using the bulk indexing API.
    :param index:
    :param es_data:
    :return:
    """
    # Step 1: Create an Elasticsearch client using the provided host(s)
    es_client = Elasticsearch(hosts= ES_NODES)

    # Step 2: Check if the Elasticsearch index already exists. If it does, delete it.
    if es_client.indices.exists(index=index):
        print("deleting the '{}' index.".format(index))
        res = es_client.indices.delete(index=index)
        print("Response from server: {}".format(res))

    # Step 3: Create the Elasticsearch index
    print("creating the '{}' index.".format(index))
    res = es_client.indices.create(index=index)
    print("Response from server: {}".format(res))

    # Step 4: Bulk index the data into the Elasticsearch index
    print("bulk index the data")
    res = es_client.bulk(index=index, body=es_data, refresh=True)
    print("Errors: {}, Num of records indexed: {}".format(res["errors"], len(res["items"])))



def create_train_test_index(train_df, test_df):
    """
    This function is responsible for creating the Elasticsearch index for the training and testing datasets.
    :return:
    """
    # Step 1: Prepare Elasticsearch data for the training and testing datasets
    train_es_data = prepare_elsatic_search_data(index="train", doc_type="pet", df=train_df)
    test_es_data = prepare_elsatic_search_data(index="test", doc_type="pet", df=test_df)

    # Step 2: Index the training data into the "train" index
    index_elastic_search_data(index="train", es_data=train_es_data)

    # Sleep for a few seconds to allow the "train" index to complete indexing (optional but may be useful).
    time.sleep(3)

    # Step 3: Index the testing data into the "test" index
    index_elastic_search_data(index="test", es_data=test_es_data)


def confirm_availability():
    # Run the 'ps' command to list processes and filter using 'grep'
    # Define the Elasticsearch startup command

    # Specify the path to the Elasticsearch startup script
    elasticsearch_startup_script = "/Users/srishma/Desktop/Home/Projects/Elastic_Stream/elasticsearch-7.9.2/bin/elasticsearch"


    # Use subprocess to start Elasticsearch as a daemon process
    subprocess.Popen(elasticsearch_startup_script, shell=True)


def train_test_dataset():
    BATCH_SIZE = 32
    HEADERS = {"Content-Type": "application/json"}

    train_ds = tfio.experimental.elasticsearch.ElasticsearchIODataset(
        nodes=[ES_NODES],
        index="train",
        doc_type="pet",
        headers=HEADERS
    )

    # Prepare a tuple of (features, label)
    train_ds = train_ds.map(lambda v: (v, v.pop("target")))
    train_ds = train_ds.batch(BATCH_SIZE)

    test_ds = tfio.experimental.elasticsearch.ElasticsearchIODataset(
        nodes=[ES_NODES],
        index="test",
        doc_type="pet",
        headers=HEADERS
    )

    # Prepare a tuple of (features, label)
    test_ds = test_ds.map(lambda v: (v, v.pop("target")))
    test_ds = test_ds.batch(BATCH_SIZE)

    return train_ds, test_ds



#  keras preprocessing layers

def get_normalization_layer(name, dataset):
    """
    This function is responsible for creating a normalization layer for a specific feature in the dataset.
    :param name:
    :param dataset:
    :return:
    """
    # Create a Normalization layer for our feature.
    normalizer = preprocessing.Normalization()

    # Prepare a Dataset that only yields our feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer



def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a StringLookup layer which will turn strings into integer indices
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_values=max_tokens)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Create a Discretization for our integer indices.
    encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

    # Prepare a Dataset that only yields our feature.
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices.
    encoder.adapt(feature_ds)

  # Apply one-hot encoding to our indices. The lambda function captures the
  # layer so you can use them, or include them in the functional model later.
    return lambda feature: encoder(index(feature))


# Check and observe features of a sample record from the training of model.
def check_features(train_ds):
    ds_iter = iter(train_ds)
    features, label = next(ds_iter)
    var = {key: value.numpy()[0] for key, value in features.items()}
    print(var)


def choose_features(train_ds):
    """
    this function selects and processes specific features from a dataset for use in a machine learning model.
    :param train_ds:
    :return:
    """
    all_inputs = []
    encoded_features = []

    # Numeric features.
    for header in ['PhotoAmt', 'Fee']:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    # Categorical features encoded as string.
    categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                        'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
    for header in categorical_cols:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(header, train_ds, dtype='string',
                                                     max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)


def build_train_compile_model(encoded_features, all_inputs, train_ds, test_ds):
    """
    This function is responsible for building and compiling a machine learning model.
    :param encoded_features:
    :return:
    """
    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(all_inputs, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    model.fit(train_ds, epochs=10, validation_data=test_ds)
    print("test loss, test accuracy: ", model.evaluate(test_ds))

    return model


def main():
    train_ds, test_ds = train_test_dataset()
    check_features(train_ds)
    encoded_features, all_inputs = choose_features(train_ds)
    model = build_train_compile_model(encoded_features, all_inputs, train_ds, test_ds)
    print("Model trained successfully")


if __name__ == '__main__':
    main()