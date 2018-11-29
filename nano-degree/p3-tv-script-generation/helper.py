import os
import pickle

import csv

def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r",encoding='utf8') as f:
        data = f.read()

    return data

def load_larger_data(path):
    """
    Created by Sai:
    Load Dataset from File
    """
    csv_file = os.path.join(path)
    data = None
    with open(csv_file, "r",encoding='utf8') as f:
        reader = csv.DictReader(f, quotechar='"')

        tmp_episode_id = None
        return_str = ""
        for row in reader:


            episode_id = int(row['episode_id'])
            if tmp_episode_id is not None and tmp_episode_id != episode_id:
                return_str += "\n\n"

            tmp_episode_id = episode_id
            return_str += row['raw_text'] + '\n'

            #try:
            #    cursor.execute(insertQuery,data)
            #except:
            #    print(">>>>>problem with row #",row_id)
            #    break
    return return_str


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)

    # Ignore notice, since we don't use it for analysing the data
    text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))
