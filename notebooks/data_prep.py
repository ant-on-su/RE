#inspired by github repo SeoSangwoo/Attention-Based-BiLSTM-relation-extraction

import numpy as np
import pandas as pd
import re

#labels encoding:

class2label = {'Other': 0,
               'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
               'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
               'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
               'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
               'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
               'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
               'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
               'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
               'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}

label2class = {0: 'Other',
               1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
               3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
               5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
               7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
               9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
               11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
               13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
               15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
               17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}

def clean_str(text):
    text = text.casefold()
    # Clean the text
    good_symbols = re.compile('[^0-9a-z #+:]')
    bad_symbols = re.compile('[/(){}\[\]\|@,;\.]')
    
    text = good_symbols.sub('', text)
    text = bad_symbols.sub(' ', text)
    
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text) 
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\-", " - ", text)

    return text.strip()

def load_data_and_labels(path):
    data = []
    lines = [line.strip() for line in open(path)]

    for idx in range(0, len(lines), 4):
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.replace('<e1>', ' _e11_ ')
        sentence = sentence.replace('</e1>', ' _e12_ ')
        sentence = sentence.replace('<e2>', ' _e21_ ')
        sentence = sentence.replace('</e2>', ' _e22_ ')

        sentence = clean_str(sentence)
        tokens = sentence.split()
        sentence = " ".join(tokens)

        data.append([id, sentence, relation])
        
    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    df['label'] = [class2label[r] for r in df['relation']]

    x_text = df['sentence'].tolist()

    y = df['label']
    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)
        
    return x_text, labels

def load_test_data(path):
    data = []
    lines = [line.strip() for line in open(path)]

    for idx in range(0, len(lines)):
        id = lines[idx].split("\t")[0]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.replace('<e1>', ' _e11_ ')
        sentence = sentence.replace('</e1>', ' _e12_ ')
        sentence = sentence.replace('<e2>', ' _e21_ ')
        sentence = sentence.replace('</e2>', ' _e22_ ')

        sentence = clean_str(sentence)
        tokens = sentence.split()
        sentence = " ".join(tokens)

        data.append([id, sentence])
        
    df = pd.DataFrame(data=data, columns=["id", "sentence"])

    x_text = df['sentence'].tolist()
    first_id = int(df.id[0])
    
    print(len(x_text), first_id)
    
    return x_text, first_id

def load_pretrained_emb(embeddings, embeddings_dim, idx_word_dict):
    #initialise ramdom matrix
    initial_embedding_matrix = np.random.randn(len(idx_word_dict), 
                                               embeddings_dim).astype(np.float32) / np.sqrt(len(idx_word_dict))
    #load embeddings from embeddings.vocab
    for idx in range(0,len(idx_word_dict)-1):
        if idx_word_dict[idx] in embeddings.vocab:
            initial_embedding_matrix[idx] = embeddings[idx_word_dict[idx]]
    return initial_embedding_matrix