#coding=utf-8

try:
    import os
except:
    pass

try:
    import imp
except:
    pass

try:
    import operator
except:
    pass

try:
    import math
except:
    pass

try:
    import glob
except:
    pass

try:
    import json
except:
    pass

try:
    import time
except:
    pass

try:
    from IPython.display import IFrame
except:
    pass

try:
    import nltk
except:
    pass

try:
    from nltk.tokenize import word_tokenize
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import pandas as pd
except:
    pass

try:
    import seaborn as sn
except:
    pass

try:
    import matplotlib.pyplot as plt
except:
    pass

try:
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
except:
    pass

try:
    import tensorflow
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK
except:
    pass

try:
    import hyperopt
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform, conditional
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers import Input, Dense, Activation, Dropout
except:
    pass

try:
    from keras.layers import LSTM, TimeDistributed
except:
    pass

try:
    from keras.optimizers import Adam
except:
    pass

try:
    from keras.callbacks import EarlyStopping, TensorBoard
except:
    pass

try:
    from keras import metrics
except:
    pass

try:
    from keras.utils.np_utils import to_categorical
except:
    pass

try:
    import gensim
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional
def get_data():
    
    files = []
    data = []
    header = ['Annotation ID', 'Batch ID', 'Annotator ID', 'Policy ID', 'Segment ID', 'Category Name', 'Attributes/Values', 'Policy URL', 'Date']
    keep_columns = ['Segment ID', 'Category Name', 'Attributes/Values']
    for file in glob.glob("data\\annotations/*.csv"):
        files.append(file[17:-4])
        data.append(pd.read_csv(file, names=header)[keep_columns])
        
    policies = []
    for file in files:
        with open("data\\sanitized_policies/{}.html".format(file)) as f:
            policies.append(f.readlines()[0].split('|||'))

    # categories = set()
    # for datum in data:
    #     cat = datum['Category Name']
    #     categories.update(cat)
    # categories

    categories = [
     'Data Retention', # 0
     'Data Security', # 1
     'Do Not Track', # 2
     'First Party Collection/Use', # 3
     'International and Specific Audiences', # 4
     'Other', # 5
     'Policy Change', # 6
     'Third Party Sharing/Collection', # 7
     'User Access, Edit and Deletion', # 8
     'User Choice/Control', # 9
     'None' # 10
    ]

    one_hot_categories = np.identity(len(categories))

    cat_dict = {
     categories[0]:  one_hot_categories[0],
     categories[1]:  one_hot_categories[1],
     categories[2]:  one_hot_categories[2],
     categories[3]:  one_hot_categories[3],
     categories[4]:  one_hot_categories[4],
     categories[5]:  one_hot_categories[5],
     categories[6]:  one_hot_categories[6],
     categories[7]:  one_hot_categories[7],
     categories[8]:  one_hot_categories[8],
     categories[9]:  one_hot_categories[9],
     categories[10]: one_hot_categories[10],
    }

    # attribute_value_types = set()
    # attribute_value_values = set()
    # for datum in data:
    #     avs = datum['Attributes/Values']
    #     for row in avs:
    #         parsed = json.loads(row)
    #         keys = list(parsed.keys())
    #         attribute_value_types.update(keys)
    #         for key in keys:
    #             attribute_value_values.add(parsed[key]['value'])

    attribute_value_types = ['Access Scope',
     'Access Type',
     'Action First-Party',
     'Action Third Party',
     'Audience Type',
     'Change Type',
     'Choice Scope',
     'Choice Type',
     'Collection Mode',
     'Do Not Track policy',
     'Does/Does Not',
     'Identifiability',
     'Notification Type',
     'Other Type',
     'Personal Information Type',
     'Purpose',
     'Retention Period',
     'Retention Purpose',
     'Security Measure',
     'Third Party Entity',
     'User Choice',
     'User Type']

    attribute_value_values = ['Additional service/feature',
     'Advertising',
     'Aggregated or anonymized',
     'Analytics/Research',
     'Basic service/feature',
     'Both',
     'Browser/device privacy controls',
     'Californians',
     'Children',
     'Citizens from other countries',
     'Collect from user on other websites',
     'Collect in mobile app',
     'Collect on first party website/app',
     'Collect on mobile website',
     'Collect on website',
     'Collection',
     'Computer information',
     'Contact',
     'Cookies and tracking elements',
     'Data access limitation',
     'Deactivate account',
     'Delete account (full)',
     'Delete account (partial)',
     'Demographic',
     'Does',
     'Does Not',
     'Dont use service/feature',
     'Edit information',
     'Europeans',
     'Explicit',
     'Export',
     'Financial',
     'First party collection',
     'First party use',
     'First-party privacy controls',
     'General notice in privacy policy',
     'General notice on website',
     'Generic',
     'Generic personal information',
     'Health',
     'Honored',
     'IP address and device IDs',
     'Identifiable',
     'Implicit',
     'In case of merger or acquisition',
     'Indefinitely',
     'Introductory/Generic',
     'Legal requirement',
     'Limited',
     'Location',
     'Marketing',
     'Mentioned, but unclear if honored',
     'Merger/Acquisition',
     'Named third party',
     'No notification',
     'Non-privacy relevant change',
     'None',
     'Not honored',
     'Not mentioned',
     'Opt-in',
     'Opt-out',
     'Opt-out link',
     'Opt-out via contacting company',
     'Other',
     'Other data about user',
     'Other part of company/affiliate',
     'Other users',
     'Perform service',
     'Personal identifier',
     'Personal notice',
     'Personalization/Customization',
     'Practice not covered',
     'Privacy contact information',
     'Privacy relevant change',
     'Privacy review/audit',
     'Privacy training',
     'Privacy/Security program',
     'Profile data',
     'Public',
     'Receive from other parts of company/affiliates',
     'Receive from other service/third-party (named)',
     'Receive from other service/third-party (unnamed)',
     'Receive/Shared with',
     'Secure data storage',
     'Secure data transfer',
     'Secure user authentication',
     'See',
     'Service Operation and Security',
     'Service operation and security',
     'Social media data',
     'Stated Period',
     'Survey data',
     'Third party sharing/collection',
     'Third party use',
     'Third-party privacy controls',
     'Track on first party website/app',
     'Track user on other websites',
     'Transactional data',
     'Unnamed third party',
     'Unspecified',
     'Use',
     'User Profile',
     'User account data',
     'User online activities',
     'User participation',
     'User profile',
     'User with account',
     'User without account',
     'View',
     'not-selected']

    model = gensim.models.Doc2Vec(vector_size=100)

    stemmer = nltk.stem.porter.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    chosen_categories = ['First Party Collection/Use', 
                         'Third Party Sharing/Collection', 
                         'Other', 
                         'User Choice/Control', 
                         'Data Security',
                         'International and Specific Audiences',
                         'User Access, Edit and Deletion',
                         'Policy Change',
                         'Data Retention',
                         'Do Not Track',
                         'None' # added by us, not in original corpus
                        ]
    remove_text = ['null', 'Not selected']

    df_columns = ['text', 'category', 'category one hot', 'text vec']
    df = pd.DataFrame([], columns=df_columns)
    series = []
    documents = []
    cats = []

    remove_spans = {} # dictionary of policy ids and list of start, stop tuples that are then removed
    # remove_spans structure:

    '''
    {
    "2": --> this is the policy id
      {
       "6": [(20, 30), (30, 50)], --> this is the segment id
       "8": [(40, 123)] --> which maps to a list of tuple of start, end indices
      }
    }
    '''


    idx = 0
    for datum_idx in range(len(data)):
        datum = data[datum_idx]
        for idx in range(len(datum)):        
            category = datum['Category Name'][idx]

            if chosen_categories is None:
                continue

            if category not in chosen_categories:
                continue

            segment_id = datum['Segment ID'][idx]
            if datum_idx not in remove_spans:
                remove_spans[datum_idx] = {}
            if segment_id not in remove_spans[datum_idx]:
                remove_spans[datum_idx][segment_id] = []

            # ok, we have our policy text, now we need to 
            # remove all of the spans that are associated with a category
            # so we can attribute that text to the 'None' category

            parsed = json.loads(datum['Attributes/Values'][idx])
            for value in attribute_value_types:
                if value in parsed.keys():
                    attributes = parsed[value]
                    has_selected_text = 'selectedText' in attributes
                    has_start_idx = 'startIndexInSegment' in attributes
                    has_end_idx = 'endIndexInSegment' in attributes
                    if has_selected_text and has_start_idx and has_end_idx:
                        text = attributes['selectedText']
                        start_idx = attributes['startIndexInSegment']
                        end_idx = attributes['endIndexInSegment']

                        if text in remove_text or start_idx == -1 or end_idx == -1:
                            continue

                        remove_spans[datum_idx][segment_id].append((start_idx, end_idx))

                        text = text.lower()
                        processed_text = word_tokenize(text)
                        #processed_text = [stemmer.stem(word) for word in processed_text]
                        processed_text = [lemmatizer.lemmatize(word) for word in processed_text]

                        doc = gensim.models.doc2vec.TaggedDocument(processed_text, [idx])
                        documents.append(doc)
                        cats.append(cat_dict[category])
                        text = ' '.join(processed_text)
                        series.append(pd.Series([text, category, cat_dict[category], None], index=df_columns))

                        idx += 1

    SHOULD_PROCESS_NONE_CATEGORY = True

    replace_items = ["<br>", "<strong>", "</strong>", "<ul>", "</ul>", "<li>", "</li>", "<ol>", "</ol>"]
    category = 'None'
    none_count = 0

    if SHOULD_PROCESS_NONE_CATEGORY:
        for policy_idx in remove_spans:
            policy = policies[policy_idx]
            for segment_idx in remove_spans[policy_idx]:
                try:
                    policy_segment = policy[segment_idx]
                except IndexError as e:
                    #print(e, policy_idx, segment_idx)
                    continue
                segment_text = policy_segment
                for span in remove_spans[policy_idx][segment_idx]:
                    start_idx = span[0]
                    end_idx = span[1]
                    segment_text = segment_text[:start_idx] + " " + segment_text[end_idx:]
                segment_text = segment_text.lower()
                for item in replace_items:
                    segment_text = segment_text.replace(item, " ")
                segment_text = segment_text.strip()
                if segment_text: # check if we have any characters at all
                    processed_text = word_tokenize(segment_text)
                    processed_text = [lemmatizer.lemmatize(word) for word in processed_text]

                    doc = gensim.models.doc2vec.TaggedDocument(processed_text, [idx])
                    documents.append(doc)
                    cats.append(cat_dict[category])
                    text = ' '.join(processed_text)
                    series.append(pd.Series([text, category, cat_dict[category], None], index=df_columns))
                    none_count += 1
                    idx += 1

        #print('None count: {}'.format(none_count))

        cats = np.array(cats)

        df = df.append(series, ignore_index=True)
        #print(df.shape)

        model.build_vocab(documents)
        model.train(documents, total_examples=len(documents), epochs=16)

        vecs = []
        for row in df.itertuples():
            category_not_chosen = chosen_categories is None
            category_chosen_and_matches = chosen_categories is not None and row.category in chosen_categories
            if category_chosen_and_matches or category_not_chosen:
                model.random = np.random.RandomState(1234)
                vecs.append(np.array(model.infer_vector(word_tokenize(row.text))))

        vecs = np.array(vecs)
        #print(vecs.shape)

        return vecs, cats



vecs, cats = get_data()

choice = np.random.choice(len(vecs), len(vecs), replace=False)
test_percentage = 0.25 # keep 25% of data for testing
test_amount = math.floor(0.25 * len(vecs))
train_indices = choice[test_amount:]
test_indices = choice[:test_amount]

# vecs, cats
x_train = vecs[train_indices]
x_test = vecs[test_indices]
y_train = cats[train_indices]
y_test = cats[test_indices]


def keras_fmin_fnct(space):


    nn_model = Sequential()
    nn_model.add(Dense( space['Dense'], batch_input_shape=(None, 100, )))
    nn_model.add(Activation( space['Activation'] ))
    
    if conditional( space['conditional'] ) == 'dropout':
        nn_model.add(Dropout( space['Dropout'] ))
    
    nn_model.add(Dense(11))
    nn_model.add(Activation('softmax'))
    
    nn_model.compile(loss='categorical_crossentropy', optimizer=space['optimizer'], metrics=[metrics.categorical_accuracy])

    #print(nn_model.summary())

    tensorboard_callback = TensorBoard(log_dir='C:/tmp/pp_run-'+time.strftime("%Y-%m-%d-%H%M%S"))
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=4, verbose=1, mode='auto')
    history = nn_model.fit(x_train, y_train, batch_size=128, epochs=16, verbose=2, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
    acc = history.history['val_categorical_accuracy'][-1]
    print('Test accuracy:', acc)
    
    return {'loss': -acc, 'status': STATUS_OK, 'model': nn_model}

def get_space():
    return {
        'Dense': hp.choice('Dense', [32, 64, 128, 256, 512, 1024]),
        'Activation': hp.choice('Activation', ['relu', 'tanh', 'sigmoid']),
        'conditional': hp.choice('conditional', ['dropout', 'no dropout']),
        'Dropout': hp.uniform('Dropout', 0, 1),
        'optimizer': hp.choice('optimizer', ['sgd', 'rmsprop', 'adagrad', 'adam', 'nadam']),
    }
