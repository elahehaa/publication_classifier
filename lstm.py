import numpy as np
import sklearn
import tensorflow as tf
import fasttext
from numpy import array
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tensorflow.python.keras.layers import  Embedding, LSTM, Dense,  Dropout, Conv1D
from tensorflow.python.keras.callbacks import EarlyStopping
import utils
import wandb
from sklearn.utils import resample





def custom_standardization(txt):
    txt = tf.strings.regex_replace(txt, '/', ' / ')
    txt = tf.strings.regex_replace(txt, '.-', ' .- ')
    txt = tf.strings.regex_replace(txt, '\'', ' .\' ')
    txt = tf.strings.regex_replace(txt, '[^\w\s]', '')  # remove punctuations
    txt = tf.strings.regex_replace(txt, '\d+', '')  # remove numbers
    txt = tf.strings.lower(txt)  # lower case

    for each in stop_words.values:
        txt = tf.strings.regex_replace(txt, ' ' + each[0] + ' ', r" ")
    txt = tf.strings.regex_replace(txt, " +", " ")

    return txt

#create vocabulary
def vocab_layer(custom_std, max_seq_length):
    vec_lyr = tf.keras.layers.TextVectorization(
            max_tokens = None,
            standardize = custom_std,
            split = 'whitespace',
            ngrams = None,
            output_mode = "int",
            output_sequence_length = max_seq_length #each sequence size
        )
    #layer = vectorize_layer.adapt(x_input)
    return vec_lyr


def create_embedding_dim(w2v_model):
    word = list(vectorize_layer.get_vocabulary())[2]
    embedding_size = w2v_model.get_word_vector(word).shape[0]
    return embedding_size


def embedding_matrix(vocab_d_size, embedding_size, vec_lyr):
    embedding_mat = np.zeros((vocab_d_size + 1, embedding_size))
    for i, word in enumerate(vec_lyr.get_vocabulary()):
        embedding_vector = biow2v_model.get_word_vector(word)
        if embedding_vector is not None:
            embedding_mat[i] = embedding_vector
    return embedding_mat

def embedding_matrix_clstm(vocab_d_size, embedding_size, vec_lyr):
    embedding_mat = np.random.uniform(-0.25, 0.25, size = (vocab_d_size + 1, embedding_size))
    count = 0
    for i, word in enumerate(vec_lyr.get_vocabulary()):
        embedding_vector = biow2v_model.get_word_vector(word)
        if embedding_vector is not None:
            embedding_mat[i] = embedding_vector
        else:
            count += 1
    print("number of words not in the vocab: ", count)
    return embedding_mat


def create_embedding_layer(vocab_d_size, embedding_size, embedding_mat, max_lng):
    embedding_lyr = Embedding(input_dim=vocab_d_size + 1,
                               output_dim=embedding_size,
                               weights=[embedding_mat],
                               input_length=max_lng,
                               trainable=False)
    return embedding_lyr


# model1 using sequential class experiments
def model1(embedding_size, max_len):
    seq_model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        Embedding(input_dim=vocab_size + 1,
                  output_dim=embedding_size,
                  weights=[embedding_matrix],
                  input_length=max_len,
                  trainable=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')
    ])
    return seq_model


def model2(vec_lyr, embedding_lyr):
    seq_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.string),
        vec_lyr,
        embedding_lyr,
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(4, activation='softmax'),

    ])
    return seq_model


def model3(vec_lyr_train,  embedding_lyr):
    input_layer = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    vec_lyr = vec_lyr_train
    vec_lyr = vec_lyr(input_layer)
    emb_lyr = embedding_lyr(vec_lyr)
    bidir_lyr = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(emb_lyr)
    dens_lyr = tf.keras.layers.Dense(64, activation='relu')(bidir_lyr)
    dens_lyr = tf.keras.layers.Dense(4, activation='softmax')(dens_lyr)
    output = tf.keras.Model(input_layer, dens_lyr)
    return output

def c_lstm(vec_lyr_train, embd_lyr, max_len, keep_prob, filter_size, num_filters, lstm_units, lstm_nlayers, num_labels, l2_reg_lambda, embd_dim):
    input_seq = tf.keras.layers.Input(shape = (1, ), dtype=tf.string)
    print("input_seq: ", input_seq.shape)
    vec_lyr = vec_lyr_train(input_seq)
    print("vec layer: ", vec_lyr.shape)
    x = embd_lyr(vec_lyr)
    print("embedding layer output dimension: ", x.shape)
    x = tf.expand_dims(x, axis = -1)
    print("x dimension after expanding: ", x.shape)
    conv_input = Dropout(keep_prob)(x)
    conv_outputs = []
    #feature map size
    max_feature_len = max_len - filter_size + 1
    print("feature length: ", max_feature_len)
    #for filter_size in filter_sizes:
        # (-1, max_len - filter_size + 1, 1, filter_num)
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(filter_size, embd_dim), activation='relu')(conv_input)
    print("conv output: ", x.shape)
        # (-1, max_len - filter_size + 1, filter_num)
    x = x[:, :max_feature_len, 0, :]
    print("conv output after reshaping: ", x.shape)
    conv_outputs.append(x)

    if conv_outputs.__len__() > 1:
        # (-1, max_feature_len, filter_num * n)
        x = tf.concat(conv_outputs, axis=-1)
    else:
        x = conv_outputs[0]
    lstm_cells = [tf.keras.layers.LSTMCell(lstm_units, dropout=(1 - keep_prob)) for _ in range(lstm_nlayers)]
    # final_state: (-1, lstm_hidden_size)
    final_state = tf.keras.layers.RNN(lstm_cells, return_state=False)(x)
    # (-1, num_class)
    outputs = tf.keras.layers.Dense(num_labels,
                                    activation='softmax',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg_lambda),
                                    bias_regularizer=tf.keras.regularizers.l2(l2_reg_lambda))(final_state)
    mdl = tf.keras.Model(input_seq, outputs)


    return mdl


def train(mdl, cnfg, train_x, train_y, valid_x, valid_y, test_x, test_y, callback):
    #wandb.init(project = 'lstm_model')
    mdl.compile(loss=cnfg['loss'], optimizer=cnfg['optimizer'], metrics=tf.keras.metrics.AUC(num_thresholds = 200,
                         curve = "PR",
                         multi_label= False,
                         from_logits= False))
    if not test_x and not test_y:
        _ = mdl.fit(train_x, train_y, batch_size=cnfg['batch'], epochs=cnfg['epochs'],
                validation_data=(valid_x, valid_y), callbacks=callback)
        _, accuracy = mdl.evaluate(valid_x, valid_y)
    elif not test_x and not test_y and not valid_y and not valid_x:
        _ = mdl.fit(train_x, train_y, batch_size=cnfg.batch, epochs=cnfg.epochs, callbacks=callback)
        _, accuracy = mdl.evaluate(test_x, test_y)
        #print('test loss and accuracy :', loss, accuracy)
    else:
        mdl = None
    return mdl , accuracy

def ensemble_prediction(members, valid_x):
    pred_y = [mdl.predict(valid_x) for mdl in members]
    pred_y = array(pred_y)
    print(pred_y)
    pred_prob_y = np.sum(pred_y, axis = 0)/len(members)
    print(pred_prob_y)
    pred_y = np.argmax(pred_prob_y, axis = 1)
    return pred_y, pred_prob_y


def evaluate_ensemble(members, n_members, valid_x, valid_y):
    new_members = members[:n_members]
    pred_y, pred_prob_y = ensemble_prediction(new_members, valid_x)
    acc = accuracy_score(valid_y, pred_y)
    clf_rep = classification_report(valid_y, pred_y, output_dict=True)
    roc_auc = roc_auc_score(y_true = valid_y, y_score = pred_prob_y, multi_class='ovo', average='weighted')
    return acc, clf_rep, roc_auc


def bagging_ensemble(estimator, n_est, n_sample, training_configs, train_x, train_y, valid_x, valid_y, callback):
    mem = []
    sc = []
    for _ in range(n_est):
        idx = [i for i in range(len(train_x))]
        train_idx = resample(idx, replace=True, n_samples=n_sample)
        print(train_idx)
        x_train_sub = train_x[train_idx]
        y_train_sub = train_y[train_idx]
        m, s = train(estimator, training_configs, x_train_sub, y_train_sub, valid_x, valid_y,
                         None, None, callback)
        mem.append(m)
        sc.append(s)
    return mem , sc


def predict(mdl, test_x):
    pred = mdl.predict(test_x)
        #.argmax(axis = -1)
    return pred

def create_sweep_config(method, epochs, batch_sizes, min_lr, max_lr, weight_decay, num_filters, filter_size,
                        lstm_units, keep_prob,a):
    sweep_config = {
        'method': method}

    parameters_dict = {
        'epochs': {
            'values': epochs
        },
        'batch_size': {
            'values': batch_sizes
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': min_lr,
            'max': max_lr
        },
        'weight_decay': {
            'values': weight_decay
        },
        'num_filters': {
            'values': num_filters
        },
        'filter_size': {
            'values': filter_size
        },
        'lstm_units': {
            'values': lstm_units
        },
        'keep_prob': {
            'values' : keep_prob
        },
        'a': {
            'values': a
        },

    }
    sweep_config['parameters'] = parameters_dict
    return sweep_config


if __name__ == '__main__':

    # Luis set
    # x, y = utils.load_data_pd('grt_training_balanced2.csv')

    #Randomized trial publications
    #x, y = utils.load_data_pd('RT_concat.csv', 'Text', 'Category')
    #x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.40, random_state=42)
    x_train , y_train = utils.load_data_pd('RT_train.csv', 'text', 'labels')
    x_valid , y_valid = utils.load_data_pd('RT_test.csv', 'text', 'labels')


    # run this in case of using categorical cross entropy loss, otherwise ignore if the loss is sparse categorical cross entropy
    y_train_cat = tf.keras.utils.to_categorical(y_train , 4)
    y_valid_cat = tf.keras.utils.to_categorical(y_valid , 4)
    max_length = utils.max_len(x_train)

    # prevention data
    # x_train, y_train = load_data_pd('ODP_FY12-20_funded_20211129_Joined.csv', 'Abstract', 'f6otherorunclear')
    # x_test, y_test = load_data('fy2021_allAC_20221109.csv', 'Abstract', 'f6otherorunclear')
    # maxLen = maxLen(x_train)
    # check_balance(y_train)
    # check_balance(y_test)

    # Payam's data
    # x, y = load_data_pd('Hodgkin_s Disease.csv', 'Text', 'Category')
    # x_train, x_test , y_train, y_test = train_test_split(x, y , test_size = 0.40, random_state = 42)
    # maxLen = maxLen(x_train)
    # check_balance(y_train)
    # check_balance(y_test)

    # w2vec vectors
    # model = KeyedVectors.load_word2vec_format("BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary=True)

    # w2vec model
    try:
        biow2v_model = fasttext.load_model("BioWordVec_PubMed_MIMICIII_d200.bin")
    except Exception as e:
        print(e)
    print('model successfully loaded')
    stop_words = utils.stop_word('nltk.txt')
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=None,
        standardize=custom_standardization,
        split='whitespace',
        ngrams=None,
        output_mode="int",
        output_sequence_length=max_length  # each sequence size
     )

    # for training

    vectorize_layer.adapt(x_train)


    embedding_dim = create_embedding_dim(biow2v_model)
    vocab_size = vectorize_layer.vocabulary_size()

    embedding_matrix = embedding_matrix(vocab_size, embedding_dim, vectorize_layer)

    print("embedding matrix size: ", embedding_matrix.shape)

    embedding_layer = create_embedding_layer(vocab_size, embedding_dim, embedding_matrix, max_length)

    model = model3(vectorize_layer, embedding_layer)


    configs = dict(
        batch=4,
        epochs=100,
        init_learning_rate=8.8e-6,
        lr_decay_rate=0.01,
        optimizer='adam',
        loss_fn = 'CategoricalCrossentropy',
        earlystopping_patience=5
    )

    run = wandb.init(project='lstm_ide', config = configs, job_type = 'train')
    config = wandb.config
    wandb_callback = wandb.keras.WandbCallback(monitor = 'auc', log_evaluation = True, save_model = False)
    callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=5),
                 #wandb.keras.WandbMetricsLogger(log_freq='epoch'),
                 # wandb.keras.WandbModelCheckpoint("models")
                 wandb_callback
                 ]

    # train and evaluate
    #electric sweep 10 settings
    model = train(model, config, x_train, y_train, x_valid, y_valid, None, None, callbacks)
    # hopeful sweep 9 settings
    model = train(model, config, x_train, y_train, x_valid, y_valid, None, None, callbacks)
    y_pred = predict(model, x_valid)
    y_true = y_valid.argmax(axis=-1)
    loss, acc = model.evaluate(x_valid, y_valid)
    wandb.log({'evaluate/acuuracy': acc})
    wandb.log({'pr': wandb.plot.roc_curve(y_true, y_pred, labels=[0,1,2,3], classes_to_plot=[1,2,3,0])})
    wandb.log({'pr': wandb.plot.pr_curve(y_true, y_pred, labels=[0, 1, 2, 3], classes_to_plot=[1, 2, 3, 0])})
    print(y_pred.shape)
    wandb.finish()

    #fine-tune lstm
    wandb.init()
    wandb_callback = wandb.keras.WandbCallback(monitor='val_loss', log_evaluation=True, save_model=False)
    callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=5),
                 # wandb.keras.WandbMetricsLogger(log_freq='epoch'),
                 # wandb.keras.WandbModelCheckpoint("models")
                 wandb_callback
                 ]
    sweep_config = utils.create_sweep_config(method = 'random', epochs=[1], batch_sizes=[4,8,16], min_lr=4e-6,
                                            max_lr=4e-4, weight_decay=[0, 0.01, 0.005])

    sweep_id = wandb.sweep(sweep=sweep_config, project='lstm_ide')

    def fine_tune(config=None):
        with wandb.init(config=config):
            config = wandb.config

            model.compile(loss='CategoricalCrossentropy', optimizer='adam', metrics=['acc'])

            _ = model.fit(x_train, y_train, batch_size=config.batch_size, epochs=config.epochs,
                        validation_data=(x_valid, y_valid), callbacks=callbacks)

            # loss, accuracy = mdl.evaluate(test_x, test_y)
            # print('test loss and accuracy :', loss, accuracy)

            return model
    count = 15
    wandb.agent(sweep_id, fine_tune, count=count)

    # fine-tune c-lstm
    embedding_matrix = embedding_matrix_clstm(vocab_size, embedding_dim, vectorize_layer)
    print("embedding matrix size: ", embedding_matrix.shape)
    embedding_layer = create_embedding_layer(vocab_size, embedding_dim, embedding_matrix, max_length)
    model = c_lstm(vectorize_layer, embedding_layer, max_length, 0.5, 3, 150, 150, 1, 4, 1.0e-3, embedding_dim)
    wandb.init()
    wandb_callback = wandb.keras.WandbCallback(monitor='val_loss', log_evaluation=True, save_model=False)
    callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=5),
                 # wandb.keras.WandbMetricsLogger(log_freq='epoch'),
                 # wandb.keras.WandbModelCheckpoint("models")
                 wandb_callback
                 ]
    sweep_config = create_sweep_config(method='random', epochs=[1], batch_sizes=[4, 8, 16], min_lr=4e-6, max_lr=4e-4,
                                       weight_decay=[0, 0.01, 0.005], num_filters=[64, 128, 256], filter_size=[2, 3, 4],
                                       lstm_units=[64, 128, 256], keep_prob=[0.3, 0.5, 0.4], a = [1])

    sweep_id = wandb.sweep(sweep=sweep_config, project='c-lstm')


    def fine_tune(config=None):
        with wandb.init(config=config):
            config = wandb.config

            model.compile(loss='SparseCategoricalCrossentropy', optimizer='RMSprop', metrics=['acc'])

            _ = model.fit(x_train, y_train, batch_size=config.batch_size, epochs=config.epochs,
                          validation_data=(x_valid, y_valid), callbacks=callbacks)

            # loss, accuracy = mdl.evaluate(test_x, test_y)
            # print('test loss and accuracy :', loss, accuracy)

            return model


    count = 15
    wandb.agent(sweep_id, fine_tune, count=count)

    #training c-lstm
    embedding_matrix = embedding_matrix_clstm(vocab_size, embedding_dim, vectorize_layer)
    print("embedding matrix size: ", embedding_matrix.shape)
    embedding_layer = create_embedding_layer(vocab_size, embedding_dim, embedding_matrix, max_length)
    model = c_lstm(vectorize_layer, embedding_layer, max_length, 0.5, 2, 256, 128, 1, 4, 7.98e-5, embedding_dim)
    model.compile(loss="SparseCategoricalCrossentropy", optimizer='sgd', metrics=['acc'])

    model.fit(x_train, y_train, batch_size=16, epochs=1, validation_data=(x_valid, y_valid))

    # test
    # model = train(model, 'CategoricalCrossentropy', 'adam', 'accuracy', callbacks,  None, None, None, None, x, y, 100, 16)

    # performance metrics
    # predict
    x_test = utils.load_data_pd("2021pubs_cleaned.csv", 'text', None)
    print("total number of examples: ", len(x_test))

    y_true = y_valid.argmax(axis = -1)
    y_pred = predict(model, x_valid)
    utils.confusion_matrix_multiclass(y_true, y_pred)
    utils.metric_multiclass(utils.confusion_matrix_multiclass(y_true, y_pred))


    #bagging ensemble improve auc 3%
    n_estimators = 2
    n_samples = 100
    configs = {'loss': 'CategoricalCrossentropy',
               'batch': 4,
               "epochs": 100,
               'init_learning_rate': 8.8e-6,
               'lr_decay_rate': 0.01,
               'optimizer': 'adam',
               'earlystopping_patience': 5}

    callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=5)]

    members , scores = bagging_ensemble(model3(vectorize_layer, embedding_layer), n_estimators, n_samples, configs,
                                        x_train, y_train_cat, x_valid, y_valid_cat, callbacks)

    eacc, eclf_rep, eroc_auc = evaluate_ensemble(members, 10, x_valid, y_valid)
    # eacc: 0.8941605839416058
    # eclf_rep :
    # {'0': {'precision': 0.9482288828337875, 'recall': 0.8854961832061069,'f1-score': 0.9157894736842106, 'support': 393},
    #  '1': {'precision': 0.8461538461538461, 'recall': 0.9368131868131868,'f1-score': 0.8891786179921775,'support': 364},
    #  '2': {'precision': 0.45454545454545453, 'recall': 0.21739130434782608,'f1-score': 0.29411764705882354,'support': 23},
    #  '3': {'precision': 1.0, 'recall': 0.9761904761904762, 'f1-score': 0.9879518072289156, 'support': 42},
    #  'accuracy': 0.8941605839416058,
    #  'macro avg': {'precision': 0.812232045883272,'recall': 0.753972787639399, 'f1-score': 0.7717593864910317,'support': 822},
    #  'weighted avg': {'precision': 0.8918594846815378, 'recall': 0.8941605839416058,'f1-score': 0.8902980071691177,'support': 822}}
    # eroc_auc : 0.9533189605845416
    # deciding on the number of estimators
    members = []
    scores = []
    n_samples = 600
    n_estimators = 10

    for _ in range(n_estimators):
        idx = [i for i in range(len(x_train))]
        train_idx = resample(idx, replace=True, n_samples=n_samples)
        x_train_sub = x_train[train_idx]
        y_train_sub = y_train_cat[train_idx]
        m, s = train(model3(vectorize_layer, embedding_layer), configs, x_train_sub, y_train_sub, x_valid, y_valid_cat,
                     None, None, callbacks)
        members.append(m)
        scores.append(s)

    acc = []
    calf_rep = []
    roc_auc = []
    for i in range(1, n_estimators + 1):
        eacc, eclf_rep, eroc_auc = evaluate_ensemble(members, i, x_valid, y_valid)
        print("eacc: ", eacc, "eroc_auc: ", eroc_auc)
        acc.append(eacc)
        calf_rep.append(eclf_rep)
        roc_auc.append(eroc_auc)

    members
    # [0.9327672719955444,0.934296727180481,0.9436808824539185, 0.9378907084465027,
    # 0.8931477069854736,  0.9222119450569153, 0.9286913275718689, 0.9075722694396973,
    # 0.9303470253944397, 0.886892557144165]
    scores
    # [{'0': {'precision': 0.938337801608579,'recall': 0.8905852417302799, 'f1-score': 0.9138381201044387, 'support': 393},
    #   '1': {'precision': 0.845,  'recall': 0.9285714285714286, 'f1-score': 0.8848167539267017, 'support': 364},
    #   '2': {'precision': 0.4, 'recall': 0.17391304347826086, 'f1-score': 0.24242424242424243,'support': 23},
    #   '3': {'precision': 1.0, 'recall': 0.9285714285714286, 'f1-score': 0.962962962962963,'support': 42},
    #   'accuracy': 0.889294403892944, 'macro avg': {'precision': 0.7958344504021447, 'recall': 0.7304102855878494, 'f1-score': 0.7510105198545864,'support': 822},
    #   'weighted avg': {'precision': 0.8850933771680919,'recall': 0.889294403892944, 'f1-score': 0.8847103183096909, 'support': 822}},
    #  {'0': {'precision': 0.9507772020725389, 'recall': 0.9338422391857506, 'f1-score': 0.9422336328626444, 'support': 393},
    #   '1': {'precision': 0.8891752577319587,'recall': 0.9478021978021978, 'f1-score': 0.9175531914893617, 'support': 364},
    #   '2': {'precision': 0.4444444444444444, 'recall': 0.17391304347826086, 'f1-score': 0.25, 'support': 23},
    #   '3': {'precision': 1.0, 'recall': 0.9285714285714286, 'f1-score': 0.962962962962963, 'support': 42},
    #   'accuracy': 0.9184914841849149, 'macro avg': {'precision': 0.8210992260622355, 'recall': 0.7460322272594095, 'f1-score': 0.7681874468287422,  'support': 822},
    #   'weighted avg': {'precision': 0.9118460540768406, 'recall': 0.9184914841849149, 'f1-score': 0.9129946762306465, 'support': 822}},
    #  {'0': {'precision': 0.9437340153452686, 'recall': 0.9389312977099237, 'f1-score': 0.9413265306122449, 'support': 393},
    #   '1': {'precision': 0.8923884514435696,  'recall': 0.9340659340659341, 'f1-score': 0.912751677852349,'support': 364},
    #   '2': {'precision': 0.45454545454545453, 'recall': 0.21739130434782608,  'f1-score': 0.29411764705882354, 'support': 23},
    #   '3': {'precision': 1.0, 'recall': 0.9285714285714286, 'f1-score': 0.962962962962963, 'support': 42},
    #   'accuracy': 0.916058394160584,'macro avg': {'precision': 0.8226669803335732, 'recall': 0.7547399911737782, 'f1-score': 0.7777897046215951, 'support': 822},
    #   'weighted avg': {'precision': 0.9101841968499944, 'recall': 0.916058394160584, 'f1-score': 0.9116692063207599,  'support': 822}},
    #  {'0': {'precision': 0.9296482412060302, 'recall': 0.9414758269720102, 'f1-score': 0.9355246523388117, 'support': 393},
    #   '1': {'precision': 0.8885941644562334, 'recall': 0.9203296703296703, 'f1-score': 0.9041835357624832, 'support': 364},
    #   '2': {'precision': 0.625,  'recall': 0.21739130434782608, 'f1-score': 0.3225806451612903, 'support': 23},
    #   '3': {'precision': 1.0, 'recall': 0.9285714285714286, 'f1-score': 0.962962962962963,  'support': 42},
    #   'accuracy': 0.9111922141119222, 'macro avg': {'precision': 0.8608106014155659,  'recall': 0.7519420575552338, 'f1-score': 0.781312949056387, 'support': 822},
    #   'weighted avg': {'precision': 0.9065389716010205, 'recall': 0.9111922141119222, 'f1-score': 0.9058975604256095, 'support': 822}},
    #  {'0': {'precision': 0.9215686274509803, 'recall': 0.9567430025445293, 'f1-score': 0.9388264669163545, 'support': 393},
    #   '1': {'precision': 0.9068493150684932, 'recall': 0.9093406593406593, 'f1-score': 0.9080932784636488,'support': 364},
    #   '2': {'precision': 0.6,'recall': 0.2608695652173913, 'f1-score': 0.36363636363636365,'support': 23},
    #   '3': {'precision': 1.0, 'recall': 0.9285714285714286, 'f1-score': 0.962962962962963, 'support': 42},
    #   'accuracy': 0.9148418491484185, 'macro avg': {'precision': 0.8571044856298684, 'recall': 0.7638811639185021, 'f1-score': 0.7933797679948325,'support': 822},
    #   'weighted avg': {'precision': 0.9100603665123683, 'recall': 0.9148418491484185, 'f1-score': 0.9103562477700442, 'support': 822}},
    #  {'0': {'precision': 0.9266503667481663, 'recall': 0.9643765903307888, 'f1-score': 0.9451371571072319, 'support': 393},
    #   '1': {'precision': 0.9123287671232877, 'recall': 0.9148351648351648,'f1-score': 0.9135802469135803,'support': 364},
    #   '2': {'precision': 0.6666666666666666, 'recall': 0.2608695652173913, 'f1-score': 0.37500000000000006,'support': 23},
    #   '3': {'precision': 1.0, 'recall': 0.9285714285714286, 'f1-score': 0.962962962962963, 'support': 42},
    #   'accuracy': 0.9209245742092458, 'macro avg': {'precision': 0.8764114501345301, 'recall': 0.7671631872386933, 'f1-score': 0.7991700917459439, 'support': 822},
    #   'weighted avg': {'precision': 0.9167817502411673, 'recall': 0.9209245742092458,'f1-score': 0.916121115649793,'support': 822}},
    #  {'0': {'precision': 0.9221411192214112, 'recall': 0.9643765903307888, 'f1-score': 0.9427860696517413, 'support': 393},
    #   '1': {'precision': 0.9171270718232044,  'recall': 0.9120879120879121, 'f1-score': 0.9146005509641872, 'support': 364},
    #   '2': {'precision': 0.6, 'recall': 0.2608695652173913, 'f1-score': 0.36363636363636365, 'support': 23},
    #   '3': {'precision': 1.0, 'recall': 0.9285714285714286, 'f1-score': 0.962962962962963 'support': 42},
    #   'accuracy': 0.9197080291970803, 'macro avg': {'precision': 0.8598170477611539, 'recall': 0.7664763740518801, 'f1-score': 0.7959964868038139, 'support': 822},
    #   'weighted avg': {'precision': 0.9148852968341374, 'recall': 0.9197080291970803,  'f1-score': 0.9151309084333081,'support': 822}},
    #  {'0': {'precision': 0.9266503667481663,'recall': 0.9643765903307888, 'f1-score': 0.9451371571072319,'support': 393},
    #   '1': {'precision': 0.9148351648351648, 'recall': 0.9148351648351648, 'f1-score': 0.9148351648351648, 'support': 364},
    #   '2': {'precision': 0.6, 'recall': 0.2608695652173913,'f1-score': 0.36363636363636365,  'support': 23},
    #   '3': {'precision': 1.0, 'recall': 0.9285714285714286, 'f1-score': 0.962962962962963, 'support': 42},
    #   'accuracy': 0.9209245742092458, 'macro avg': {'precision': 0.8603713828958328,'recall': 0.7671631872386933,'f1-score': 0.7966429121354308,'support': 822},
    #   'weighted avg': {'precision': 0.9160262702336124, 'recall': 0.9209245742092458, 'f1-score': 0.9163588607679111, 'support': 822}},
    #  {'0': {'precision': 0.9245742092457421, 'recall': 0.9669211195928753, 'f1-score': 0.945273631840796, 'support': 393},
    #   '1': {'precision': 0.9146005509641874, 'recall': 0.9120879120879121, 'f1-score': 0.9133425034387895, 'support': 364},
    #   '2': {'precision': 0.6666666666666666, 'recall': 0.2608695652173913,'f1-score': 0.37500000000000006,  'support': 23},
    #   '3': {'precision': 1.0, 'recall': 0.9285714285714286, 'f1-score': 0.962962962962963, 'support': 42},
    #   'accuracy': 0.9209245742092458, 'macro avg': {'precision': 0.876460356719149, 'recall': 0.7671125063674018, 'f1-score': 0.7991447745606373,'support': 822},
    #   'weighted avg': {'precision': 0.9167951315302606, 'recall': 0.9209245742092458,'f1-score': 0.9160810863863708, 'support': 822}},
    #  {'0': {'precision': 0.9238329238329238, 'recall': 0.9567430025445293, 'f1-score': 0.9400000000000001, 'support': 393},
    #   '1': {'precision': 0.904891304347826, 'recall': 0.9148351648351648,'f1-score': 0.9098360655737705, 'support': 364},
    #   '2': {'precision': 0.75, 'recall': 0.2608695652173913, 'f1-score': 0.3870967741935483, 'support': 23},
    #   '3': {'precision': 1.0, 'recall': 0.9285714285714286, 'f1-score': 0.962962962962963, 'support': 42},
    #   'accuracy': 0.9172749391727494, 'macro avg': {'precision': 0.8946810570451875,'recall': 0.7652547902921285,'f1-score': 0.7999739506825705, 'support': 822},
    #   'weighted avg': {'precision': 0.9144729608867977, 'recall': 0.9172749391727494, 'f1-score': 0.9123454964960444, 'support': 822}}]
    calf_rep
    # [0.9427400353380171, 0.9517794311742253,  0.9616016511245157,
    # 0.9668701936399576, 0.9632297318786945, 0.9609982174553339,
    # 0.9611353144653793, 0.9592162471319337,  0.9599155715601241, 0.9599349908125929]
    roc_auc













