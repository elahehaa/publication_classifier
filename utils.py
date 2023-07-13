import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification


def load_data(train_file_path, valid_file_path, test_file_path):
    if train_file_path and valid_file_path and test_file_path:
        dataset = load_dataset("csv", data_files={'train': train_file_path, 'validation': valid_file_path,
                                                  'test': test_file_path})
    elif not valid_file_path and not test_file_path:
        dataset = load_dataset("csv", data_files={'train': train_file_path})
    elif not test_file_path:
        dataset = load_dataset("csv", data_files={'train': train_file_path, 'validation': valid_file_path})
    else:
        dataset = None
    return dataset


def load_data_unknown(unknown_file_path):
    dataset = load_dataset("csv", data_files={'unknown': unknown_file_path})
    return dataset


def load_data_pd(file_path, feature, label):
    df = pd.read_csv(file_path, encoding='latin-1')
    x = df[feature]
    x.dropna(inplace=True)
    if label:
        y = df[label]
        y.dropna(inplace=True)
        return x, y
    print("total number of examples: ", len(x))
    if not label:
        return x


def check_balance(y):
    pos = 0
    neg = 0
    for item in y:
        if item == 1:
            pos += 1
        else:
            neg += 1
    print("Out of ", pos + neg, " examples ", pos, " are positive and ", neg, " are negative.")


def max_len(x_train):
    ml = len(max(x_train, key=len).split())
    print("Max length of training example:", ml)
    return ml


def data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize_data(example, max_length):
    return tokenizer(example['text'],  padding = 'max_length', truncation=True, max_length=max_length)


def model_init(model_checkpoint):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=4)
    # id2label={index: label for index, label in enumerate(labels.names)},
    # label2id={label: index for index, label in enumerate(labels.names)})
    return model


# method : 'random', 'grid'
# epoch : list of epochs
# weight_decay : list of weight decay
def create_sweep_config(method, epochs, batch_sizes, min_lr, max_lr, weight_decay):
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

    }
    sweep_config['parameters'] = parameters_dict
    return sweep_config


def create_sweep_config(method, epochs, batch_sizes, min_lr, max_lr, weight_decay, num_filters, filter_size,
                        lstm_units):
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

    }
    sweep_config['parameters'] = parameters_dict
    return sweep_config



def confusion_matrix_binary(y_true, y_pred):
    tp, fn, tn, fp = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 0:
            fn += 1
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            tn += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            fp += 1
    return [tp, fn, tn, fp]


def metric_binary(tp, fn, tn, fp):
    try:
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        f1 = (2 * precision * recall) / (precision + recall)

    except ZeroDivisionError:
        pass

    print("recall: ", recall, "specificity: ", specificity, "f1: ", f1, "precision:", precision, "accuracy:", accuracy,
          "false negatives: ", fn, "false positive :", fp, "true positive:", tp, "true neg: ", tn)


def compute_metrics_fn(eval_preds):
    metrics = dict()

    accuracy_metric = load_metric('accuracy')
    precision_metric = load_metric('precision')
    recall_metric = load_metric('recall')
    f1_metric = load_metric('f1')

    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    preds = np.argmax(logits, axis=-1)

    metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
    metrics.update(precision_metric.compute(predictions=preds, references=labels, average='macro'))
    metrics.update(recall_metric.compute(predictions=preds, references=labels, average='macro'))
    metrics.update(f1_metric.compute(predictions=preds, references=labels, average='macro'))
    return metrics


def confusion_matrix_multiclass(y_true, y_pred):
    cm = {'1': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
          '2': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
          '3': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
          }
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1:
                cm['1']['tp'] += 1
            elif y_true[i] == 0:
                cm['1']['tn'] += 1
                cm['2']['tn'] += 1
                cm['3']['tn'] += 1
            elif y_true[i] == 2:
                cm['2']['tp'] += 1
            else:
                cm['3']['tp'] += 1
        elif y_true[i] != y_pred[i]:
            if y_true[i] == 1:
                cm['1']['fn'] += 1
            elif y_true[i] == 0:
                if y_pred[i] == 1:
                    cm['1']['fp'] += 1
                elif y_pred[i] == 2:
                    cm['2']['fp'] += 1
                elif y_pred[i] == 3:
                    cm['3']['fp'] += 1
            elif y_true[i] == 2:
                cm['2']['fn'] += 1
            else:
                cm['3']['fn'] += 1

    return cm


def metric_multiclass(confusion_matrix):
    for i, val in confusion_matrix.items():
        recall, specificity, precision, accuracy, f1 = 0, 0, 0, 0, 0
        try:
            recall = val['tp'] / (val['tp'] + val['fn'])
            specificity = val['tn'] / (val['tn'] + val['fp'])
            precision = val['tp'] / (val['tp'] + val['fp'])
            accuracy = (val['tp'] + val['tn']) / (val['tp'] + val['fn'] + val['fp'] + val['tn'])
            f1 = (2 * precision * recall) / (precision + recall)

        except ZeroDivisionError:
            pass

        print("for class: ", i, "recall: ", recall, "specificity: ", specificity, "f1: ", f1, "precision:", precision,
              "accuracy:", accuracy,
              "false negatives: ", val['fn'], "false positive :", val['fp'], "true positive:", val['tp'], "true neg: ",
              val['tn'])


def create_save_predictions(output_file_name, dataset, clf, max_length):
    y_pred = {}
    for i in range(len(dataset['unknown'])):
        p = clf(dataset['unknown']['text'][i], truncation=True, max_length=max_length)[0]['label']
        if p == 'LABEL_0':
            y_pred[dataset['unknown'][i]['PMID']] = 0
        elif p == 'LABEL_1':
            y_pred[dataset['unknown'][i]['PMID']] = 1
        elif p == 'LABEL_2':
            y_pred[dataset['unknown'][i]['PMID']] = 2
        else:
            y_pred[dataset['unknown'][i]['PMID']] = 3
    predictions = pd.DataFrame.from_dict(y_pred, orient='index', columns=['Category'])
    predictions.index.names = ['PMID']
    predictions.to_csv(output_file_name)


def outputs(dataset, max_length, clf):
    y_pred = {}
    y_true = {}
    for i in range(len(dataset['validation'])):
        y_true[i] = dataset['validation']['labels'][i]
        p = clf(dataset['validation']['text'][i], truncation=True, max_length=max_length)[0]['label']
        if p == 'LABEL_0':
            y_pred[i] = 0
        elif p == 'LABEL_1':
            y_pred[i] = 1
        elif p == 'LABEL_2':
            y_pred[i] = 2
        else:
            y_pred[i] = 3
    return y_true, y_pred


def stop_word(file_path):
    return pd.read_csv(file_path)
