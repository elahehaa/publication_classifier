
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils import resample
from transformers import AutoTokenizer, TrainingArguments, Trainer, pipeline, AutoModelForSequenceClassification
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda
import wandb
import utils
import torch
from torch.nn import functional as F

torch.cuda.empty_cache()


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


# experiments using 60/40 - hyperparam tuning
def fine_tune(mdl, output_dir, run_name, tokenized_train, tokenized_valid, config=None):
    with wandb.init(config=config):
        config = wandb.config

        training_args = TrainingArguments(
            output_dir=output_dir,
            report_to='wandb',
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=16,
            save_strategy='epoch',
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            load_best_model_at_end=True,
            remove_unused_columns=False,
            fp16=True,
            save_total_limit=1,
            run_name=run_name,
        )

        trainer = Trainer(
            model_init=mdl,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            # ?
            compute_metrics=utils.compute_metrics_fn
        )

        trainer.train()
        return trainer




# candidate model more epochs
def train(mdl, tokenized_train, tokenized_valid, learning_rate, w_decay, run_name, epoch, batch_size):
    training_args = TrainingArguments(
        output_dir=run_name,
        report_to='wandb',
        num_train_epochs=epoch,
        learning_rate=learning_rate,
        weight_decay=w_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        remove_unused_columns=False,
        save_total_limit=1,
        run_name=run_name,
    )

    trainer = Trainer(
        model=mdl,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        compute_metrics=utils.compute_metrics_fn
    )

    trainer.train()
    return trainer

# train final model with all data
def train_final(mdl, learning_rate, w_decay, run_name, epoch, batch_size, train_tkn):
    training_args = TrainingArguments(
        output_dir=run_name,
        num_train_epochs=epoch,
        learning_rate=learning_rate,
        weight_decay=w_decay,
        per_device_train_batch_size=batch_size,
        save_strategy='epoch',
        logging_strategy='epoch',
        remove_unused_columns=False,
        fp16=True,
        save_total_limit=1,
        run_name=run_name,
    )

    trainer = Trainer(
        model=mdl,
        args=training_args,
        train_dataset=train_tkn,
        compute_metrics=utils.compute_metrics_fn
    )

    trainer.train()
    return trainer

def tokenize_data(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=max_length)

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

def ensemble_prediction(members, valid_x):
    size = len(members)
    pred_y = [mdl.predict(valid_x) for mdl in members]
    #torch_logits = torch.from_numpy(pred_y[:][0])
    #print(len(torch_logits), torch_logits)
    #pred_y = F.softmax(torch_logits, dim = -1).numpy()
    #print(len(pred_y))
    #pred_prob_y = np.sum(pred_y, axis = 0)
    #pred_y = np.argmax(pred_prob_y, axis = 1)
    return pred_y


def evaluate_ensemble(members, n_members, valid_x, valid_y):
    members = members[:n_members]
    pred_y, pred_prob_y = ensemble_prediction(members, valid_x)
    acc = accuracy_score(valid_y, pred_y)
    clf_rep = classification_report(valid_y, pred_y, output_dict=True)
    roc_auc = roc_auc_score(y_true = valid_y, y_score = pred_prob_y, multi_class='ovo', average='weighted')
    return acc, clf_rep, roc_auc



if __name__ == '__main__':

    free_gpu_cache()
    model_dict = {'PubMedBERT': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                  'BioLinkBERT': 'michiyasunaga/BioLinkBERT-base',
                  'DistillBERT': 'distilbert-base-uncased'}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_length = 512
    model_checkpoint = model_dict['PubMedBERT']
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    num_labels = 4
    # training and validation - 60/40
    dataset = utils.load_data(['RT_train.csv'], ['RT_test.csv'], None)

    train_tokenized = dataset['train'].map(tokenize_data, batched=True)
    valid_tokenized = dataset['validation'].map(tokenize_data, batched=True)

    # training with whole data for final prediction
    # dataset = utils.load_data(['RT_train.csv', 'RT_test.csv'], None, None)
    # train_tokenized = dataset['train'].map(utils.tokenize_data, max_length, tokenizer, batched=True)

    data_collator = utils.data_collator(tokenizer)
    # model = utils.model_init(model_checkpoint, num_labels)

    # hyperparam search
    epochs = [1]
    batch_sizes = [4, 8, 16]
    minLR = 2e-7
    maxLR = 2e-4
    weight_decay = [0.0, 0.01, 0.005, 0.001]
    sweep_config = utils.create_sweep_config('random', epochs=epochs, batch_sizes=batch_sizes, min_lr=minLR,
                                             max_lr=maxLR, weight_decay=weight_decay)
    project = 'ft_all+PK_PubMedBERT_ide'
    sweep_id = wandb.sweep(sweep=sweep_config, project=project)

    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=4)
        return model

    def fine_tune(config=None):
        with wandb.init(config=config):
            config = wandb.config

            training_args = TrainingArguments(
                output_dir=project,
                report_to='wandb',
                num_train_epochs=config.epochs,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=16,
                save_strategy='epoch',
                evaluation_strategy='epoch',
                logging_strategy='epoch',
                load_best_model_at_end=True,
                remove_unused_columns=False,
                # fp16=True,
                save_total_limit=1,
                run_name=project,
            )

            trainer = Trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=valid_tokenized,
                # ?
                compute_metrics=utils.compute_metrics_fn
            )

            trainer.train()

    count = 15
    wandb.agent(sweep_id, fine_tune, count=count)

    # more epochs
    # selected1
    # train(4.495e-5, 0, "ft_all_PubMedBERT_selected1", 10, 16)
    # selected2
    # train(2.896e-5, 0.01, "ft_all_PubMedBERT_selected2", 10, 8)
    # selected3 - candidate
    # train(mdl = model, tokenized_train = train_tokenized, tokenized_valid = valid_tokenized, learning_rate = 1.098e-4, w_decay = 0.01, run_name = project, epoch = 10, batch_size = 8)
    # PubMedBERT3selected
    # train(4.3e-5, 0.01, "ft_all+PK_PubMedBERT3_selected2", 4, 16)

    # PubMedBERT_ide project,  proud_sweep with 10 epochs
    train(model_init, tokenized_train=train_tokenized, tokenized_valid=valid_tokenized, learning_rate=5.017e-5,
          w_decay=0.01, run_name=project, epoch=10, batch_size=16)
    # PubMedBERT_ide project,  scarlet_sweep with 10 epochs
    train(model_init, tokenized_train=train_tokenized, tokenized_valid=valid_tokenized, learning_rate=4.603e-5,
          w_decay=0.005, run_name=project, epoch=10, batch_size=16)
    # PubMedBERT_ide project,  frosty_sweep with 10 epochs
    train(model_init, tokenized_train=train_tokenized, tokenized_valid=valid_tokenized, learning_rate=1.372e-5,
          w_decay=0, run_name="frosty-sweep-10epoch", epoch=10, batch_size=4)

    # final model training with train+validation data
    # train_final(model, 1.098e-4, 0.01, "final_PubMedBERT", 3, 8)

    # validation set performance
    # candidate_model = "ft_all_PubMedBERT_selected3/checkpoint-465"
    # candidate_model = utils.model_init(candidate_model, num_labels)
    # clf = pipeline('text_classification', model = candidate_model, tokenizer = tokenizer)
    # y_true, y_pred = utils.outputs(dataset = dataset, max_length= max_length, clf=clf)
    # cm = utils.confusion_matrix_multiclass(y_true, y_pred)
    # utils.metric_multiclass(cm)

    # predicting unknown data
    # candidate_model = "ft_all_PubMedBERT_selected3/checkpoint-465"
    # candidate_model = utils.model_init(candidate_model, num_labels)
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # dataset_unknown = utils.load_data_unknown(["2021pubs_cleaned.csv"])
    # unknown_tokenized = dataset_unknown['unknown'].map(utils.tokenize_data, max_length, tokenizer, batched=True)
    # clf = pipeline("text-classification", model=candidate_model, tokenizer=tokenizer)
    # file_name = "output"
    # utils.create_save_predictions(file_name, dataset_unknown, clf, max_length)

    #bagging ensemble of candidate model
    members = []
    scores = []
    logits = []
    n_samples = 600
    n_estimators = 10
    candidate_model = "ft_all_PubMedBERT_selected3/checkpoint-465"
    #candidate_model = model_dict['PubMedBERT']
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def load_model():
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=4)
        return model

    for _ in range(n_estimators):
        idx = [i for i in range(len(train_tokenized))]
        train_idx = resample(idx, replace=True, n_samples=n_samples)
        train_sub = train_tokenized.select(train_idx)
        model = load_model()
        m = train(model, tokenized_train=train_sub, tokenized_valid=valid_tokenized, learning_rate=1.372e-5,
          w_decay=0, run_name="frosty-sweep-10epoch", epoch=7, batch_size=4)
        logit = m.predict(valid_tokenized)
        members.append(m)
        logits.append(logit[0])

    acc = []
    calf_rep = []
    roc_auc = []
    for i in range(1, n_estimators + 1):
        eacc, eclf_rep, eroc_auc = evaluate_ensemble(members, i, valid_tokenized, dataset['validation']['labels'])
        print("eacc: ", eacc, "eroc_auc: ", eroc_auc)
        acc.append(eacc)
        calf_rep.append(eclf_rep)
        roc_auc.append(eroc_auc)




