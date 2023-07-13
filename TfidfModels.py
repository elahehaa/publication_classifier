from sklearn.svm import SVC
import utils
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score,  classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

stop_words = pd.read_csv('nltk.txt')

class LinearModel(object):
    def __init__(self, clf, params = None):
        self.clf = clf(**params)

    def preprocess(self,txt):
        txt = str.replace(txt, '/', ' / ')
        txt = str.replace(txt, '.-', ' .- ')
        txt = str.replace(txt, '\'', ' .\' ')
        txt = str.replace(txt, '[^\w\s]', '')  # remove punctuations
        txt = str.replace(txt, '\d+', '')  # remove numbers
        txt = str.lower(txt)  # lower case


        for each in stop_words.values:
            txt = str.replace(txt, ' ' + each[0] + ' ', r" ")
        txt = str.replace(txt, " +", " ")

        return txt

    def train(self, train_tfidf_x, train_y):
        _ = self.clf.fit(train_tfidf_x, train_y)

    def predict(self,  valid_tfidf_x):
        valid_pred = self.clf.predict(valid_tfidf_x)
        prob = self.clf.predict_proba(valid_tfidf_x)
        #log_prob = self.clf.predict_log_proba(x_valid_tfidf)
        return  valid_pred, prob

    def report_results(self, valid_y, valid_pred_y, valid_prob_y):
        conf_matrix = confusion_matrix(valid_y, valid_pred_y)
        acc = accuracy_score(valid_y, valid_pred_y)
        auc_roc = roc_auc_score(y_true = valid_y, y_score = valid_prob_y, multi_class='ovo', average='weighted')
        clf_report = classification_report(valid_y, valid_pred_y, output_dict=True)
        return conf_matrix, acc, auc_roc, clf_report

    def grid_search(self, params, train_tfidf_x, train_y):
        grid = GridSearchCV(estimator = self.clf, param_grid= params, cv = 5 , return_train_score = True)
        grid.fit(train_tfidf_x, train_y)
        return grid.best_params_

    def grid_search_woCV(self, params, train_tfidf_x, train_y, valid_x, valid_y):
        b_score = 0
        for C in params['C']:
            for gamma in params['gamma']:
                for kernel in params['kernel']:
                    mdl = self.train(train_tfidf_x, train_y, C)
                    score = mdl.score(valid_x, valid_y)
                    if score > b_score:
                        b_score = score
                        best_params = {'C': C, 'gamma': gamma, 'kernel': kernel}
                        best_mdl = mdl
        return b_score, best_params, best_mdl

    def tfidf_vec(self, train_path, valid_path):
        if train_path and valid_path:
            train_x, train_y = utils.load_data_pd(train_path, 'text', 'labels')
            valid_x, valid_y = utils.load_data_pd(valid_path, 'text', 'labels')
            train_cleaned_x = train_x.map(self.preprocess)
            valid_cleaned_x = valid_x.map(self.preprocess)
            vec = TfidfVectorizer()
            vec.fit(train_cleaned_x)
            train_tfidf_x = vec.transform(train_cleaned_x)
            valid_tfidf_x = vec.transform(valid_cleaned_x)
            return train_tfidf_x , valid_tfidf_x , train_y , valid_y
        elif train_path:
            x, y = utils.load_data_pd(train_path, 'Text', 'Category')
            x = x.map(self.preprocess)
            vec = TfidfVectorizer()
            vec.fit(x)
            tfidf_x = vec.transform(x)
            return tfidf_x, y






if __name__ == '__main__' :
    parameters = {'C': 100,
                  'degree': 2,
                  'gamma': 0.01,
                  'kernel': 'rbf',
                  'probability' : True
                  }
    model = LinearModel(clf = SVC, params = parameters)
    x_train, y_train = utils.load_data_pd('RT_train.csv', 'text', 'labels')
    x_valid, y_valid = utils.load_data_pd('RT_test.csv', 'text', 'labels')

    x_train_cleaned = x_train.map(model.preprocess)
    x_valid_cleaned = x_valid.map(model.preprocess)

    x_train_tfidf, x_valid_tfidf, y_train, y_valid = model.tfidf_vec('RT_train.csv', 'RT_test.csv')
    # model.train(x_train_tfidf, y_train)
    # y_valid_pred , y_valid_prob = model.predict(x_valid_tfidf)
    # conf_matrix, acc, auc_roc, clf_report = model.report_results(y_valid, y_valid_pred , y_valid_prob)
    # print(conf_matrix)
    # print(acc)
    # print(auc_roc)
    # print(clf_report)



    #tuning
    #SVC
    # parameters = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000],
    #               'degree': [2, 3, 4, 5],
    #               'gamma': [0.001, 0.01, 0.1, 0.5, 1],
    #               'kernel': ['rbf', 'poly']
    #               }
    #
    # best_score, best_parameters, best_model = model.grid_search(parameters, x_train_tfidf, y_train, x_valid_tfidf, y_valid)
    #
    #
    # # {'C': 50, 'gamma': 0.1, 'kernel': 'rbf'}
    # # SVC(C=100, degree=2, gamma=0.01)
    # clf_candidate = model.train(y_train, )
    # y_valid_pred, probability, log_probability = model.predict(clf_candidate, x_valid_tfidf)
    # classification_report(y_valid, y_valid_pred, output_dict=True)




    #multinomial naive bayes tuning
    nb_params = {'alpha':[0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000],
                 'fit_prior': [True, False]}
    model = LinearModel(clf = MultinomialNB, params = {'alpha':1})
    x_train_cleaned = x_train.map(model.preprocess)
    x_valid_cleaned = x_valid.map(model.preprocess)

    x_train_tfidf, x_valid_tfidf, y_train, y_valid = model.tfidf_vec('RT_train.csv', 'RT_test.csv')
    nb_best_params = model.grid_search(params = nb_params, train_tfidf_x = x_train_tfidf, train_y = y_train)
    print(nb_best_params)
    model = LinearModel(clf = MultinomialNB, params = nb_best_params)
    model.train(x_train_tfidf, y_train)
    y_valid_pred, y_valid_prob = model.predict(x_valid_tfidf)
    model.report_results(y_valid, y_valid_pred, y_valid_prob)

    #lr tuning
    lr_params = {'penalty': ['l2'],
                 'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                 'solver': ['newton-cg', 'sag', 'saga', 'lbfgs']}
    model = LinearModel(clf = LogisticRegression, params={'C': 1})
    lr_best_params = model.grid_search(lr_params, x_train_tfidf, y_train)
    model = LinearModel(clf = LogisticRegression, params = lr_best_params)
    model.train(x_train_tfidf, y_train)
    y_valid_pred, y_valid_prob = model.predict(x_valid_tfidf)
    model.report_results(y_valid, y_valid_pred, y_valid_prob)
    print(lr_best_params)
    y_valid_prob
    y_valid_pred

















