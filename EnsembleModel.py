from mlxtend.evaluate import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from TfidfModels import LinearModel
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import  MultinomialNB
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from mlxtend.classifier import EnsembleVoteClassifier
import numpy as np


#creates a selection of boosting (adaboost, xgboost), averaging, bagging (random forest), voting ensemble learner
#types are : voting, average, bagging, ada , xg
class EnsembleModel():
    def __init__(self, ensembleClf, params=None):
        self.ensembleClf = ensembleClf(**params)
        self.estimators = []


    def addModel(self, trained_mdl):
        self.estimators.append(trained_mdl)


    def train(self, train_x, train_y):
        self.ensembleClf.fit(train_x, train_y)

    def grid_search(self, params, train_tfidf_x, train_y):
        grid = GridSearchCV(estimator = self.clf, param_grid= params, cv = 5 , return_train_score = True)
        grid.fit(train_tfidf_x, train_y)
        return grid.best_params_

    def predict(self,  valid_tfidf_x):
        valid_pred = self.ensembleClf.predict(valid_tfidf_x)
        prob = self.ensembleClf.predict_proba(valid_tfidf_x)
        #log_prob = self.clf.predict_log_proba(x_valid_tfidf)
        return  valid_pred, prob

    def report_results(self, valid_y, valid_pred_y):
        conf_matrix = confusion_matrix(valid_y, valid_pred_y)
        acc = accuracy_score(valid_y, valid_pred_y)
        clf_report = classification_report(valid_y, valid_pred_y, output_dict=True)
        return conf_matrix, acc,  clf_report


if __name__ == '__main__' :

    #{'0': {'precision': 0.9251207729468599, 'recall': 0.9745547073791349, 'f1-score': 0.9491945477075588, 'support': 393},
    # '1': {'precision': 0.9355742296918768, 'recall': 0.9175824175824175, 'f1-score': 0.9264909847434118, 'support': 364},
    # '2': {'precision': 0.7272727272727273, 'recall': 0.34782608695652173, 'f1-score': 0.4705882352941176, 'support': 23},
    # '3': {'precision': 1.0, 'recall': 0.9523809523809523, 'f1-score': 0.975609756097561, 'support': 42},
    # 'accuracy': 0.9306569343065694, 'macro avg': {'precision': 0.8969919324778659, 'recall': 0.7980860410747566, 'f1-score': 0.8304708809606623, 'support': 822},
    # 'weighted avg': {'precision': 0.9280398492740046, 'recall': 0.9306569343065694, 'f1-score': 0.9270989231916483, 'support': 822}}
    #{'C': 100, 'degree': 2, 'gamma': 0.01, 'kernel': 'rbf'}
    svc_parameters = {'C': 100, 'degree': 2, 'gamma': 0.01, 'kernel': 'rbf', 'probability': True}
    clf1 = LinearModel(clf = SVC, params = svc_parameters)
    x_train_tfidf , x_valid_tfidf, y_train, y_valid = clf1.tfidf_vec('RT_train.csv', 'RT_test.csv')
    clf1.train(x_train_tfidf, y_train)
    y_valid_pred1, y_valid_prob1 = clf1.predict(x_valid_tfidf)
    conf_matrix1, acc1, auc_roc1, clf1_report = clf1.report_results(y_valid, y_valid_pred1, y_valid_prob1)

    # {'0': {'precision': 0.9278846153846154,'recall': 0.9821882951653944, 'f1-score': 0.954264524103832,'support': 393},
    #  '1': {'precision': 0.9277777777777778,'recall': 0.9175824175824175, 'f1-score': 0.9226519337016573, 'support': 364},
    #  '2': {'precision': 0.6,'recall': 0.2608695652173913, 'f1-score': 0.36363636363636365,  'support': 23},
    #  '3': {'precision': 1.0, 'recall': 0.8571428571428571, 'f1-score': 0.923076923076923, 'support': 42},
    #  'accuracy': 0.927007299270073,'macro avg': {'precision': 0.8639155982905983, 'recall': 0.7544457837770151,'f1-score': 0.7909074361296939, 'support': 822},
    #  'weighted avg': {'precision': 0.9223476459334123, 'recall': 0.927007299270073, 'f1-score': 0.9221461423030126, 'support': 822}})
    # {'C': 100, 'penalty': 'l2', 'solver': 'saga'}
    lr_parameters = {'C': 100, 'penalty' : 'l2' , 'solver': 'saga'}
    clf2 = LinearModel(clf=LogisticRegression, params=lr_parameters)
    clf2.train(x_train_tfidf, y_train)
    y_valid_pred2, y_valid_prob2 = clf2.predict(x_valid_tfidf)
    conf_matrix2, acc2, auc_roc2, clf2_report = clf2.report_results(y_valid, y_valid_pred2, y_valid_prob2)


    # {'alpha': 0.001, 'fit_prior': True}
    # {'0': {'precision': 0.9075, 'recall': 0.9236641221374046, 'f1-score': 0.9155107187894074, 'support': 393},
    #  '1': {'precision': 0.8354755784061697,'recall': 0.8928571428571429,'f1-score': 0.8632138114209829,'support': 364},
    #  '2': {'precision': 0.5714285714285714, 'recall': 0.17391304347826086,'f1-score': 0.26666666666666666, 'support': 23},
    #  '3': {'precision': 0.9230769230769231, 'recall': 0.5714285714285714,'f1-score': 0.7058823529411765, 'support': 42},
    #  'accuracy': 0.8710462287104623,'macro avg': {'precision': 0.809370268227916,  'recall': 0.640465719975345, 'f1-score': 0.6878183874545584, 'support': 822},
    #  'weighted avg': {'precision': 0.8669984166081919, 'recall': 0.8710462287104623,'f1-score': 0.863486535277783, 'support': 822}})
    nb_parameters = {'alpha': 0.001, 'fit_prior' : True}
    clf3 = LinearModel(clf = MultinomialNB, params = nb_parameters)
    clf3.train(x_train_tfidf, y_train)
    y_valid_pred3, y_valid_prob3 = clf3.predict(x_valid_tfidf)
    conf_matrix3, acc3, auc_roc3, clf3_report = clf3.report_results(y_valid, y_valid_pred3, y_valid_prob3)



    ens_parameters = {'clfs' : [clf1, clf2, clf3], 'weights' : [1,1,1], 'fit_base_estimators' : False}
    emdl = EnsembleModel(ensembleClf = EnsembleVoteClassifier, params = ens_parameters)
    emdl.train(x_train_tfidf, y_train)
    print('accuracy:', np.mean(y_train == emdl.predict(x_train_tfidf)))
    y_pred_valid = emdl.predict(x_valid_tfidf)
    y_pred_valid

    #directly using the model
    mdl = VotingClassifier(estimators = [('svc', SVC(C= 100, degree= 2, gamma= 0.01, kernel= 'rbf', probability= True)),
                                         ('lr', LogisticRegression(C = 100, penalty = 'l2' , solver = 'saga')),
                                         ('nb', MultinomialNB(alpha = 0.001, fit_prior =  True))], weights = [1,1,1], voting = 'soft')
    mdl.fit(x_train_tfidf, y_train)
    y_valid_pred = mdl.predict(x_valid_tfidf)
    accuracy_score( y_valid, y_valid_pred)

    print(x_train_tfidf.shape)


    print(mdl.predict())
    print(y_train)

    #pipeline?
    #direct voting calc


    # #bagging svc n_estimators = 10
    # 0.9367396593673966,
    # {'0': {'precision': 0.9385749385749386,'recall': 0.9720101781170484, 'f1-score': 0.9550000000000001, 'support': 393},
    #  '1': {'precision': 0.9436619718309859, 'recall': 0.9203296703296703,'f1-score': 0.9318497913769124, 'support': 364},
    #  '2': {'precision': 0.7058823529411765,'recall': 0.5217391304347826, 'f1-score': 0.6,'support': 23},
    #  '3': {'precision': 0.9534883720930233, 'recall': 0.9761904761904762, 'f1-score': 0.9647058823529412, 'support': 42},
    #  'accuracy': 0.9367396593673966, 'macro avg': {'precision': 0.885401908860031,'recall': 0.8475673637679944,'f1-score': 0.8628889184324634, 'support': 822},
    #  'weighted avg': {'precision': 0.9350787279221214, 'recall': 0.9367396593673966,'f1-score': 0.9353114003893183, 'support': 822}})

    # bagging svc n_estimators = 100
    # 0.9379562043795621,
    # {'0': {'precision': 0.9477611940298507,'recall': 0.9694656488549618, 'f1-score': 0.9584905660377357, 'support': 393},
    #  '1': {'precision': 0.9338842975206612, 'recall': 0.9313186813186813,'f1-score': 0.9325997248968363,'support': 364},
    #  '2': {'precision': 0.7333333333333333, 'recall': 0.4782608695652174, 'f1-score': 0.5789473684210527,'support': 23},
    #  '3': {'precision': 0.9523809523809523, 'recall': 0.9523809523809523,'f1-score': 0.9523809523809523, 'support': 42},
    #  'accuracy': 0.9379562043795621, 'macro avg': {'precision': 0.8918399443161993,'recall': 0.8328565380299533,'f1-score': 0.8556046529341443,'support': 822},
    #  'weighted avg': {'precision': 0.9358524333551322, 'recall': 0.9379562043795621,'f1-score': 0.936093530156889,'support': 822}})
    svc_bagging_params = {'base_estimator': SVC(C = 100, degree = 2, gamma = 0.01, kernel = 'rbf', probability = True), 'n_estimators' : 100}
    model = EnsembleModel(ensembleClf=BaggingClassifier, params=svc_bagging_params)
    model.train(x_train_tfidf, y_train)
    y_valid_pred, y_valid_prob = model.predict(x_valid_tfidf)
    model.report_results(y_valid, y_valid_pred)

    # #bagging lr
    # 0.9245742092457421,
    # {'0': {'precision': 0.9258373205741627, 'recall': 0.9847328244274809, 'f1-score': 0.9543773119605425, 'support': 393},
    #  '1': {'precision': 0.925, 'recall': 0.9148351648351648, 'f1-score': 0.9198895027624309, 'support': 364},
    #  '2': {'precision': 0.6, 'recall': 0.2608695652173913, 'f1-score': 0.36363636363636365,'support': 23},
    #  '3': {'precision': 1.0, 'recall': 0.8095238095238095,  'f1-score': 0.8947368421052632, 'support': 42},
    #  'accuracy': 0.9245742092457421, 'macro avg': {'precision': 0.8627093301435407, 'recall': 0.7424903410009616, 'f1-score': 0.7831600051161501,'support': 822},
    #  'weighted avg': {'precision': 0.9201387676224403, 'recall': 0.9245742092457421, 'f1-score': 0.9195287668346417, 'support': 822}})
    lr_bagging_params = {'base_estimator': LogisticRegression(C = 100, penalty = 'l2', solver = 'saga'), 'n_estimators': 10}
    model = EnsembleModel(ensembleClf=BaggingClassifier, params=lr_bagging_params)
    model.train(x_train_tfidf, y_train)
    y_valid_pred, y_valid_prob = model.predict(x_valid_tfidf)
    model.report_results(y_valid, y_valid_pred)

    # #bagging nb, 10 estimator
    # 0.864963503649635,
    # {'0': {'precision': 0.894484412470024, 'recall': 0.9491094147582697,'f1-score': 0.9209876543209876,
    #        'support': 393},
    #  '1': {'precision': 0.8368421052631579,'recall': 0.8736263736263736,'f1-score': 0.8548387096774193,'support': 364},
    #  '2': {'precision': 0.5,'recall': 0.13043478260869565, 'f1-score': 0.20689655172413793, 'support': 23},
    #  '3': {'precision': 0.8947368421052632, 'recall': 0.40476190476190477, 'f1-score': 0.5573770491803278,'support': 42},
    #  'accuracy': 0.864963503649635, 'macro avg': {'precision': 0.7815158399596113, 'recall': 0.5894831189388109, 'f1-score': 0.6350249912257182, 'support': 822},
    #  'weighted avg': {'precision': 0.8579341213928589, 'recall': 0.864963503649635,'f1-score': 0.8531361255790239,'support': 822}})

    # #bagging nb, 100 estimator
    # 0.8540145985401459,
    # {'0': {'precision': 0.8891566265060241,'recall': 0.9389312977099237,'f1-score': 0.9133663366336634,'support': 393},
    #  '1': {'precision': 0.8207792207792208, 'recall': 0.8681318681318682, 'f1-score': 0.8437917222963952, 'support': 364},
    #  '2': {'precision': 0.5,'recall': 0.13043478260869565,'f1-score': 0.20689655172413793,'support': 23},
    #  '3': {'precision': 0.875,'recall': 0.3333333333333333, 'f1-score': 0.48275862068965514,'support': 42},
    #  'accuracy': 0.8540145985401459, 'macro avg': {'precision': 0.7712339618213112,'recall': 0.5677078204459551, 'f1-score': 0.6117033078359629, 'support': 822},
    #  'weighted avg': {'precision': 0.8472654386624134, 'recall': 0.8540145985401459, 'f1-score': 0.8407878831770537, 'support': 822}}
    nb_bagging_params = {'base_estimator': MultinomialNB(alpha = 0.001, fit_prior = True), 'n_estimators': 100}
    model = EnsembleModel(ensembleClf=BaggingClassifier, params=nb_bagging_params)
    model.train(x_train_tfidf, y_train)
    y_valid_pred, y_valid_prob = model.predict(x_valid_tfidf)
    model.report_results(y_valid, y_valid_pred)















