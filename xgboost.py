import xgboost
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def preprocess(txt):
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





if __name__ == '__main__' :
    x_train, y_train = utils.load_data_pd('RT_train.csv', 'text', 'labels')
    x_valid, y_valid = utils.load_data_pd('RT_test.csv', 'text', 'labels')


    x_train_cleaned = x_train.map(preprocess)
    x_valid_cleaned = x_valid.map(preprocess)


    vectorizer = TfidfVectorizer()
    vectorizer.fit(x_train_cleaned)
    x_train_tfidf = vectorizer.transform(x_train_cleaned)
    x_valid_tfidf = vectorizer.transform(x_valid_cleaned)
    clf = XGBClassifier(colsample_bytree=0.6, subsample=0.7)
    clf.fit(x_train_tfidf, y_train)
    y_valid_pred = clf.predict(x_valid_tfidf)
    clf.score(x_valid_tfidf, y_valid)

    cm = confusion_matrix(y_valid , y_valid_pred)
    accuracy_score(y_valid , y_valid_pred)
    roc_auc_score(y_valid , clf.predict_proba(x_valid_tfidf), multi_class = 'ovr' )
    classification_report(y_valid, y_valid_pred, output_dict = True)

    #tuning
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    clf = XGBClassifier(learning_rate = 0.02, n_estimators = 600)
    grid = GridSearchCV(clf, params)
    grid.fit(x_train_tfidf, y_train)
    print(grid.best_params_)
    print(grid.best_estimator_)

    # {'C': 100, 'degree': 2, 'gamma': 0.01, 'kernel': 'rbf'}
    # SVC(C=100, degree=2, gamma=0.01)
    clf_candidate = XGBClassifier()
    y_valid_pred = clf_candidate(x_valid_tfidf)
    classification_report(y_valid, y_valid_pred, output_dict=True)