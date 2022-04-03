import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('/home/dmitrii/covtype.data')

X = df.drop('Cover_Type', axis=1).values
y = df['Cover_Type'].values

positive_class = 5
negative_class = [1,2,3,4,6,7]
for i in range(len(y)):
    if y[i] == positive_class:
        y[i] = 1
    else:
        y[i] = 0

# Разделим данные на учебную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Нормализуем данные
scaler = MinMaxScaler()
le = LabelEncoder()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# GridSearch для RandomForest
pipelineRFC = make_pipeline(StandardScaler(), RandomForestClassifier(criterion='gini', random_state=1))

param_grid_rfc = [{
    'randomforestclassifier__max_depth':[2, 3, 4],
    'randomforestclassifier__max_features':[2, 3, 4, 5, 6]
}]

gsRFC = GridSearchCV(estimator=pipelineRFC,
                     param_grid = param_grid_rfc,
                     scoring='accuracy',
                     cv=10,
                     refit=True,
                     n_jobs=1)

gsRFC = gsRFC.fit(X_train, y_train)

print(gsRFC.best_score_)

print(gsRFC.best_params_)

clfRFC = gsRFC.best_estimator_
print('Test accuracy Random Forest: %.3f' % clfRFC.score(X_test, y_test))

# GridSearch для LogisticRegression
pipelineLR = make_pipeline(StandardScaler(), LogisticRegression(random_state=1, penalty='l2', solver='lbfgs'))

param_grid_lr = [{
    'logisticregression__C': [0.001, 0.01, 1.0, 10.0],
}]

gsLR = GridSearchCV(estimator=pipelineLR,
                     param_grid = param_grid_lr,
                     scoring='accuracy',
                     cv=10,
                     refit=True,
                     n_jobs=1)

gsLR = gsLR.fit(X_train, y_train)

print(gsLR.best_score_)

print(gsLR.best_params_)

clfLR = gsLR.best_estimator_
print('Test accuracy LogisticRegression: %.3f' % clfLR.score(X_test, y_test))

# GridSearch для SVC
pipelineSVC = make_pipeline(StandardScaler(), SVC(random_state=1))

param_grid_svc = [{
                    'svc__C': [0.001, 0.01, 1000],
                    'svc__kernel': ['linear'],
                    'svc__gamma': [0.001, 0.01, 0.1]
                  }]

gsSVC = GridSearchCV(estimator=pipelineSVC,
                     param_grid = param_grid_svc,
                     scoring='accuracy',
                     cv=10,
                     refit=True,
                     n_jobs=1)

gsSVC.fit(X_train[:2000], y_train[:2000])

print(gsSVC.best_score_)

print(gsSVC.best_params_)

print('Test accuracy SVC: %.3f' % gsSVC.score(X_test, y_test))

clfSVC = gsSVC.best_estimator_
print('Test accuracy SVC: %.3f' % clfSVC.score(X_test, y_test))
