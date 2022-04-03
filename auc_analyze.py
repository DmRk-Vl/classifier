import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
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


# auc анализ для логистической регрессии
model = LogisticRegression(random_state=1, penalty='l2', solver='lbfgs', C = 0.001)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)

probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)

pyplot.plot([0, 1], [0, 1], linestyle='--')

pyplot.plot(fpr, tpr, marker='.')

pyplot.show()

# auc анализ для RandomForest
model = RandomForestClassifier(criterion='gini', random_state=1, max_depth = 2, max_features = 2)
model.fit(X_train, y_train)
# predict probabilities
probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()

