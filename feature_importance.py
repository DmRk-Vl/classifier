import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Нормализуем данные
scaler = MinMaxScaler()
le = LabelEncoder()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Визуализация важности признаков для RandomForest (первоначальная выборка)
model = RandomForestClassifier(criterion='gini', random_state=1, max_depth = 2, max_features = 2)
model.fit(X_train, y_train)
df_feature_importance = pd.DataFrame(model.feature_importances_, df.drop('Cover_Type', axis=1).columns, columns=['feature importance']).sort_values('feature importance', ascending=False)
df_feature_importance.plot(kind='bar',figsize = (15,10));

model = LogisticRegression(random_state=1, penalty='l2', solver='lbfgs', C = 0.001)
model.fit(X_train, y_train)

coeffs = []
for i in model:
    coeffs = i

df_feature_importance = pd.DataFrame(coeffs, df.drop('Cover_Type', axis=1).columns, columns=['feature importance']).sort_values('feature importance', ascending=False)
df_feature_importance.plot(kind='bar',figsize = (15,10));


