import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
# Task 1:


def plot_relation_between_age_and_disease(df):
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                  'fbs', 'restecg', 'thalach', 'exang',
                  'oldpeak', 'slope', 'ca', 'thal', 'target']

    df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())
    df['age'] = df['age'].astype(int)

    # distribution of target vs age
    sns.set_context("paper", font_scale=1, rc={
                    "font.size": 3, "axes_titlesize": 15, "axes.labelsize": 10})

    ax = sns.catplot(kind='count', x='age', hue='target',
                     data=df, order=df['age'].sort_values().unique())

    ax.ax.set_xticks(np.arange(0, 80, 5))

    plt.title('Distribution of target vs age')
    plt.show()

# Task 2:


def plot_relation_between_age_gender_and_disease(df):
    # Ensure columns are set properly
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                  'fbs', 'restecg', 'thalach', 'exang',
                  'oldpeak', 'slope', 'ca', 'thal', 'target']

    df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())
    df['age'] = df['age'].astype(int)

    # distribution of target vs age vs sex
    sns.catplot(kind='bar', data=df, y='age', x='sex', hue='target')
    plt.title('Distribution of target vs age vs sex with the target class')
    plt.show()

# Task 3:


def knn_classifier(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # define the model
    classifier = KNeighborsClassifier(
        n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    accuracy_for_train = np.round(
        (cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
    accuracy_for_test = np.round(
        (cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
    print('Accuracy for training set for KNeighborsClassifier = {}'.format(
        accuracy_for_train))
    print('Accuracy for test set for KNeighborsClassifier = {}'.format(
        accuracy_for_test))

# Task 4:


def svm_classifier(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    classifier = SVC(kernel='rbf', random_state=42)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    accuracy_for_train = np.round(
        (cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
    accuracy_for_test = np.round(
        (cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
    print('Accuracy for training set for SVC = {}'.format(accuracy_for_train))
    print('Accuracy for test set for SVC = {}'.format(accuracy_for_test))

# Task 5:


def naive_bayes_classifier(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    accuracy_for_train = np.round(
        (cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
    accuracy_for_test = np.round(
        (cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
    print('Accuracy for training set for GaussianNB = {}'.format(accuracy_for_train))
    print('Accuracy for test set for GaussianNB = {}'.format(accuracy_for_test))

# Task 6:


def decision_tree_classifier(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    classifier = DecisionTreeClassifier(
        criterion='gini', max_depth=10, min_samples_split=2, random_state=42)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    accuracy_for_train = np.round(
        (cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
    accuracy_for_test = np.round(
        (cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
    print('Accuracy for training set for DecisionTreeClassifier = {}'.format(
        accuracy_for_train))
    print('Accuracy for test set for DecisionTreeClassifier = {}'.format(
        accuracy_for_test))

# Task 7:


def random_forest_classifier(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    classifier = RandomForestClassifier(
        criterion='gini', max_depth=10, min_samples_split=2, n_estimators=10, random_state=42)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    accuracy_for_train = np.round(
        (cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
    accuracy_for_test = np.round(
        (cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
    print('Accuracy for training set for RandomForestClassifier = {}'.format(
        accuracy_for_train))
    print('Accuracy for test set for RandomForestClassifier = {}'.format(
        accuracy_for_test))

# Task 8:


def ada_boost_classifier(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # define the model
    classifier = AdaBoostClassifier(
        n_estimators=50, learning_rate=1.0, random_state=42)

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    accuracy_for_train = np.round(
        (cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
    accuracy_for_test = np.round(
        (cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
    print('Accuracy for training set for AdaBoostClassifier = {}'.format(
        accuracy_for_train))
    print('Accuracy for test set for AdaBoostClassifier = {}'.format(accuracy_for_test))

# Task 9:


def gradient_boosting_classifier(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # define the model

    classifier = GradientBoostingClassifier(
        learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, max_depth=3, random_state=42)

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    accuracy_for_train = np.round(
        (cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
    accuracy_for_test = np.round(
        (cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
    print('Accuracy for training set for GradientBoostingClassifier = {}'.format(
        accuracy_for_train))
    print('Accuracy for test set for GradientBoostingClassifier = {}'.format(
        accuracy_for_test))


# Task 10:
def xgboost_classifier(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    xg = XGBClassifier(objective="binary:logistic",
                       random_state=42, n_estimators=100)
    xg.fit(X_train, y_train)
    y_pred = xg.predict(X_test)
    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = xg.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    accuracy_for_train = np.round(
        (cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
    accuracy_for_test = np.round(
        (cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
    print('Accuracy for training set for XGBClassifier = {}'.format(accuracy_for_train))
    print('Accuracy for test set for XGBClassifier = {}'.format(accuracy_for_test))

# Task 11:


def stacking_classifier(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    dtc = DecisionTreeClassifier(random_state=42)
    rfc = RandomForestClassifier(random_state=42)
    knn = KNeighborsClassifier()
    xgb = XGBClassifier(XGBClassifier)
    gc = GradientBoostingClassifier(random_state=42)
    svc = SVC(kernel='rbf', random_state=42)
    ad = AdaBoostClassifier(random_state=42)

    clf = [('dtc', dtc), ('rfc', rfc), ('knn', knn), ('gc', gc),
           ('ad', ad), ('svc', svc)]  # list of (str, estimator)

    xg = XGBClassifier()
    classifier = StackingClassifier(estimators=clf, final_estimator=xg)
    classifier.fit(X_train, y_train)

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    print()
    accuracy_for_train = np.round(
        (cm_train[0][0] + cm_train[1][1])/len(y_train), 2)
    accuracy_for_test = np.round(
        (cm_test[0][0] + cm_test[1][1])/len(y_test), 2)
    print('Accuracy for training set for StackingClassifier = {}'.format(
        accuracy_for_train))
    print('Accuracy for test set for StackingClassifier = {}'.format(accuracy_for_test))


if __name__ == '__main__':
    # Load the data
    df = pd.read_csv(
        'module3-project-heart-disease-prediction/Heart-Disease-Prediction/cleveland.csv', header=None)

    # Task 1:
    # plot_relation_between_age_and_disease(df)
    # Task 2:
    # plot_relation_between_age_gender_and_disease(df)
    # X = df.iloc[:, :-1].values
    # y = df.iloc[:, -1].values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # sc = ss()
    # X_train_after = sc.fit_transform(X_train)
    # X_test_after = sc.transform(X_test)
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                  'fbs', 'restecg', 'thalach', 'exang',
                  'oldpeak', 'slope', 'ca', 'thal', 'target']
    df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())
    stacking_classifier(df)
