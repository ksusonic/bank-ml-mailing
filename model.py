import pandas as pd

from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

pd.options.mode.chained_assignment = None


def open_data(path="dataset.csv"):
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame):
    """ runs preprocessing on dataset """

    # разделение данных
    X, y = df.drop(['AGREEMENT_RK', 'TARGET'], axis=1), df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # масштабирование
    ss = MinMaxScaler()
    ss.fit(X_train)

    X_train = pd.DataFrame(ss.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(ss.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test, ss


def fit_and_save_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
                       test_model=True,
                       metric='accuracy'):
    """ fits logistic regression model """
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    if test_model:
        preds = model.predict(X_test)
        if metric == 'accuracy':
            score = accuracy_score(y_test, preds)
        elif metric == 'recall':
            score = recall_score(y_test, preds)
        elif metric == 'precision':
            score = precision_score(y_test, preds)
        print(f'{metric.title()}: {round(score, 3)}')

    dump_model(model)
    save_importances(model, X_train.columns)


def dump_model(model, path="model_weights.mw"):
    """ saves model as pickle file """

    with open(path, "wb") as file:
        dump(model, file)

    print(f'Model was saved to {path}')


def save_importances(model, feature_names, path='importances.csv'):
    """ saves sorted feature weights as df """

    importances = pd.DataFrame({'Признак': feature_names, 'Вес': model.coef_[0]})
    importances.sort_values(by='Вес', key=abs, ascending=False, inplace=True)

    importances.to_csv(path, index=False)
    print(f'Importances were saved to {path}')


def load_model(path="model_weights.mw"):
    """ load model from saved weights """

    with open(path, "rb") as file:
        model = load(file)

    return model


def get_importances(top_n=5, importance='most', path='importances.csv'):
    """ returns top n most important or least important weights """

    importances = pd.read_csv(path, encoding='utf-8')
    if importance == 'most':
        return importances.head(top_n)
    else:
        return importances.tail(top_n).iloc[::-1]


def predict_on_input(df: pd.DataFrame):
    """ loads model and returns prediction and probability """

    model = load_model()
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)

    return pred, proba


if __name__ == "__main__":
    df = open_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    fit_and_save_model(X_train, X_test, y_train, y_test)
