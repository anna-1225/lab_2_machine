
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    Binarizer
)
from sklearn.naive_bayes import (
    GaussianNB,
    MultinomialNB,
    ComplementNB,
    BernoulliNB
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


RANDOM_STATE = 42

BASE_PATH = Path(__file__).resolve().parent
DATA_PATH = BASE_PATH / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS_PATH = BASE_PATH / "results"
LOGS_PATH = BASE_PATH / "logs"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    print("Размерность датасета:", df.shape)

    print("\nПервые 5 строк:")
    print(df.head())

    print("\nТипы данных:")
    print(df.dtypes)

    print("\nОписательная статистика:")
    print(df.describe())

    print("\nПропущенные значения:")
    print(df.isnull().sum())

    print("\nКоличество дубликатов:")
    print(df.duplicated().sum())

    return df


def analyze_data(df):
    class_counts = df["Churn"].value_counts()
    class_percent = df["Churn"].value_counts(normalize=True) * 100

    print("\nРаспределение классов:")
    print(pd.DataFrame({
        "Количество": class_counts,
        "Доля, %": class_percent.round(2)
    }))

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Churn",
        order=["No", "Yes"],
        hue="Churn",
        palette={"No": "steelblue", "Yes": "salmon"},
        legend=False,
        ax=ax
    )

    ax.set_title("Распределение целевой переменной Churn")
    ax.set_xlabel("Статус клиента")
    ax.set_ylabel("Количество клиентов")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Остался", "Ушёл"])

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "churn_distribution.png", dpi=300)
    plt.close()

    numeric_df = df[
        ["tenure", "MonthlyCharges", "TotalCharges"]
    ].copy()

    numeric_df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax
    )

    ax.set_title("Корреляционная матрица числовых признаков")

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "correlation_matrix.png", dpi=300)
    plt.close()

    print("\nКорреляция с Churn:")
    print(
        corr_matrix["Churn"]
        .drop("Churn")
        .sort_values(key=abs, ascending=False)
        .round(3)
    )

    contract_churn = (
        df.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    print("\nДоля оттока по типам договоров:")
    print(contract_churn.round(2))

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        x=contract_churn.index,
        y=contract_churn.values,
        color="steelblue",
        ax=ax
    )

    ax.set_title("Зависимость оттока от типа договора")
    ax.set_xlabel("Тип договора")
    ax.set_ylabel("Доля оттока, %")

    for i, value in enumerate(contract_churn.values):
        ax.text(i, value + 0.5, f"{value:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "contract_churn.png", dpi=300)
    plt.close()


def prepare_data(df):
    X = df.drop(columns=["customerID", "Churn"])

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features
    ]

    print("\nЧисловые признаки:")
    print(numeric_features)

    print("\nКатегориальные признаки:")
    print(categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nОбучающая выборка:", X_train.shape)
    print("Тестовая выборка:", X_test.shape)

    train_dist = y_train.value_counts(normalize=True) * 100
    test_dist = y_test.value_counts(normalize=True) * 100

    distribution = pd.DataFrame({
        "Обучающая выборка": train_dist,
        "Тестовая выборка": test_dist
    }).sort_index()

    distribution.index = ["Остался (0)", "Ушёл (1)"]

    print("\nРаспределение классов после разбиения:")
    print(distribution.round(2))

    ax = distribution.T.plot(
        kind="bar",
        figsize=(9, 5),
        color=["steelblue", "salmon"]
    )

    ax.set_title("Распределение классов после разбиения")
    ax.set_xlabel("Выборка")
    ax.set_ylabel("Доля клиентов, %")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "train_test_distribution.png", dpi=300)
    plt.close()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    nonnegative_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    preprocessor_nonnegative = ColumnTransformer([
        ("num", nonnegative_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_nonnegative = (
        preprocessor_nonnegative.fit_transform(X_train)
    )

    X_test_nonnegative = (
        preprocessor_nonnegative.transform(X_test)
    )

    print("\nСтандартная предобработка:")
    print("Обучающая:", X_train_processed.shape)
    print("Тестовая:", X_test_processed.shape)

    print("\nКоличество пропусков после обработки:")
    print("Обучающая:", np.isnan(X_train_processed).sum())
    print("Тестовая:", np.isnan(X_test_processed).sum())

    return (
        X_train_processed,
        X_test_processed,
        X_train_nonnegative,
        X_test_nonnegative,
        y_train,
        y_test
    )


def evaluate_model(
    name,
    model,
    X_train_data,
    X_test_data,
    y_train,
    y_test
):
    model.fit(X_train_data, y_train)

    y_pred = model.predict(X_test_data)
    y_prob = model.predict_proba(X_test_data)[:, 1]

    metrics = {
        "Модель": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    print(f"\nРезультаты модели {name}:")
    print(pd.DataFrame([metrics]).to_string(index=False))

    return metrics, model, y_pred, y_prob


def train_models(
    X_train_processed,
    X_test_processed,
    X_train_nonnegative,
    X_test_nonnegative,
    y_train,
    y_test
):
    results = []
    trained_models = {}
    predictions = {}
    probabilities = {}

    binarizer = Binarizer(threshold=0.5)

    X_train_binary = binarizer.transform(X_train_nonnegative)
    X_test_binary = binarizer.transform(X_test_nonnegative)

    models = {
        "GaussianNB": (
            GaussianNB(),
            X_train_processed,
            X_test_processed
        ),
        "MultinomialNB": (
            MultinomialNB(),
            X_train_nonnegative,
            X_test_nonnegative
        ),
        "ComplementNB": (
            ComplementNB(),
            X_train_nonnegative,
            X_test_nonnegative
        ),
        "BernoulliNB": (
            BernoulliNB(),
            X_train_binary,
            X_test_binary
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                criterion="gini",
                random_state=RANDOM_STATE
            ),
            X_train_processed,
            X_test_processed
        ),
        "LDA": (
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto"
            ),
            X_train_processed,
            X_test_processed
        ),
        "SVM": (
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=RANDOM_STATE
            ),
            X_train_processed,
            X_test_processed
        ),
        "kNN": (
            KNeighborsClassifier(
                n_neighbors=5,
                metric="euclidean",
                weights="uniform"
            ),
            X_train_processed,
            X_test_processed
        )
    }

    for name, (model, X_tr, X_te) in models.items():
        metrics, fitted_model, y_pred, y_prob = evaluate_model(
            name,
            model,
            X_tr,
            X_te,
            y_train,
            y_test
        )

        results.append(metrics)
        trained_models[name] = fitted_model
        predictions[name] = y_pred
        probabilities[name] = y_prob

    results_df = pd.DataFrame(results)

    return results_df, trained_models, predictions, probabilities


def visualize_results(results_df, predictions, y_test):
    nb_results = results_df[
        results_df["Модель"].isin([
            "GaussianNB",
            "MultinomialNB",
            "ComplementNB",
            "BernoulliNB"
        ])
    ]

    ax = nb_results.set_index("Модель")[
        ["Accuracy", "Precision", "Recall", "F1-score"]
    ].plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_title("Сравнение наивных байесовских классификаторов")
    ax.set_xlabel("Модель")
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "naive_bayes_comparison.png", dpi=300)
    plt.close()

    for name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Остался", "Ушёл"]
        )

        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False
        )

        ax.set_title(f"Матрица ошибок: {name}")

        plt.tight_layout()

        filename = name.lower().replace(" ", "_")

        plt.savefig(
            RESULTS_PATH / f"confusion_matrix_{filename}.png",
            dpi=300
        )

        plt.close()


def main():
    df = load_data()

    analyze_data(df)

    (
        X_train_processed,
        X_test_processed,
        X_train_nonnegative,
        X_test_nonnegative,
        y_train,
        y_test
    ) = prepare_data(df)

    (
        results_df,
        trained_models,
        predictions,
        probabilities
    ) = train_models(
        X_train_processed,
        X_test_processed,
        X_train_nonnegative,
        X_test_nonnegative,
        y_train,
        y_test
    )

    results_df = results_df.sort_values(
        by="F1-score",
        ascending=False
    ).reset_index(drop=True)

    print("\nИтоговая таблица классификаторов:")
    print(results_df.round(4).to_string(index=False))

    print("\nЛучшая модель по F1-score:")
    print(results_df.iloc[0]["Модель"])

    print("\nЛучшее значение F1-score:")
    print(round(results_df.iloc[0]["F1-score"], 4))

    results_df.to_csv(
        RESULTS_PATH / "baseline_metrics.csv",
        index=False
    )

    visualize_results(results_df, predictions, y_test)

    print("\nВсе результаты сохранены в папке results.")
    print("Выполнение разделов 1–4 завершено.")


if __name__ == "__main__":
    main()
