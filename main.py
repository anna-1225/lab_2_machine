
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
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
    ConfusionMatrixDisplay,
    roc_curve
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



def get_raw_split(df):
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    return train_test_split(
        X, y, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessors(X):
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        col for col in X.columns if col not in numeric_features
    ]
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ))
    ])
    standard = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", clone(categorical_pipeline), categorical_features)
    ])
    nonnegative = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ]), numeric_features),
        ("cat", clone(categorical_pipeline), categorical_features)
    ])
    return standard, nonnegative


def tune_models(X_train, X_test, y_train, y_test):
    standard, nonnegative = build_preprocessors(X_train)
    specifications = {
        "GaussianNB": (
            Pipeline([("preprocessor", clone(standard)), ("model", GaussianNB())]),
            {"model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]}
        ),
        "MultinomialNB": (
            Pipeline([("preprocessor", clone(nonnegative)), ("model", MultinomialNB())]),
            {"model__alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}
        ),
        "ComplementNB": (
            Pipeline([("preprocessor", clone(nonnegative)), ("model", ComplementNB())]),
            {"model__alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}
        ),
        "BernoulliNB": (
            Pipeline([
                ("preprocessor", clone(nonnegative)),
                ("binarizer", Binarizer(threshold=0.5)),
                ("model", BernoulliNB())
            ]),
            {"model__alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}
        ),
        "Decision Tree": (
            Pipeline([
                ("preprocessor", clone(standard)),
                ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
            ]),
            {
                "model__max_depth": [3, 5, 7, 10, None],
                "model__min_samples_leaf": [1, 5, 10],
                "model__min_samples_split": [2, 10]
            }
        ),
        "LDA": (
            Pipeline([
                ("preprocessor", clone(standard)),
                ("model", LinearDiscriminantAnalysis(solver="lsqr"))
            ]),
            {"model__shrinkage": ["auto", 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
        ),
        "SVM": (
            Pipeline([
                ("preprocessor", clone(standard)),
                ("model", SVC(random_state=RANDOM_STATE))
            ]),
            [
                {"model__kernel": ["linear"], "model__C": [0.1, 1, 10]},
                {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10],
                 "model__gamma": ["scale", 0.01, 0.1]}
            ]
        ),
        "kNN": (
            Pipeline([
                ("preprocessor", clone(standard)),
                ("model", KNeighborsClassifier())
            ]),
            {
                "model__n_neighbors": [3, 5, 7, 11, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"]
            }
        )
    }
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    metrics_list = []
    fitted_models = {}
    scores = {}
    for name, (pipeline, grid) in specifications.items():
        print(f"\nНастройка {name}")
        search = GridSearchCV(
            pipeline, grid, scoring="f1", cv=folds,
            n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        prediction = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X_test)[:, 1]
        else:
            probability = model.decision_function(X_test)
        result = {
            "Модель": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1-score": f1_score(y_test, prediction, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probability)
        }
        metrics_list.append(result)
        fitted_models[name] = model
        scores[name] = probability
        print("Лучшие параметры:", search.best_params_)
        print("F1 на кросс-валидации:", round(search.best_score_, 4))
        print("F1 на тесте:", round(result["F1-score"], 4))
        cv_table = pd.DataFrame(search.cv_results_)
        cv_table.to_csv(
            RESULTS_PATH / f"grid_{name.lower().replace(' ', '_')}.csv",
            index=False
        )
    tuned_df = pd.DataFrame(metrics_list)
    tuned_df.to_csv(RESULTS_PATH / "tuned_metrics.csv", index=False)
    return tuned_df, fitted_models, scores


def plot_tuning_comparison(baseline_df, tuned_df):
    before = baseline_df[["Модель", "F1-score"]].rename(
        columns={"F1-score": "До настройки"}
    )
    after = tuned_df[["Модель", "F1-score"]].rename(
        columns={"F1-score": "После настройки"}
    )
    summary = before.merge(after, on="Модель")
    summary.to_csv(RESULTS_PATH / "f1_before_after.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 6))
    summary.set_index("Модель")[["До настройки", "После настройки"]].plot.bar(ax=ax)
    ax.set_title("Сравнение F1-score до и после настройки")
    ax.set_xlabel("Классификатор")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "f1_before_after.png", dpi=300)
    plt.close(fig)


def create_neural_network(input_dim, layers=(64, 32), activation="relu", learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    for neurons in layers:
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def plot_history(history, filename):
    hist = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, ax in [("accuracy", axes[0]), ("loss", axes[1])]:
        ax.plot(hist[key], label="Обучение")
        ax.plot(hist[f"val_{key}"], label="Валидация")
        ax.set_title(key.upper())
        ax.set_xlabel("Эпоха")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / filename, dpi=300)
    plt.close(fig)


def train_neural_network(X_train, X_test, y_train, y_test):
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    standard, _ = build_preprocessors(X_train)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2,
        stratify=y_train, random_state=RANDOM_STATE
    )
    local_preprocessor = clone(standard)
    X_fit_array = local_preprocessor.fit_transform(X_fit).astype("float32")
    X_val_array = local_preprocessor.transform(X_val).astype("float32")
    baseline = create_neural_network(X_fit_array.shape[1])
    baseline_history = baseline.fit(
        X_fit_array, y_fit.to_numpy().astype("float32"),
        validation_data=(X_val_array, y_val.to_numpy().astype("float32")),
        epochs=20, batch_size=32,
        callbacks=[tf.keras.callbacks.TensorBoard(
            log_dir=str(LOGS_PATH / "baseline"), histogram_freq=1
        )], verbose=0
    )
    plot_history(baseline_history, "neural_network_baseline.png")
    experiments = [
        ("Базовая", (64, 32), "relu", 20, 32),
        ("10 эпох", (64, 32), "relu", 10, 32),
        ("35 эпох", (64, 32), "relu", 35, 32),
        ("Batch 16", (64, 32), "relu", 20, 16),
        ("Batch 64", (64, 32), "relu", 20, 64),
        ("1 слой", (64,), "relu", 20, 32),
        ("3 слоя", (128, 64, 32), "relu", 20, 32),
        ("Меньше нейронов", (32, 16), "relu", 20, 32),
        ("Больше нейронов", (128, 64), "relu", 20, 32),
        ("Tanh", (64, 32), "tanh", 20, 32)
    ]
    rows = []
    histories = {}
    for i, (name, layers, activation, epochs, batch_size) in enumerate(experiments):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_STATE)
        model = create_neural_network(
            X_fit_array.shape[1], layers=layers, activation=activation
        )
        history = model.fit(
            X_fit_array, y_fit.to_numpy().astype("float32"),
            validation_data=(X_val_array, y_val.to_numpy().astype("float32")),
            epochs=epochs, batch_size=batch_size,
            callbacks=[tf.keras.callbacks.TensorBoard(
                log_dir=str(LOGS_PATH / f"experiment_{i:02d}")
            )], verbose=0
        )
        probability = model.predict(X_val_array, verbose=0).ravel()
        prediction = (probability >= 0.5).astype(int)
        row = {
            "Эксперимент": name,
            "Слои": str(layers), "Активация": activation,
            "Эпохи": epochs, "Batch": batch_size,
            "Accuracy": accuracy_score(y_val, prediction),
            "Precision": precision_score(y_val, prediction, zero_division=0),
            "Recall": recall_score(y_val, prediction, zero_division=0),
            "F1-score": f1_score(y_val, prediction, zero_division=0),
            "ROC-AUC": roc_auc_score(y_val, probability),
            "Val Loss": history.history["val_loss"][-1]
        }
        rows.append(row)
        histories[name] = history.history
        print(f"Нейросеть: {name}: F1 = {row['F1-score']:.4f}")
    experiments_df = pd.DataFrame(rows)
    experiments_df.to_csv(
        RESULTS_PATH / "neural_network_experiments.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=experiments_df.sort_values("F1-score", ascending=False),
        x="Эксперимент", y="F1-score", color="steelblue", ax=ax
    )
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "neural_network_tuning.png", dpi=300)
    plt.close(fig)
    best_row = experiments_df.loc[experiments_df["F1-score"].idxmax()]
    best = next(item for item in experiments if item[0] == best_row["Эксперимент"])
    plot_history(histories[best[0]], "best_neural_network_history.png")
    print("Лучшая нейросеть:", best[0], "F1 валидации:", round(best_row["F1-score"], 4))
    final_preprocessor = clone(standard)
    X_train_array = final_preprocessor.fit_transform(X_train).astype("float32")
    X_test_array = final_preprocessor.transform(X_test).astype("float32")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    final_model = create_neural_network(
        X_train_array.shape[1], layers=best[1], activation=best[2]
    )
    final_model.fit(
        X_train_array, y_train.to_numpy().astype("float32"),
        epochs=best[3], batch_size=best[4],
        callbacks=[tf.keras.callbacks.TensorBoard(
            log_dir=str(LOGS_PATH / "final_model")
        )], verbose=0
    )
    scores = final_model.predict(X_test_array, verbose=0).ravel()
    predictions = (scores >= 0.5).astype(int)
    metrics = {
        "Модель": "TensorFlow NN",
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, scores)
    }
    pd.DataFrame([metrics]).to_csv(
        RESULTS_PATH / "neural_network_metrics.csv", index=False
    )
    final_model.save(RESULTS_PATH / "telco_neural_network.keras")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, predictions),
        display_labels=["Остался", "Ушёл"]
    ).plot(ax=axes[0], cmap="Blues", colorbar=False)
    fpr, tpr, _ = roc_curve(y_test, scores)
    axes[1].plot(fpr, tpr, label=f"ROC-AUC = {metrics['ROC-AUC']:.4f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "neural_network_evaluation.png", dpi=300)
    plt.close(fig)
    print("Итоговые метрики нейросети:")
    print(pd.DataFrame([metrics]).round(4).to_string(index=False))
    return metrics, scores


def final_comparison(tuned_df, neural_metrics, classical_scores, neural_scores, y_test):
    full_df = pd.concat(
        [tuned_df, pd.DataFrame([neural_metrics])], ignore_index=True
    ).sort_values("F1-score", ascending=False)
    full_df.to_csv(RESULTS_PATH / "final_models_comparison.csv", index=False)
    print("\nИтоговое сравнение девяти моделей:")
    print(full_df.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=full_df, x="Модель", y="F1-score", color="steelblue", ax=ax)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "comparison_f1.png", dpi=300)
    plt.close(fig)
    long_df = full_df.melt(
        id_vars="Модель",
        value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        var_name="Метрика", value_name="Значение"
    )
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(data=long_df, x="Модель", y="Значение", hue="Метрика", ax=ax)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "comparison_all_metrics.png", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 8))
    combined_scores = dict(classical_scores)
    combined_scores["TensorFlow NN"] = neural_scores
    for name, scores in combined_scores.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "comparison_roc_curves.png", dpi=300)
    plt.close(fig)
    rows = []
    for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
        maximum = full_df[metric].max()
        names = full_df.loc[
            np.isclose(full_df[metric], maximum), "Модель"
        ].tolist()
        rows.append({
            "Метрика": metric,
            "Лучшая модель": ", ".join(names),
            "Значение": maximum
        })
    pd.DataFrame(rows).to_csv(
        RESULTS_PATH / "best_models_by_metric.csv", index=False
    )
    print("\nЛучшая модель по F1-score:", full_df.iloc[0]["Модель"])


def main():
    df = load_data()
    analyze_data(df)
    (
        X_train_processed, X_test_processed,
        X_train_nonnegative, X_test_nonnegative,
        y_train, y_test
    ) = prepare_data(df)
    (
        results_df, trained_models,
        predictions, probabilities
    ) = train_models(
        X_train_processed, X_test_processed,
        X_train_nonnegative, X_test_nonnegative,
        y_train, y_test
    )
    results_df = results_df.sort_values(
        by="F1-score", ascending=False
    ).reset_index(drop=True)
    print("\nИтоговая таблица базовых классификаторов:")
    print(results_df.round(4).to_string(index=False))
    results_df.to_csv(RESULTS_PATH / "baseline_metrics.csv", index=False)
    visualize_results(results_df, predictions, y_test)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = get_raw_split(df)
    tuned_df, tuned_models, tuned_scores = tune_models(
        X_train_raw, X_test_raw, y_train_raw, y_test_raw
    )
    plot_tuning_comparison(results_df, tuned_df)
    neural_metrics, neural_scores = train_neural_network(
        X_train_raw, X_test_raw, y_train_raw, y_test_raw
    )
    final_comparison(
        tuned_df, neural_metrics, tuned_scores, neural_scores, y_test_raw
    )
    print("\nРазделы 1–8 выполнены. Результаты сохранены в папке results.")


if __name__ == "__main__":
    main()
