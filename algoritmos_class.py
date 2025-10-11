import pandas as pd
import numpy as np

from holdout import holdout_split
from holdout import holdout_indices

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# opcional: SHAP (se instalado)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


def build_model_pipeline(model_name, model_params=None):
    """Retorna pipeline de pré-processamento + estimator"""
    model_params = model_params or {}
    name = model_name.lower()
    if name == 'knn':
        clf = KNeighborsClassifier(**model_params)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)]) # knn precisa de escala
    elif name == 'logistic_regression' or name == 'logistic':
        clf = LogisticRegression(max_iter=1000, **model_params)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)]) # logistic precisa de escala
    elif name == 'rf' or name == 'random_forest':
        clf = RandomForestClassifier(**model_params)
        pipe = Pipeline([("clf", clf)]) # não precisa de escala
    elif name == 'svm':
        clf = SVC(probability=True, **model_params)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)]) # SVM precisa de escala
    elif name == 'arvore_decisao' or name == 'decision_tree':
        clf = DecisionTreeClassifier(**model_params)
        pipe = Pipeline([("clf", clf)]) # não precisa de escala
    elif name == 'naive_bayes' or name == 'naive' or name == 'nb':
        clf = GaussianNB(**model_params)
        pipe = Pipeline([("clf", clf)]) # não precisa de escala
    elif name == 'ensemble':
        clf1 = LogisticRegression(max_iter=1000)
        clf2 = RandomForestClassifier(n_estimators=100)
        clf = VotingClassifier(estimators=[("lr", clf1), ("rf", clf2)], voting="soft")
        pipe = Pipeline([("clf", clf)])  # não precisa de escala
    elif name == 'mlp' or name == 'neural_network' or name == 'rede_neural':
        clf = MLPClassifier(max_iter=1000, **model_params)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)]) # MLP precisa de escala
    elif name == 'bagging': 
        base_clf = DecisionTreeClassifier()
        clf = BaggingClassifier(base_estimator=base_clf, **model_params)
        pipe = Pipeline([("clf", clf)]) # não precisa de escala
    else:
        raise NotImplementedError(f"Modelo {model_name} não implementado")
    return pipe


def train_and_test_classifier(
    model_name,
    csv_path,
    feature_columns,
    target_column,
    train_size=0.65,
    random_state=None,
    model_params=None,
    verbose=False,
    importance_method='auto',   # 'auto', 'permutation', 'coef', 'shap'
    permutation_n_repeats=10
):
    """
    Treina e testa um classificador (holdout) e calcula importância das features.
    Retorna dict com o modelo, dados de treino/teste, métricas e feature_importances (DataFrame).

    importance_method:
      - 'auto' : usa coef_ para modelos lineares, permutation para os outros;
      - 'permutation' : força permutation importance;
      - 'coef' : força coef (falhará se o modelo não tiver coef_);
      - 'shap' : tenta usar SHAP (requer pacote shap).
    """

    # 1) Leitura
    df = pd.read_csv(csv_path)

    # 2) Seleção de colunas
    if feature_columns == '*':
        X = df.drop(columns=[target_column]).copy()
    else:
        X = df[list(feature_columns)].copy()
    y = df[target_column].copy()

    # 3) One-hot em categóricas
    X = pd.get_dummies(X, drop_first=True)

    # salvar nomes das features para usar depois
    feature_names = X.columns.tolist()

    # 4) Arrays
    X_arr = X.values
    y_arr = y.values

    # 5) Holdout indices
    n = len(X_arr)
    if n == 0:
        raise ValueError("O dataset está vazio após o pré-processamento.")

    train_idx, test_idx = holdout_indices(n, train_size=train_size, random_state=random_state)

    X_train = X_arr[train_idx]
    X_test = X_arr[test_idx]
    y_train = y_arr[train_idx]
    y_test = y_arr[test_idx]

    # 6) Pipeline e treino
    pipeline = build_model_pipeline(model_name, model_params=model_params)
    pipeline.fit(X_train, y_train)

    # 7) Previsões e métricas
    y_pred = pipeline.predict(X_test)

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['classification_report'] = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    labels = np.unique(np.concatenate((y_test, y_pred)))
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    metrics['confusion_matrix'] = cm_df

    # 8) Importância das features
    feature_importances_df = None
    method_used = None

    # decidir método
    model_lower = model_name.lower()
    if importance_method == 'auto':
        if model_lower in ['logistic_regression', 'logistic']:
            chosen = 'coef'
        else:
            chosen = 'permutation'
    else:
        chosen = importance_method

    # 8a) coeficientes (modelos lineares)
    if chosen == 'coef':
        # tenta obter coef_ do estimador dentro do pipeline
        try:
            estimator = pipeline.named_steps['clf']
            coefs = estimator.coef_
            # se multiclass, coefs.shape = (n_classes, n_features) -> resumimos como média absoluta
            if coefs.ndim == 1:
                imp = np.abs(coefs)
            else:
                imp = np.mean(np.abs(coefs), axis=0)
            feature_importances_df = pd.DataFrame({
                'feature': feature_names,
                'importance': imp
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            method_used = 'coef'
        except Exception as e:
            # não há coef_; fallback para permutation
            if verbose:
                print("Erro ao extrair coef_: ", e, " -> fallback para permutation importance")
            chosen = 'permutation'

    # 8b) permutation importance
    if chosen == 'permutation':
        # usa sklearn.inspection.permutation_importance no pipeline (ele aceita pipeline)
        try:
            perm = permutation_importance(pipeline, X_test, y_test, n_repeats=permutation_n_repeats, random_state=random_state, n_jobs=-1)
            imp_means = perm.importances_mean
            imp_stds = perm.importances_std
            feature_importances_df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': imp_means,
                'importance_std': imp_stds
            }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
            method_used = 'permutation'
        except Exception as e:
            if verbose:
                print("Erro em permutation_importance:", e)
            feature_importances_df = None

    # 8c) SHAP (opcional)
    if chosen == 'shap':
        if not _HAS_SHAP:
            if verbose:
                print("SHAP não encontrado (pip install shap) — fallback para permutation importance")
            # fallback
            chosen = 'permutation'
            # we'll run permutation after (it will run below)
            if feature_importances_df is None:
                # force permutation branch
                try:
                    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=permutation_n_repeats, random_state=random_state, n_jobs=-1)
                    imp_means = perm.importances_mean
                    imp_stds = perm.importances_std
                    feature_importances_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance_mean': imp_means,
                        'importance_std': imp_stds
                    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
                    method_used = 'permutation'
                except Exception:
                    feature_importances_df = None
        else:
            # tenta usar SHAP Explainer
            try:
                # Para pipelines, podemos passar o estimador e os dados transformados
                estimator = pipeline.named_steps['clf']
                # precisamos dos dados transformados para o explainer, então aplicamos o pipeline até antes do clf:
                # se houver scaler, usamos pipeline[:-1].transform
                try:
                    X_train_transformed = pipeline[:-1].transform(X_train)
                    X_test_transformed = pipeline[:-1].transform(X_test)
                except Exception:
                    # fallback: passa X_train/X_test crus (alguns explainers aceitam)
                    X_train_transformed = X_train
                    X_test_transformed = X_test

                explainer = shap.Explainer(estimator, X_train_transformed)
                shap_values = explainer(X_test_transformed)
                # shap_values.values shape: (n_samples, n_features) or (n_classes, n_samples, n_features)
                # usamos mean absolute SHAP value por feature
                shap_abs_mean = np.mean(np.abs(shap_values.values), axis=0)
                feature_importances_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_abs_mean': shap_abs_mean
                }).sort_values('shap_abs_mean', ascending=False).reset_index(drop=True)
                method_used = 'shap'
            except Exception as e:
                if verbose:
                    print("Erro ao rodar SHAP:", e, "-> fallback para permutation")
                # fallback permutation
                try:
                    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=permutation_n_repeats, random_state=random_state, n_jobs=-1)
                    imp_means = perm.importances_mean
                    imp_stds = perm.importances_std
                    feature_importances_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance_mean': imp_means,
                        'importance_std': imp_stds
                    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
                    method_used = 'permutation'
                except Exception:
                    feature_importances_df = None

    # 9) Print verbose
    if verbose:
        print("Modelo:", model_name)
        print("Tamanho treino/teste:", len(X_train), "/", len(X_test))
        print(f"Acurácia: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
        print("\nClassification Report:\n", metrics['classification_report'])
        print("\nConfusion Matrix:\n", metrics['confusion_matrix'])
        print("\nFeature importance method used:", method_used)
        if feature_importances_df is not None:
            print(feature_importances_df.head(20))
        print("-" * 40)

    # 10) retorno
    return {
        'model': pipeline,
        'X-train': X_train,
        'X-test': X_test,
        'y-train': y_train,
        'y-test': y_test,
        'metrics': metrics,
        'feature_names': feature_names,
        'feature_importances': feature_importances_df,
        'importance_method': method_used
    }
