#algoritmos_class.py
import pandas as pd
import numpy as np

from time import perf_counter

from sklearn.gaussian_process import GaussianProcessRegressor

from holdout import holdout_indices

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, VotingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, StackingRegressor


from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# imports para o BlendingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


class BlendingClassifier(BaseEstimator, ClassifierMixin):
    """
    Blending ensemble.
    - estimators: list of (name, estimator) tuples
    - final_estimator: estimator to combine as meta
    - holdout_size: fraction of the training set used as holdout for meta (e.g. 0.2)
    - use_proba: 'auto' (tries predict_proba, else predict), True (force predict_proba), False (use predict)
    - random_state: for reproducibility (passed to train_test_split)
    - retrain_base_on_full: if True (default), re-treina os base learners no conjunto train+holdout antes do deploy
    """
    def __init__(self, estimators, final_estimator, holdout_size=0.2, use_proba='auto', random_state=None, retrain_base_on_full=True):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.holdout_size = holdout_size
        self.use_proba = use_proba
        self.random_state = random_state
        self.retrain_base_on_full = retrain_base_on_full

    def _get_pred_method(self, est):
        """Decide se usamos predict_proba ou predict para um estimator."""
        if self.use_proba == 'auto':
            return 'predict_proba' if hasattr(est, "predict_proba") else 'predict'
        return 'predict_proba' if self.use_proba else 'predict'

    def fit(self, X, y):
        if self.holdout_size is None or self.holdout_size <= 0 or self.holdout_size >= 1:
            raise ValueError("holdout_size must be between 0 and 1 for blending.")

        # stratify se for classificação (y não nulo)
        stratify = y if y is not None else None
        X_train, X_hold, y_train, y_hold = train_test_split(
            X, y, test_size=self.holdout_size, stratify=stratify, random_state=self.random_state
        )

        # 2) clona e treina bases em X_train
        self.base_learners_ = []
        meta_features_hold = []

        for name, est in self.estimators:
            est_clone = clone(est)
            est_clone.fit(X_train, y_train)
            self.base_learners_.append((name, est_clone))

            method = self._get_pred_method(est_clone)
            if method == 'predict_proba':
                preds = est_clone.predict_proba(X_hold)  # (n_samples, n_classes)
            else:
                preds = est_clone.predict(X_hold).reshape(-1, 1)  # (n_samples, 1)

            meta_features_hold.append(preds)

        # concatena features meta
        meta_X_hold = np.hstack(meta_features_hold)

        # treina meta-estimator sobre meta_X_hold
        self.meta_estimator_ = clone(self.final_estimator)
        self.meta_estimator_.fit(meta_X_hold, y_hold)

        # opcional: re-treina bases no conjunto inteiro (X_train + X_hold) para deploy
        if self.retrain_base_on_full:
            X_full = np.vstack([X_train, X_hold])
            y_full = np.concatenate([y_train, y_hold])
            for idx, (name, est) in enumerate(self.base_learners_):
                est.fit(X_full, y_full)
                self.base_learners_[idx] = (name, est)

        return self

    def _build_meta_features_from_bases(self, X):
        meta_feats = []
        for name, est in self.base_learners_:
            method = self._get_pred_method(est)
            if method == 'predict_proba':
                preds = est.predict_proba(X)
            else:
                preds = est.predict(X).reshape(-1, 1)
            meta_feats.append(preds)
        return np.hstack(meta_feats)

    def predict(self, X):
        meta_X = self._build_meta_features_from_bases(X)
        return self.meta_estimator_.predict(meta_X)

    def predict_proba(self, X):
        if hasattr(self.meta_estimator_, "predict_proba"):
            meta_X = self._build_meta_features_from_bases(X)
            return self.meta_estimator_.predict_proba(meta_X)
        else:
            raise AttributeError("meta_estimator does not support predict_proba")
        


def build_model_pipeline_regressor(model_name, model_params=None):
    """
    Retorna pipeline de pré-processamento + regressor (paralela à build_model_pipeline de classificação).
    Aceita as mesmas chaves usadas na função de classificação; aqui retornamos as versões regressor.
    Model names suportados (mesmos da versão de classificação):
      - 'knn'
      - 'logistic_regression' or 'logistic'   -> LinearRegression (equivalente)
      - 'rf' or 'random_forest'
      - 'svm'
      - 'arvore_decisao' or 'decision_tree'
      - 'naive_bayes' or 'naive' or 'nb'       -> NÃO IMPLEMENTADO (sem equivalente direto)
      - 'ensemble'                            -> VotingRegressor
      - 'mlp' or 'neural_network' or 'rede_neural'
      - 'bagging'
      - 'boosting'
      - 'stacking'
      - 'blending'                            -> NÃO IMPLEMENTADO (sem equivalente padrão)
    """
    model_params = model_params or {}
    name = model_name.lower()

    if name == 'knn':
        reg = KNeighborsRegressor(**model_params)
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])  # KNN precisa de escala

    elif name in ('linear_regression'):
        reg = LinearRegression(**model_params)
        pipe = Pipeline([("reg", reg)])  # linear não precisa necessariamente de scaler

    elif name in ('rf', 'random_forest'):
        reg = RandomForestRegressor(**model_params)
        pipe = Pipeline([("reg", reg)])

    elif name == 'svm':
        reg = SVR(**model_params)
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])  # SVR precisa de escala

    elif name in ('arvore_decisao', 'decision_tree'):
        reg = DecisionTreeRegressor(**model_params)
        pipe = Pipeline([("reg", reg)])

    elif name in ('gaussian_process', 'gp'):
        reg = GaussianProcessRegressor(**model_params)
        pipe = Pipeline([("reg", reg)])

    elif name == 'ensemble':
        # VotingRegressor com uma combinação simples (Linear + RF)
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100)
        reg = VotingRegressor(estimators=[("lr", lr), ("rf", rf)])
        pipe = Pipeline([("reg", reg)])

    elif name in ('mlp', 'neural_network', 'rede_neural'):
        reg = MLPRegressor(max_iter=1000, **model_params)
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])  # MLP se beneficia de escala

    elif name == 'bagging':
        # evitar modificar model_params original
        params = dict(model_params) if model_params is not None else {}
        # aceitar tanto 'estimator' (novo nome) quanto 'base_estimator' (compatibilidade)
        if 'estimator' in params:
            base = params.pop('estimator')
        else:
            base = params.pop('base_estimator', DecisionTreeRegressor())
        # BaggingRegressor espera o parâmetro 'estimator' em versões recentes do sklearn
        reg = BaggingRegressor(estimator=base, **params)
        pipe = Pipeline([("reg", reg)])

    elif name == 'boosting':
        reg = GradientBoostingRegressor(**model_params)
        pipe = Pipeline([("reg", reg)])

    elif name == 'stacking':
        # aceita 'estimators' (list of (name, estimator)), 'final_estimator', 'cv', 'n_jobs'
        estimators = model_params.get('estimators', None)
        final_estimator = model_params.get('final_estimator', LinearRegression())
        cv = model_params.get('cv', 5)
        n_jobs = model_params.get('n_jobs', None)

        if estimators is None:
            # encapsular os modelos que precisariam de scaler quando for o caso
            estimators = [
                ("lr", LinearRegression()),
                ("rf", RandomForestRegressor(n_estimators=100))
            ]

        reg = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=cv, n_jobs=n_jobs)
        pipe = Pipeline([("reg", reg)])

    elif name == 'bagging':
        # assume model_params é um dict opcional disponível no escopo; caso não tenha, use {}
        mp = model_params.copy() if model_params is not None else {}

        # permitir passar um estimator personalizado via 'estimator' em model_params
        estimator = mp.pop('estimator', DecisionTreeRegressor(random_state=random_state))

        # parâmetros comuns com defaults sensatos
        n_estimators = mp.pop('n_estimators', 10)
        max_samples = mp.pop('max_samples', 1.0)       # proporção ou int
        max_features = mp.pop('max_features', 1.0)     # proporção ou int
        bootstrap = mp.pop('bootstrap', True)
        bootstrap_features = mp.pop('bootstrap_features', False)
        n_jobs = mp.pop('n_jobs', -1)
        verbose = mp.pop('verbose', 0)
        random_state = mp.pop('random_state', random_state)

        model = BaggingRegressor(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            **mp  # permite pass-through de params extras
        )
    

    else:
        raise NotImplementedError(f"Modelo {model_name} não implementado para regressão.")

    return pipe

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
        clf = BaggingClassifier(estimator=base_clf, **model_params)
        pipe = Pipeline([("clf", clf)]) # não precisa de escala
    elif name == 'boosting':
        clf = GradientBoostingClassifier(**model_params)
        pipe = Pipeline([("clf", clf)]) # não precisa de escala
    elif name == 'stacking':
        # model_params aceita:
        # 'estimators' (lista de tuples (name, estimator)),
        # 'final_estimator' (estimator),
        # 'cv', 'stack_method', 'passthrough', 'n_jobs'
        # extrai parâmetros do dict (ou usa defaults)
        estimators = model_params.get('estimators', None)
        final_estimator = model_params.get('final_estimator', LogisticRegression(max_iter=10000))
        cv = model_params.get('cv', 5)
        stack_method = model_params.get('stack_method', 'predict_proba')
        passthrough = model_params.get('passthrough', False)
        n_jobs = model_params.get('n_jobs', None)

        # se o usuário não passou estimadores, criamos um conjunto razoável
        if estimators is None:
            # encapsular modelos que precisam de escala em Pipeline
            svc_pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True))])
            estimators = [
                ("lr", LogisticRegression(max_iter=1000)),
                ("rf", RandomForestClassifier(n_estimators=100)),
                ("svc", svc_pipe)
            ]

        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method=stack_method,
            passthrough=passthrough,
            n_jobs=n_jobs
        )
        pipe = Pipeline([("clf", clf)])
    elif name == 'blending':
        # model_params aceita:
        # 'estimators' (lista of tuples), 'final_estimator', 'holdout_size',
        # 'use_proba', 'random_state', 'n_jobs' (para os base, se aplicável)
        estimators = model_params.get('estimators', None)
        final_estimator = model_params.get('final_estimator', LogisticRegression(max_iter=10000))
        holdout_size = model_params.get('holdout_size', 0.2)
        use_proba = model_params.get('use_proba', 'auto')
        random_state = model_params.get('random_state', None)
        retrain_base_on_full = model_params.get('retrain_base_on_full', True)

        if estimators is None:
            # cria um conjunto padrão (encapsula SVC com scaler)
            estimators = [
                ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
                ("svc", Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=random_state))])),
                ("lr", LogisticRegression(max_iter=1000, random_state=random_state))
            ]

        clf = BlendingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            holdout_size=holdout_size,
            use_proba=use_proba,
            random_state=random_state,
            retrain_base_on_full=retrain_base_on_full
        )
        pipe = Pipeline([("clf", clf)])
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
    permutation_n_repeats=10,
    compute_importance=True,    # ligar/desligar cálculo de importâncias
    validation_method='holdout' # 'holdout' (default) ou 'k_fold'
):
    """
    Treina e testa um classificador.
    validation_method:
      - 'holdout' : comportamento original (usa train_size + holdout_indices)
      - 'k_fold'  : aplica StratifiedKFold com 5 folds sobre todo o dataset
    Retorna dicionário com:
      - model: pipeline treinado no conjunto inteiro (para k_fold) ou pipeline treinado no holdout
      - cv_models: list de pipelines treinados por fold (apenas k_fold)
      - metrics: para holdout -> métricas únicas; para k_fold -> lista de folds + médias
      - feature_importances, timings, etc.
    """

    from sklearn.model_selection import StratifiedKFold

    t_start_total = perf_counter()

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
    y_arr = np.array(y)

    n = len(X_arr)
    if n == 0:
        raise ValueError("O dataset está vazio após o pré-processamento.")

    # Inicializações comuns
    feature_importances_df = None
    method_used = None
    permutation_time = None
    shap_time = None
    timings = {}

    # --------------------------
    # K-FOLD VALIDATION
    # --------------------------
    if validation_method == 'k_fold':
        # fixed 5 folds as requested
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_metrics = []
        cv_models = []
        fold_timings = []

        # labels for confusion matrix consistent across folds
        all_labels = np.unique(y_arr)

        print("=" * 70)
        print(f" K-FOLD CROSS VALIDATION - {model_name.upper()}")
        print("=" * 70)
        print(f" Dataset: {len(X_arr)} samples, {len(feature_names)} features")
        print(f" Target classes: {list(all_labels)}")
        print(f" Folds: {n_splits}")
        print(f" Model parameters: {model_params}")
        print("-" * 70)

        fold_idx = 0
        for train_idx, val_idx in skf.split(X_arr, y_arr):
            fold_idx += 1

            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            # build pipeline for this fold (fresh instance)
            pipeline = build_model_pipeline(model_name, model_params=model_params)

            print(f"\n FOLD {fold_idx}/{n_splits}")
            print(f"    Train: {len(train_idx)} samples | Test: {len(val_idx)} samples")

            t0 = perf_counter()
            pipeline.fit(X_tr, y_tr)
            t1 = perf_counter()
            fit_t = t1 - t0

            t0p = perf_counter()
            y_pred = pipeline.predict(X_val)
            t1p = perf_counter()
            pred_t = t1p - t0p

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

            fold_metrics.append({
                'fold': fold_idx,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            })

            cv_models.append(pipeline)
            fold_timings.append({'fit_time': fit_t, 'predict_time': pred_t})
            
            print(f"    Fit: {fit_t:.4f}s | Predict: {pred_t:.4f}s")
            print(f"    Metrics:")
            print(f"       Accuracy:  {acc:.4f}")
            print(f"       Precision: {prec:.4f}")
            print(f"       Recall:    {rec:.4f}")
            print(f"       F1-Score:  {f1:.4f}")

        # calcular médias e desvios padrão
        accuracies = [m['accuracy'] for m in fold_metrics]
        precisions = [m['precision'] for m in fold_metrics]
        recalls = [m['recall'] for m in fold_metrics]
        f1s = [m['f1'] for m in fold_metrics]

        mean_metrics = {
            'accuracy_mean': float(np.mean(accuracies)),
            'precision_mean': float(np.mean(precisions)),
            'recall_mean': float(np.mean(recalls)),
            'f1_mean': float(np.mean(f1s))
        }
        
        std_metrics = {
            'accuracy_std': float(np.std(accuracies)),
            'precision_std': float(np.std(precisions)),
            'recall_std': float(np.std(recalls)),
            'f1_std': float(np.std(f1s))
        }

        # Treinar modelo final em todo o conjunto
        print(f"\n TREINANDO MODELO FINAL COM TODOS OS DADOS")
        final_pipeline = build_model_pipeline(model_name, model_params=model_params)
        t0_full = perf_counter()
        final_pipeline.fit(X_arr, y_arr)
        t1_full = perf_counter()
        fit_full_time = t1_full - t0_full
        print(f"    Modelo final treinado em {fit_full_time:.4f}s")

        # Cálculo de importância das features
        method_used = None
        feature_importances_df = None
        
        if compute_importance:
            print(f"\n🔍 CALCULANDO IMPORTÂNCIA DAS FEATURES")
            model_lower = model_name.lower()
            if importance_method == 'auto':
                chosen = 'coef' if model_lower in ['logistic_regression', 'logistic'] else 'permutation'
            else:
                chosen = importance_method
                
            print(f"   Método selecionado: {chosen.upper()}")

            if chosen == 'coef':
                try:
                    estimator = final_pipeline.named_steps['clf']
                    coefs = estimator.coef_
                    if coefs.ndim == 1:
                        imp = np.abs(coefs)
                    else:
                        imp = np.mean(np.abs(coefs), axis=0)
                    feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False).reset_index(drop=True)
                    method_used = 'coef'
                    print(f"    Importância por coeficientes calculada")
                except Exception as e:
                    print(f"    Erro em coef: {e} -> usando permutation")
                    chosen = 'permutation'

            if chosen == 'permutation':
                t0_perm = perf_counter()
                try:
                    perm = permutation_importance(final_pipeline, X_arr, y_arr, n_repeats=permutation_n_repeats, random_state=random_state, n_jobs=-1)
                    t1_perm = perf_counter()
                    permutation_time = t1_perm - t0_perm
                    imp_means = perm.importances_mean
                    imp_stds = perm.importances_std
                    feature_importances_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance_mean': imp_means,
                        'importance_std': imp_stds
                    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
                    method_used = 'permutation'
                    print(f"    Permutation importance calculada ({permutation_time:.2f}s)")
                except Exception as e:
                    print(f"    Erro em permutation: {e}")

            if chosen == 'shap':
                if not _HAS_SHAP:
                    print("    SHAP não disponível")
                else:
                    try:
                        t0_shap = perf_counter()
                        estimator = final_pipeline.named_steps['clf']
                        try:
                            X_trans = final_pipeline[:-1].transform(X_arr)
                        except Exception:
                            X_trans = X_arr
                        explainer = shap.Explainer(estimator, X_trans)
                        shap_values = explainer(X_trans)
                        t1_shap = perf_counter()
                        shap_time = t1_shap - t0_shap
                        shap_abs_mean = np.mean(np.abs(shap_values.values), axis=0)
                        feature_importances_df = pd.DataFrame({'feature': feature_names, 'shap_abs_mean': shap_abs_mean}).sort_values('shap_abs_mean', ascending=False).reset_index(drop=True)
                        method_used = 'shap'
                        print(f"    SHAP importance calculada ({shap_time:.2f}s)")
                    except Exception as e:
                        print(f"    Erro em SHAP: {e}")

        t_end_total = perf_counter()
        total_time = t_end_total - t_start_total

        # RESULTADOS FINAIS
        print("\n" + "=" * 70)
        print(f" RESULTADOS FINAIS - {model_name.upper()}")
        print("=" * 70)
        
        print(f"📊 Métricas por fold:")
        for i, metrics in enumerate(fold_metrics, 1):
            print(f"   Fold {i}: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
                f"Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        print(f"\n Estatísticas consolidadas:")
        print(f"    Accuracy:  {mean_metrics['accuracy_mean']:.4f} (±{std_metrics['accuracy_std']:.4f})")
        print(f"    Precision: {mean_metrics['precision_mean']:.4f} (±{std_metrics['precision_std']:.4f})")
        print(f"    Recall:    {mean_metrics['recall_mean']:.4f} (±{std_metrics['recall_std']:.4f})")
        print(f"    F1-Score:  {mean_metrics['f1_mean']:.4f} (±{std_metrics['f1_std']:.4f})")
        
        print(f"\nTempos de execução:")
        print(f"   Fit por fold:    {np.mean([ft['fit_time'] for ft in fold_timings]):.4f}s")
        print(f"   Predict por fold: {np.mean([ft['predict_time'] for ft in fold_timings]):.4f}s")
        print(f"   Modelo final:    {fit_full_time:.4f}s")
        print(f"   Total:           {total_time:.4f}s")
        
        if feature_importances_df is not None:
            print(f"\n🔝 Top 5 features mais importantes ({method_used}):")
            top_features = feature_importances_df.head(5)
            for _, row in top_features.iterrows():
                if 'importance_mean' in row:
                    print(f"   📍 {row['feature']}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})")
                else:
                    importance_val = row.get('importance', row.get('shap_abs_mean', 0))
                    print(f"   📍 {row['feature']}: {importance_val:.4f}")

        print("=" * 70)

        # montar métricas de retorno
        metrics = {
            'validation_method': 'k_fold',
            'n_splits': n_splits,
            'folds': fold_metrics,
            'mean': mean_metrics,
            'std': std_metrics
        }

        timings = {
            'fit_time_per_fold': [ft['fit_time'] for ft in fold_timings],
            'predict_time_per_fold': [ft['predict_time'] for ft in fold_timings],
            'fit_full_time': fit_full_time,
            'total_time': total_time
        }

        return {
            'model': final_pipeline,
            'cv_models': cv_models,
            'metrics': metrics,
            'feature_names': feature_names,
            'feature_importances': feature_importances_df,
            'importance_method': method_used,
            'timings': timings
        }

    # --------------------------
    # HOLDOUT (original flow)
    # --------------------------
    else:
        # 5) Holdout indices (original)
        train_idx, test_idx = holdout_indices(n, train_size=train_size, random_state=random_state)

        X_train = X_arr[train_idx]
        X_test = X_arr[test_idx]
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]

        # 6) Pipeline e treino
        pipeline = build_model_pipeline(model_name, model_params=model_params)

        t0_fit = perf_counter()
        pipeline.fit(X_train, y_train)
        t1_fit = perf_counter()
        fit_time = t1_fit - t0_fit

        # 7) Previsões e métricas
        t0_pred = perf_counter()
        y_pred = pipeline.predict(X_test)
        t1_pred = perf_counter()
        predict_time = t1_pred - t0_pred

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

        # 8) Importância das features (opcional) — mesma lógica anterior
        feature_importances_df = None
        method_used = None
        permutation_time = None
        shap_time = None

        if compute_importance:
            model_lower = model_name.lower()
            if importance_method == 'auto':
                chosen = 'coef' if model_lower in ['logistic_regression', 'logistic'] else 'permutation'
            else:
                chosen = importance_method

            # coef
            if chosen == 'coef':
                try:
                    estimator = pipeline.named_steps['clf']
                    coefs = estimator.coef_
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
                    if verbose:
                        print("Erro ao extrair coef_: ", e, " -> fallback para permutation importance")
                    chosen = 'permutation'

            if chosen == 'permutation':
                t0_perm = perf_counter()
                try:
                    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=permutation_n_repeats, random_state=random_state, n_jobs=-1)
                    t1_perm = perf_counter()
                    permutation_time = t1_perm - t0_perm

                    imp_means = perm.importances_mean
                    imp_stds = perm.importances_std
                    feature_importances_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance_mean': imp_means,
                        'importance_std': imp_stds
                    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
                    method_used = 'permutation'
                except Exception as e:
                    t1_perm = perf_counter()
                    permutation_time = t1_perm - t0_perm
                    if verbose:
                        print("Erro em permutation_importance:", e)
                    feature_importances_df = None

            if chosen == 'shap':
                if not _HAS_SHAP:
                    if verbose:
                        print("SHAP não encontrado (pip install shap) — fallback para permutation importance")
                    # fallback handled above
                else:
                    try:
                        t0_shap = perf_counter()

                        estimator = pipeline.named_steps['clf']
                        try:
                            X_train_transformed = pipeline[:-1].transform(X_train)
                            X_test_transformed = pipeline[:-1].transform(X_test)
                        except Exception:
                            X_train_transformed = X_train
                            X_test_transformed = X_test

                        explainer = shap.Explainer(estimator, X_train_transformed)
                        shap_values = explainer(X_test_transformed)

                        t1_shap = perf_counter()
                        shap_time = t1_shap - t0_shap

                        shap_abs_mean = np.mean(np.abs(shap_values.values), axis=0)
                        feature_importances_df = pd.DataFrame({
                            'feature': feature_names,
                            'shap_abs_mean': shap_abs_mean
                        }).sort_values('shap_abs_mean', ascending=False).reset_index(drop=True)
                        method_used = 'shap'
                    except Exception as e:
                        if verbose:
                            print("Erro ao rodar SHAP:", e, "-> fallback para permutation")
                        try:
                            t0_perm = perf_counter()
                            perm = permutation_importance(pipeline, X_test, y_test, n_repeats=permutation_n_repeats, random_state=random_state, n_jobs=-1)
                            t1_perm = perf_counter()
                            permutation_time = t1_perm - t0_perm

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

        t_end_total = perf_counter()
        total_time = t_end_total - t_start_total

        # timings and return (holdout)
        timings = {
            'fit_time': fit_time,
            'predict_time': predict_time,
            'permutation_time': permutation_time,
            'shap_time': shap_time,
            'total_time': total_time
        }

        if verbose:
            print("Modelo:", model_name)
            print("Tamanho treino/teste:", len(X_train), "/", len(X_test))
            print(f"Acurácia: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
            print("\nClassification Report:\n", metrics['classification_report'])
            print("\nConfusion Matrix:\n", metrics['confusion_matrix'])
            print("\nFeature importance method used:", method_used)
            if feature_importances_df is not None:
                print(feature_importances_df.head(20))
            print("\n--- Timings (em segundos) ---")
            print(f"Fit time:       {fit_time:.4f}s")
            print(f"Predict time:   {predict_time:.4f}s")
            if permutation_time is not None:
                print(f"Permutation importance time: {permutation_time:.4f}s")
            if shap_time is not None:
                print(f"SHAP time: {shap_time:.4f}s")
            print(f"Total elapsed time: {total_time:.4f}s")
            print("-" * 100)

        return {
            'model': pipeline,
            'X-train': X_train,
            'X-test': X_test,
            'y-train': y_train,
            'y-test': y_test,
            'metrics': metrics,
            'feature_names': feature_names,
            'feature_importances': feature_importances_df,
            'importance_method': method_used,
            'timings': timings
        }


def train_and_test_regressor(
    model_name,
    csv_path,
    feature_columns,
    target_column,
    train_size=0.65,
    random_state=None,
    model_params=None,
    verbose=False,
    compute_importance=True,
    permutation_n_repeats=10,
    validation_method='holdout'  # 'holdout' (default) ou 'k_fold'
):
    """
    Versão para regressão da função de treino/teste com saída colorida via ANSI escape codes.
    - Retorna métricas: R2, MSE, RMSE, MAE
    - Adiciona baseline (mean predictor), razões vs baseline e métricas normalizadas
    - Possui opção de ajuste automático: use model_params='auto' ou model_params={'auto':True}
      - Para personalizar o grid, passe model_params={'auto':True, 'auto_grid': {...}}
    - validation_method: 'holdout' ou 'k_fold'
    - Mantém saída similar à versão classificadora.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # habilita ANSI no Windows (se possível)
    import os
    try:
        if os.name == 'nt':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        # se falhar, continua (ANSI pode não estar habilitado)
        pass

    # ANSI colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GREY = "\033[90m"

    def c(text, color=RESET, bold=False):
        if bold:
            return f"{BOLD}{color}{text}{RESET}"
        return f"{color}{text}{RESET}"

    
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    from time import perf_counter
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import RandomForestRegressor

    t_start_total = perf_counter()

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
    feature_names = X.columns.tolist()

    X_arr = X.values
    y_arr = np.array(y)
    n = len(X_arr)
    if n == 0:
        raise ValueError("O dataset está vazio após o pré-processamento.")

    # ===== Auto-tuning (KNN e RandomForest) =====
    do_auto = False
    auto_grid = None
    auto_model_kind = None  # 'knn' or 'rf'

    # --- bloco que detecta "auto" e faz autotune para vários modelos ---
    if model_params == 'auto' or (isinstance(model_params, dict) and model_params.get('auto', False)):
        do_auto = True
        mn = model_name.lower()

        # mapeamento do nome recebido para a chave do grid
        if mn in ('knn', 'kneighbors', 'kneighborsregressor'):
            auto_model_kind = 'knn'
        elif mn in ('rf', 'random_forest', 'randomforest'):
            auto_model_kind = 'rf'
        elif mn in ('linear_regression', 'linear', 'lr'):
            auto_model_kind = 'linear_regression'
        elif mn in ('svr', 'svm'):
            auto_model_kind = 'svm'
        elif mn in ('arvore_decisao', 'decision_tree', 'dt'):
            auto_model_kind = 'decision_tree'
        elif mn in ('gaussian_process', 'gp'):
            auto_model_kind = 'gaussian_process'
        elif mn in ('mlp', 'neural_network', 'rede_neural'):
            auto_model_kind = 'mlp'
        elif mn in ('bagging',):
            auto_model_kind = 'bagging'
        elif mn in ('gbr', 'gradient_boosting', 'gradient_boosting_regressor', 'gradient'):
            auto_model_kind = 'gbr'
        elif mn in ('stacking',):
            auto_model_kind = 'stacking'
        elif mn in ('ensemble', 'voting', 'votingregressor'):
            auto_model_kind = 'voting'
        elif mn in ('bagging'):
            auto_model_kind = 'bagging'    
        else:
            auto_model_kind = mn  # fallback — se tiver um nome novo, tentamos usar direto

        # grids default por modelo (pode ajustar conforme seu gosto)
        default_grids = {
            'knn': {
                'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
                'knn__weights': ['uniform', 'distance'],
                'knn__p': [1, 2]
            },
            'rf': {
                'rf__n_estimators': [100, 200, 400],
                'rf__max_depth': [None, 10, 20],
                'rf__min_samples_split': [2, 5],
                'rf__min_samples_leaf': [1, 2],
                'rf__max_features': ['auto', 'sqrt']
            },
            'linear_regression': {
                'lr__fit_intercept': [True, False],
                # 'normalize' deprecated em versões recentes; mantive só como exemplo
                'lr__normalize': [True, False]
            },
           'svm': {
                # experimentar kernels: rbf é default bom; testar linear (rápido) e poly (se suspeitar interação)
                'svr__kernel': ['rbf', 'linear', 'poly'],

                # C em escala log (regularização): do muito fraco ao muito forte
                'svr__C': [1e-3, 1e-2, 1e-1, 1, 10],

                # gamma para RBF/poly: testar 'scale' e valores explícitos em ordens de magnitude
                'svr__gamma': ['scale', 'auto', 1e-3, 1e-2, 1e-1, 1],

                # epsilon (tamanho da zona insensível): valores pequenos para regressão precisa
                'svr__epsilon': [1e-4, 1e-3, 1e-2, 0.05],

                # parâmetros para kernel polinomial (somente usados se kernel='poly')
                'svr__degree': [2, 3, 4],
                'svr__coef0': [0.0, 0.1, 0.5],

                # tolerância e número de iterações podem ajudar convergência
                'svr__tol': [1e-4, 1e-3],
                'svr__max_iter': [-1]  # -1 = sem limite; altere se precisar de limites
            },
            'decision_tree': {
                'dt__max_depth': [None, 5, 10, 20],
                'dt__min_samples_split': [2, 5, 10],
                'dt__min_samples_leaf': [1, 2, 4]
            },
            'gaussian_process': {
                # GPR tuning costuma ser pesado; exemplos simples:
                'gp__alpha': [1e-10, 1e-2, 1e-1]
            },
            'mlp': {
                'mlp__hidden_layer_sizes': [(50,), (100,), (100,50)],
                'mlp__activation': ['relu', 'tanh'],
                'mlp__alpha': [1e-4, 1e-3, 1e-2]
            },
            'bagging': {
                'bag__n_estimators': [10, 50, 100],
                'bag__max_samples': [0.5, 0.8, 1.0]
            },
            'gbr': {
                'gbr__n_estimators': [100, 200],
                'gbr__learning_rate': [0.01, 0.1],
                'gbr__max_depth': [3, 5]
            },
            'stacking': {
                # ajustar final_estimator params via prefix 'stack__final_estimator__...'
                'stack__final_estimator': [LinearRegression()],
                # exemplo de grid que muda estimators (útil só se você aceitar objetos diferentes)
                # 'stack__estimators': [[("lr", LinearRegression()), ("rf", RandomForestRegressor())]]
            },
            'voting': {
                # VotingRegressor não tem muitos parâmetros; podemos tentar ajustar pesos se usar 'weights' (lista)
                # ao usar GridSearch, forneça listas de pesos compatíveis com o número de estimators definidos.
                # exemplo vazio por padrão (usuário deve fornecer auto_grid customizada se quiser)
            }
        }

        # pegar auto_grid customizado se fornecido
        if isinstance(model_params, dict) and 'auto_grid' in model_params and isinstance(model_params['auto_grid'], dict):
            auto_grid = model_params['auto_grid']
        else:
            if auto_model_kind in default_grids:
                auto_grid = default_grids[auto_model_kind]
            else:
                # se não tivermos grid definido, não habilita auto
                raise ValueError(f"Nenhum grid padrão definido para '{auto_model_kind}'. Forneça model_params['auto_grid'].")

    # inicializações
    feature_importances_df = None
    method_used = None
    permutation_time = None
    timings = {}

    # helper: função para rodar GridSearchCV para vários estimadores
    def _autotune_for_model(X_train_for_grid, y_train_for_grid, model_kind, cv=3):
        """
        model_kind: string (ex.: 'knn', 'rf', 'linear_regression', 'svm', 'decision_tree', 'gaussian_process',
                    'mlp', 'bagging', 'gbr', 'stacking', 'voting')
        retorna: (best_params_for_estimator, best_score, gridsearch_obj)
        best_params_for_estimator: dct sem o primeiro prefixo (ex.: {'n_neighbors':5} ou {'n_estimators':100})
        """
        mk = model_kind.lower()

        # montar pipeline apropriada (usar nomes de passo compatíveis com as chaves dos grids acima)
        if mk == 'knn':
            pipeline_for_grid = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsRegressor())
            ])
        elif mk in ('rf', 'random_forest'):
            pipeline_for_grid = Pipeline([
                ('scaler', StandardScaler()),  # mesmo que RF não precise, mantém consistência
                ('rf', RandomForestRegressor(n_jobs=-1))
            ])
        elif mk in ('linear_regression', 'linear', 'lr'):
            pipeline_for_grid = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LinearRegression())
            ])
        elif mk in ('svm', 'svr'):
            pipeline_for_grid = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR())
            ])
        elif mk in ('arvore_decisao', 'decision_tree', 'dt'):
            pipeline_for_grid = Pipeline([
                ('dt', DecisionTreeRegressor())
            ])
        elif mk in ('gaussian_process', 'gp'):
            pipeline_for_grid = Pipeline([
                ('gp', GaussianProcessRegressor())
            ])
        elif mk in ('mlp', 'neural_network', 'rede_neural'):
            pipeline_for_grid = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(max_iter=1000))
            ])
        elif mk == 'bagging':
            # use 'estimator' keyword (newer sklearn) but keep compatibility by constructing explicitly
            pipeline_for_grid = Pipeline([
                    ('bag', BaggingRegressor(estimator=DecisionTreeRegressor()))
                ])
        elif mk in ('gbr', 'gradient_boosting', 'gradient_boosting_regressor', 'gradient'):
            pipeline_for_grid = Pipeline([
                ('gbr', GradientBoostingRegressor())
            ])
        elif mk in ('stacking',):
            # stacking: criamos um stacking com estimators default (podem ser sobrescritos no grid)
            estimators = [
                ("lr", LinearRegression()),
                ("rf", RandomForestRegressor(n_estimators=100))
            ]
            pipeline_for_grid = Pipeline([
                ('stack', StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), n_jobs=-1))
            ])
        elif mk in ('voting', 'ensemble', 'votingregressor'):
            estimators = [
                ("lr", LinearRegression()),
                ("rf", RandomForestRegressor(n_estimators=100))
            ]
            pipeline_for_grid = Pipeline([
                ('vot', VotingRegressor(estimators=estimators))
            ])
        else:
            raise ValueError(f"model_kind '{model_kind}' não suportado pelo autotune.")

        # escolher grid apropriada — espera-se que auto_grid esteja definido no escopo acima
        grid_to_use = auto_grid

        # executar GridSearchCV
        gs = GridSearchCV(pipeline_for_grid, grid_to_use, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1, refit=True)
        gs.fit(X_train_for_grid, y_train_for_grid)

        best = gs.best_params_
        # remover apenas o primeiro prefixo (ex.: 'knn__n_neighbors' -> 'n_neighbors')
        best_params_for_estimator = {k.split('__', 1)[1]: v for k, v in best.items()}
        best_score = gs.best_score_
        return best_params_for_estimator, best_score, gs

    def _sanitize_model_params_for_estimator(model_params):
        """
        - Remove chaves controle ('auto','auto_grid').
        - Se encontrar chaves do tipo 'prefix__param', retorna {'param': value}.
        - Se model_params is 'auto' -> retorna {}.
        """
        if model_params is None:
            return {}
        if isinstance(model_params, str):
            # caso model_params == 'auto' (string), retorna vazio (os params virão do auto-tuning)
            return {}
        if not isinstance(model_params, dict):
            # guard para qualquer outro tipo inesperado
            return {}

        sanitized = {}
        for k, v in model_params.items():
            if k in ('auto', 'auto_grid'):
                continue
            if '__' in k:
                # pega só a parte após o __ (ex.: 'knn__n_neighbors' -> 'n_neighbors')
                sanitized[k.split('__', 1)[1]] = v
            else:
                sanitized[k] = v
        return sanitized

    # --------------------------
    # K-FOLD
    # --------------------------
    if validation_method == 'k_fold':
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Se pediu auto e é KNN, rodar GridSearch antes dos folds (observação abaixo)
        if do_auto:
            if verbose:
                print(c("   Ajuste automático solicitado — executando GridSearch (cv=3) ...", YELLOW))
                print(c("   Nota: para avaliação sem viés, prefira nested CV. Aqui usamos GridSearch sobre todo o dataset antes do k-fold final,", GREY))
                print(c("   o que pode introduzir viés otimista nas métricas reportadas. Se quiser, eu posso ajustar para nested CV.", GREY))
            try:
                best_params_est, best_score, gs_obj = _autotune_for_model(X_arr, y_arr, auto_model_kind, cv=3)
                # atualizar model_params para usar os melhores parâmetros encontrados
                if isinstance(model_params, dict):
                    model_params = {**model_params, **best_params_est}
                    # remover as chaves de controle se existirem
                    model_params.pop('auto', None)
                    model_params.pop('auto_grid', None)
                else:
                    model_params = best_params_est
                if verbose:
                    print(c(f"   -> Melhores params encontrados (GridSearch): {best_params_est}", CYAN))
                    print(c(f"   -> best_score (neg MAE): {best_score:.4f}", CYAN))
            except Exception as e:
                if verbose:
                    print(c("Erro durante auto-tuning KNN:", RED), e)
                do_auto = False

        fold_metrics = []
        cv_models = []
        fold_timings = []

        if verbose:
            print(c("=" * 70, GREY))
            print(c(f"K-FOLD CROSS VALIDATION (REGRESSÃO) - {model_name.upper()}", BOLD + MAGENTA))
            print(c(f"Dataset: {len(X_arr)} samples, {len(feature_names)} features", CYAN))
            print(c(f"Folds: {n_splits}", CYAN))
            print(c(f"Model parameters: {model_params}", CYAN))
            print(c("-" * 70, GREY))

        fold_idx = 0
        for train_idx, val_idx in kf.split(X_arr):
            fold_idx += 1
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            model_params_sanitized = _sanitize_model_params_for_estimator(model_params)

            pipeline = build_model_pipeline_regressor(model_name, model_params=model_params_sanitized)

            if verbose:
                print()
                print(c(f"📁 FOLD {fold_idx}/{n_splits} - Train {len(train_idx)} | Val {len(val_idx)}", BOLD + BLUE))

            t0 = perf_counter()
            pipeline.fit(X_tr, y_tr)
            t1 = perf_counter()
            fit_t = t1 - t0

            t0p = perf_counter()
            y_pred = pipeline.predict(X_val)
            t1p = perf_counter()
            pred_t = t1p - t0p

            # métricas do modelo
            r2 = float(r2_score(y_val, y_pred))
            mse = float(mean_squared_error(y_val, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_val, y_pred))

            # baseline: prever a média do conjunto de treino (chutar a média)
            y_tr_mean = float(np.mean(y_tr))
            baseline_pred = np.full_like(y_val, fill_value=y_tr_mean, dtype=float)
            baseline_mse = float(mean_squared_error(y_val, baseline_pred))
            baseline_rmse = float(np.sqrt(baseline_mse))
            baseline_mae = float(mean_absolute_error(y_val, baseline_pred))

            # razões vs baseline (quanto do erro do baseline o modelo alcança)
            mse_ratio = mse / baseline_mse if baseline_mse != 0 else np.nan
            rmse_ratio = rmse / baseline_rmse if baseline_rmse != 0 else np.nan
            mae_ratio = mae / baseline_mae if baseline_mae != 0 else np.nan

            # métricas normalizadas (em relação à média do alvo no conjunto de validação) — úteis em percentuais
            mean_y_val = float(np.mean(y_val)) if len(y_val) > 0 else np.nan
            rmse_norm = rmse / mean_y_val if mean_y_val != 0 else np.nan
            mae_norm = mae / mean_y_val if mean_y_val != 0 else np.nan

            # MAPE (apenas quando valores não-zereos)
            eps = 1e-9
            if np.any(np.abs(y_val) > eps):
                mape = float(np.mean(np.abs((y_val - y_pred) / (y_val + eps)))) * 100.0
            else:
                mape = np.nan

            fold_metrics.append({
                'fold': fold_idx,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'baseline_mse': baseline_mse,
                'baseline_rmse': baseline_rmse,
                'baseline_mae': baseline_mae,
                'mse_ratio': mse_ratio,
                'rmse_ratio': rmse_ratio,
                'mae_ratio': mae_ratio,
                'rmse_norm': rmse_norm,
                'mae_norm': mae_norm,
                'mape_percent': mape
            })

            cv_models.append(pipeline)
            fold_timings.append({'fit_time': fit_t, 'predict_time': pred_t})

            if verbose:
                print(c(f"    Fit: {fit_t:.4f}s | Predict: {pred_t:.4f}s", GREY))
                # explicação curta para interpretação
                print(c(f"   {CYAN}R2{RESET}: {GREEN}{r2:.4f}{RESET} | {CYAN}MSE{RESET}: {YELLOW}{mse:.4f}{RESET} (unidade^2) | {CYAN}RMSE{RESET}: {GREEN}{rmse:.4f}{RESET} | {CYAN}MAE{RESET}: {GREEN}{mae:.4f}{RESET}"))
                print(c(f"      -> RMSE e MAE estão na mesma unidade do alvo (ex.: reais). MSE está ao quadrado; use RMSE/MAE para interpretação prática.", GREY))
                print(c(f"   Baseline(mean-train) RMSE: {YELLOW}{baseline_rmse:.4f}{RESET} | MAE: {YELLOW}{baseline_mae:.4f}{RESET}", GREY))
                print(c(f"      -> RMSE_ratio (model/baseline): {MAGENTA}{rmse_ratio:.4f}{RESET} | MAE_ratio: {MAGENTA}{mae_ratio:.4f}{RESET}"))
                if not np.isnan(rmse_norm):
                    print(c(f"      -> RMSE/mean(y_val): {GREEN}{rmse_norm:.4f}{RESET} ({rmse_norm*100:.2f}%) | MAE/mean(y_val): {GREEN}{mae_norm:.4f}{RESET} ({mae_norm*100:.2f}%)", GREY))
                if not np.isnan(mape):
                    print(c(f"      -> MAPE: {GREEN}{mape:.2f}%{RESET}", GREY))

        # médias e desvios
        r2s = [m['r2'] for m in fold_metrics]
        mses = [m['mse'] for m in fold_metrics]
        rmses = [m['rmse'] for m in fold_metrics]
        maes = [m['mae'] for m in fold_metrics]
        baseline_rmses = [m['baseline_rmse'] for m in fold_metrics]
        baseline_maes = [m['baseline_mae'] for m in fold_metrics]
        rmse_ratios = [m['rmse_ratio'] for m in fold_metrics]
        mae_ratios = [m['mae_ratio'] for m in fold_metrics]
        rmse_norms = [m['rmse_norm'] for m in fold_metrics if not np.isnan(m['rmse_norm'])]
        mae_norms = [m['mae_norm'] for m in fold_metrics if not np.isnan(m['mae_norm'])]
        mapes = [m['mape_percent'] for m in fold_metrics if not np.isnan(m['mape_percent'])]

        mean_metrics = {
            'r2_mean': float(np.mean(r2s)),
            'mse_mean': float(np.mean(mses)),
            'rmse_mean': float(np.mean(rmses)),
            'mae_mean': float(np.mean(maes)),
            'baseline_rmse_mean': float(np.mean(baseline_rmses)),
            'baseline_mae_mean': float(np.mean(baseline_maes)),
            'rmse_ratio_mean': float(np.mean([x for x in rmse_ratios if not np.isnan(x)])),
            'mae_ratio_mean': float(np.mean([x for x in mae_ratios if not np.isnan(x)])),
            'rmse_norm_mean': float(np.mean(rmse_norms)) if len(rmse_norms) > 0 else np.nan,
            'mae_norm_mean': float(np.mean(mae_norms)) if len(mae_norms) > 0 else np.nan,
            'mape_mean_percent': float(np.mean(mapes)) if len(mapes) > 0 else np.nan
        }

        std_metrics = {
            'r2_std': float(np.std(r2s)),
            'mse_std': float(np.std(mses)),
            'rmse_std': float(np.std(rmses)),
            'mae_std': float(np.std(maes)),
            'baseline_rmse_std': float(np.std(baseline_rmses)),
            'baseline_mae_std': float(np.std(baseline_maes)),
            'rmse_ratio_std': float(np.std([x for x in rmse_ratios if not np.isnan(x)])),
            'mae_ratio_std': float(np.std([x for x in mae_ratios if not np.isnan(x)]))
        }

        # treinar modelo final com todos os dados
        final_pipeline = build_model_pipeline_regressor(model_name, model_params=model_params_sanitized)
        t0_full = perf_counter()
        final_pipeline.fit(X_arr, y_arr)
        t1_full = perf_counter()
        fit_full_time = t1_full - t0_full

        # importância de features (tenta coef / feature_importances_ / permutation)
        method_used = None
        feature_importances_df = None

        if compute_importance:
            chosen = 'coef' if model_name.lower() in ('linear', 'lr', 'ridge', 'lasso') else 'permutation'
            if chosen == 'coef':
                try:
                    estimator = final_pipeline.named_steps['reg']
                    coefs = getattr(estimator, 'coef_', None)
                    if coefs is not None:
                        imp = np.abs(coefs)
                        if imp.ndim > 1:
                            imp = imp.mean(axis=0)
                        feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False).reset_index(drop=True)
                        method_used = 'coef'
                except Exception:
                    chosen = 'permutation'

            if chosen == 'permutation':
                try:
                    t0_perm = perf_counter()
                    perm = permutation_importance(final_pipeline, X_arr, y_arr, n_repeats=permutation_n_repeats, random_state=random_state, n_jobs=-1)
                    t1_perm = perf_counter()
                    permutation_time = t1_perm - t0_perm
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
                        print(c("Erro permutation importance:", RED), e)

        t_end_total = perf_counter()
        total_time = t_end_total - t_start_total

        timings = {
            'fit_time_per_fold': [ft['fit_time'] for ft in fold_timings],
            'predict_time_per_fold': [ft['predict_time'] for ft in fold_timings],
            'fit_full_time': fit_full_time,
            'total_time': total_time
        }

        metrics = {
            'validation_method': 'k_fold',
            'n_splits': n_splits,
            'folds': fold_metrics,
            'mean': mean_metrics,
            'std': std_metrics
        }

        if verbose:
            print()
            print(c("=" * 60, GREY))
            print(c("Estatísticas consolidadas (k-fold):", BOLD + GREEN))
            # print dicts but color key/value
            for k, v in metrics['mean'].items():
                print(c(f" - {k}: ", CYAN, bold=True) + c(f"{v}", GREEN))
            for k, v in metrics['std'].items():
                print(c(f" - {k}: ", CYAN, bold=True) + c(f"{v}", YELLOW))
            print(c("=" * 60, GREY))
            print(c("Interpretação rápida:", BOLD))
            print(c(" - MSE está em unidades ao quadrado (ex.: reais^2). Use RMSE/MAE para entender o erro na unidade original.", GREY))
            print(c(" - RMSE/mean(y) e MAE/mean(y) mostram o erro relativo em porcentagem.", GREY))
            print(c(" - As colunas baseline_* mostram o desempenho do previsor que sempre prevê a média do TREINO (baseline). Razões < 1 indicam melhora sobre o baseline.", GREY))

        return {
            'model': final_pipeline,
            'cv_models': cv_models,
            'metrics': metrics,
            'feature_names': feature_names,
            'feature_importances': feature_importances_df,
            'importance_method': method_used,
            'timings': timings
        }

    # --------------------------
    # HOLDOUT
    # --------------------------
    else:
        train_idx, test_idx = holdout_indices(n, train_size=train_size, random_state=random_state)
        X_train = X_arr[train_idx]
        X_test = X_arr[test_idx]
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]

        # Se pediu auto e é KNN, rodar GridSearch no conjunto de treino
        if do_auto:
            if verbose:
                print(c("Ajuste automático KNN solicitado — executando GridSearch no conjunto de treino (cv=3) ...", YELLOW))
            try:
                best_params_est, best_score, gs_obj = _autotune_for_model(X_train, y_train, auto_model_kind, cv=3)
                # atualizar model_params para usar os melhores parâmetros encontrados
                if isinstance(model_params, dict):
                    model_params = {**model_params, **best_params_est}
                    model_params.pop('auto', None)
                    model_params.pop('auto_grid', None)
                else:
                    model_params = best_params_est
                if verbose:
                    print(c(f"   -> Melhores params encontrados (GridSearch sobre treino): {best_params_est}", CYAN))
                    print(c(f"   -> best_score (neg MAE): {best_score:.4f}", CYAN))
            except Exception as e:
                if verbose:
                    print(c("Erro durante auto-tuning KNN:", RED), e)
                do_auto = False

        pipeline = build_model_pipeline_regressor(model_name, model_params=model_params)

        t0_fit = perf_counter()
        pipeline.fit(X_train, y_train)
        t1_fit = perf_counter()
        fit_time = t1_fit - t0_fit

        t0_pred = perf_counter()
        y_pred = pipeline.predict(X_test)
        t1_pred = perf_counter()
        predict_time = t1_pred - t0_pred

        r2 = float(r2_score(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))

        # baseline avaliado no teste: prever média do treino
        y_train_mean = float(np.mean(y_train))
        baseline_pred = np.full_like(y_test, fill_value=y_train_mean, dtype=float)
        baseline_mse = float(mean_squared_error(y_test, baseline_pred))
        baseline_rmse = float(np.sqrt(baseline_mse))
        baseline_mae = float(mean_absolute_error(y_test, baseline_pred))

        mse_ratio = mse / baseline_mse if baseline_mse != 0 else np.nan
        rmse_ratio = rmse / baseline_rmse if baseline_rmse != 0 else np.nan
        mae_ratio = mae / baseline_mae if baseline_mae != 0 else np.nan

        mean_y_test = float(np.mean(y_test)) if len(y_test) > 0 else np.nan
        rmse_norm = rmse / mean_y_test if mean_y_test != 0 else np.nan
        mae_norm = mae / mean_y_test if mean_y_test != 0 else np.nan

        eps = 1e-9
        if np.any(np.abs(y_test) > eps):
            mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + eps)))) * 100.0
        else:
            mape = np.nan

        metrics = {
            'validation_method': 'holdout',
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'baseline_mse': baseline_mse,
            'baseline_rmse': baseline_rmse,
            'baseline_mae': baseline_mae,
            'mse_ratio': mse_ratio,
            'rmse_ratio': rmse_ratio,
            'mae_ratio': mae_ratio,
            'rmse_norm': rmse_norm,
            'mae_norm': mae_norm,
            'mape_percent': mape
        }

        # importância das features
        feature_importances_df = None
        method_used = None
        permutation_time = None

        if compute_importance:
            chosen = 'coef' if model_name.lower() in ('linear', 'lr', 'ridge', 'lasso') else 'permutation'
            if chosen == 'coef':
                try:
                    estimator = pipeline.named_steps['reg']
                    coefs = getattr(estimator, 'coef_', None)
                    if coefs is not None:
                        imp = np.abs(coefs)
                        if imp.ndim > 1:
                            imp = imp.mean(axis=0)
                        feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False).reset_index(drop=True)
                        method_used = 'coef'
                except Exception:
                    chosen = 'permutation'

            if chosen == 'permutation':
                try:
                    t0_perm = perf_counter()
                    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=permutation_n_repeats, random_state=random_state, n_jobs=-1)
                    t1_perm = perf_counter()
                    permutation_time = t1_perm - t0_perm
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
                        print(c("Erro permutation importance:", RED), e)

        t_end_total = perf_counter()
        total_time = t_end_total - t_start_total

        timings = {
            'fit_time': fit_time,
            'predict_time': predict_time,
            'permutation_time': permutation_time,
            'total_time': total_time
        }

        if verbose:
            print(c("=" * 70, GREY))
            print(c(f"Modelo: {model_name}", BOLD + MAGENTA))
            print(c(f"Tamanho treino/teste: {len(X_train)}/{len(X_test)}", CYAN))
            # metrics line with colored keys/values
            print(c("R2: ", CYAN, bold=True) + c(f"{r2:.4f}", GREEN) + " | " +
                  c("MSE: ", CYAN, bold=True) + c(f"{mse:.4f}", YELLOW) + " (unidade^2) | " +
                  c("RMSE: ", CYAN, bold=True) + c(f"{rmse:.4f}", GREEN) + " | " +
                  c("MAE: ", CYAN, bold=True) + c(f"{mae:.4f}", GREEN))
            print(c("Baseline(mean-train) RMSE: ", CYAN, bold=True) + c(f"{baseline_rmse:.4f}", YELLOW) +
                  " | " + c("MAE: ", CYAN, bold=True) + c(f"{baseline_mae:.4f}", YELLOW))
            if mean_y_test and not np.isnan(mean_y_test):
                print(c("RMSE/mean(y_test): ", CYAN, bold=True) + c(f"{rmse_norm:.4f}", GREEN) + c(f" ({rmse_norm*100:.2f}%)", GREY) +
                      " | " + c("MAE/mean(y_test): ", CYAN, bold=True) + c(f"{mae_norm:.4f}", GREEN) + c(f" ({mae_norm*100:.2f}%)", GREY))
            if not np.isnan(mape):
                print(c("MAPE: ", CYAN, bold=True) + c(f"{mape:.2f}%", GREEN))
            print(c("Feature importance method: ", CYAN, bold=True) + c(str(method_used), MAGENTA))
            if feature_importances_df is not None:
                print(c("Top features:", BOLD + BLUE))
                # print small table header colored
                print(c(f"{'feature':<40} {'importance':>12}", BOLD + CYAN))
                for i, row in feature_importances_df.head(20).iterrows():
                    # choose color intensity by rank
                    rank_color = GREEN if i < 5 else (YELLOW if i < 10 else GREY)
                    print(rank_color + f"{row.iloc[0]:<40} {row.iloc[1]:>12.6f}" + RESET)
            print(c("Timings:", CYAN, bold=True), timings)
            print(c("=" * 70, GREY))

        return {
            'model': pipeline,
            'X-train': X_train,
            'X-test': X_test,
            'y-train': y_train,
            'y-test': y_test,
            'metrics': metrics,
            'feature_names': feature_names,
            'feature_importances': feature_importances_df,
            'importance_method': method_used,
            'timings': timings
        }
