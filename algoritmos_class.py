import pandas as pd
import numpy as np

from time import perf_counter

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# imports para o BlendingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split

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
        print(f"🚀 K-FOLD CROSS VALIDATION - {model_name.upper()}")
        print("=" * 70)
        print(f"📊 Dataset: {len(X_arr)} samples, {len(feature_names)} features")
        print(f"🎯 Target classes: {list(all_labels)}")
        print(f"📁 Folds: {n_splits}")
        print(f"🔧 Model parameters: {model_params}")
        print("-" * 70)

        fold_idx = 0
        for train_idx, val_idx in skf.split(X_arr, y_arr):
            fold_idx += 1

            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            # build pipeline for this fold (fresh instance)
            pipeline = build_model_pipeline(model_name, model_params=model_params)

            print(f"\n📁 FOLD {fold_idx}/{n_splits}")
            print(f"   📚 Train: {len(train_idx)} samples | Test: {len(val_idx)} samples")

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
            
            print(f"   ⚡ Fit: {fit_t:.4f}s | Predict: {pred_t:.4f}s")
            print(f"   📊 Metrics:")
            print(f"      ✅ Accuracy:  {acc:.4f}")
            print(f"      🎯 Precision: {prec:.4f}")
            print(f"      🔄 Recall:    {rec:.4f}")
            print(f"      ⚖️  F1-Score:  {f1:.4f}")

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
        print(f"\n🎯 TREINANDO MODELO FINAL COM TODOS OS DADOS")
        final_pipeline = build_model_pipeline(model_name, model_params=model_params)
        t0_full = perf_counter()
        final_pipeline.fit(X_arr, y_arr)
        t1_full = perf_counter()
        fit_full_time = t1_full - t0_full
        print(f"   ✅ Modelo final treinado em {fit_full_time:.4f}s")

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
                    print(f"   ✅ Importância por coeficientes calculada")
                except Exception as e:
                    print(f"   ❌ Erro em coef: {e} -> usando permutation")
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
                    print(f"   ✅ Permutation importance calculada ({permutation_time:.2f}s)")
                except Exception as e:
                    print(f"   ❌ Erro em permutation: {e}")

            if chosen == 'shap':
                if not _HAS_SHAP:
                    print("   ❌ SHAP não disponível")
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
                        print(f"   ✅ SHAP importance calculada ({shap_time:.2f}s)")
                    except Exception as e:
                        print(f"   ❌ Erro em SHAP: {e}")

        t_end_total = perf_counter()
        total_time = t_end_total - t_start_total

        # RESULTADOS FINAIS
        print("\n" + "=" * 70)
        print(f"🎯 RESULTADOS FINAIS - {model_name.upper()}")
        print("=" * 70)
        
        print(f"📊 Métricas por fold:")
        for i, metrics in enumerate(fold_metrics, 1):
            print(f"   Fold {i}: Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
                f"Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        print(f"\n📈 Estatísticas consolidadas:")
        print(f"   ✅ Accuracy:  {mean_metrics['accuracy_mean']:.4f} (±{std_metrics['accuracy_std']:.4f})")
        print(f"   🎯 Precision: {mean_metrics['precision_mean']:.4f} (±{std_metrics['precision_std']:.4f})")
        print(f"   🔄 Recall:    {mean_metrics['recall_mean']:.4f} (±{std_metrics['recall_std']:.4f})")
        print(f"   ⚖️  F1-Score:  {mean_metrics['f1_mean']:.4f} (±{std_metrics['f1_std']:.4f})")
        
        print(f"\n⏰ Tempos de execução:")
        print(f"   ⚡ Fit por fold:    {np.mean([ft['fit_time'] for ft in fold_timings]):.4f}s")
        print(f"   🎯 Predict por fold: {np.mean([ft['predict_time'] for ft in fold_timings]):.4f}s")
        print(f"   ✅ Modelo final:    {fit_full_time:.4f}s")
        print(f"   🕐 Total:           {total_time:.4f}s")
        
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
