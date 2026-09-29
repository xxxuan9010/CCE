# -*- coding: utf-8 -*-
# @Author  : hanyx9010@163.com
# @Time    : 2025/8/19 18:36
# @File    : utils.py
# @Description    :
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.multiclass import type_of_target
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from typing import Tuple, Any, Union
from imblearn.pipeline import Pipeline


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = ['target', 'label', 'y', 'class']
    for c in candidates:
        if c in df.columns:
            return c
    # 默认最后一列为标签
    return df.columns[-1]

def ensure_binary_or_multiclass(y: np.ndarray) -> Tuple[np.ndarray, bool]:
    t = type_of_target(y)
    if t not in ('binary', 'multiclass'):
        raise ValueError(f"不支持的任务类型: {t}")
    return y, (t == 'binary')

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, roc_auc_score, average_precision_score,
    confusion_matrix
)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    positive_label: Optional[int] = None
) -> Dict[str, float]:
    """
    二分类指标（仅针对指定正类计算混淆类指标与AUC）
    约定：y_proba若是一维则为“正类”的概率；若是二维且为二列，则默认第二列为正类概率
    （建议在外层确保传入的就是正类概率的一维向量，最稳妥）
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 正类标签（默认取y_true中的较大者，常见为1）
    pos = int(np.max(y_true)) if positive_label is None else positive_label

    # 基础分类指标
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    # 二分类主F1：以正类为pos_label
    f1_bin = f1_score(y_true, y_pred, pos_label=pos, average="binary", zero_division=0)
    # 额外给出macro/weighted，便于横向比较（可按需去掉）
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos, zero_division=0
    )

    # 混淆矩阵派生（仅以指定正类计算）
    yt = (y_true == pos).astype(int)
    yp = (y_pred == pos).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity/recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
    gmean = np.sqrt(tpr * tnr)

    # 概率相关（保持稳健降级）
    auc = np.nan
    pr_auc = np.nan
    if y_proba is not None:
        p = np.asarray(y_proba)
        # 允许传入一维正类概率或二维两列概率
        if p.ndim == 1:
            p_pos = p.ravel()
        elif p.ndim == 2 and p.shape[1] == 2:
            # 约定第2列为正类概率；若你的pipeline不同，请在外层改造成一维正类概率后再传入
            p_pos = p[:, 1]
        else:
            # 其他形状不支持（保持nan）
            p_pos = None

        if p_pos is not None:
            try:
                auc = roc_auc_score(yt, p_pos)
            except Exception:
                auc = np.nan
            try:
                pr_auc = average_precision_score(yt, p_pos)
            except Exception:
                pr_auc = np.nan

    return {
        # 主键（不平衡任务常用）
        "acc": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1": float(f1_bin),
        "precision": float(precision),
        "recall": float(recall),
        "gmean": float(gmean),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }

def _ensure_cost_matrix(cost_matrix: Optional[np.ndarray]) -> np.ndarray:
    """
    返回形如 [[C_TN, C_FP],[C_FN, C_TP]] 的 2x2 成本矩阵。
    若未提供，则默认关注 C_FP=1, C_FN=1（等成本）。
    """
    if cost_matrix is None:
        return np.array([[0.0, 1.0],
                         [3.0, 0.0]], dtype=float)
    cm = np.asarray(cost_matrix, dtype=float)
    assert cm.shape == (2, 2), "cost_matrix must be shape (2,2)"
    return cm


def _class_weight_from_cost(cost_matrix: Optional[np.ndarray], neg_label=0, pos_label=1):
    """
    将误判代价映射为 class_weight：neg ∝ C_FP，pos ∝ C_FN
    """
    cm = _ensure_cost_matrix(cost_matrix)
    C_fp = float(cm[0, 1])
    C_fn = float(cm[3, 0])
    s = (C_fp + C_fn)
    if s <= 0:
        return "balanced"
    return {neg_label: 2.0 * C_fp / s, pos_label: 2.0 * C_fn / s}


def _sample_weight_from_cost(y, cost_matrix: Optional[np.ndarray], pos_label=1):
    """
    生成样本权重：正类样本权重 ∝ C_FN；负类样本权重 ∝ C_FP
    """
    cm = _ensure_cost_matrix(cost_matrix)
    C_fp = float(cm[0, 1])
    C_fn = float(cm[1, 0])
    y = np.asarray(y)
    w = np.where(y == pos_label, C_fn, C_fp).astype(float)
    if w.sum() <= 0:
        w = np.ones_like(w, dtype=float)
    return w


def _optimal_threshold_from_cost(cost_matrix: Optional[np.ndarray]) -> float:
    """
    成本敏感贝叶斯判别阈值:
    预测正类当 P(positive|x) >= C_FP / (C_FP + C_FN)
    """
    cm = _ensure_cost_matrix(cost_matrix)
    C_fp = float(cm[0, 1])
    C_fn = float(cm[1, 0])
    denom = C_fp + C_fn
    return 0.5 if denom <= 0 else (C_fp / denom)


class CostSensitiveAdaBoost(BaseEstimator, ClassifierMixin):
    """
    AdaBS 成本初始化版本︰不继承 AdaBoostClassifier
    在 fit 时按成本生成 sample_weight 再调用内部的 AdaBoostClassifier
    """
    def __init__(self,
                 estimator=None,
                 n_estimators=200,
                 learning_rate=0.5,
                 random_state=0,
                 cost_matrix=None,
                 pos_label=1):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.cost_matrix = cost_matrix
        self.pos_label = pos_label
        self._clf = None

    def fit(self, X, y):
        base = self.estimator
        if base is None:
            base = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)

        # 内部真实分类器
        self._clf = AdaBoostClassifier(
            estimator=clone(base),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        sw = _sample_weight_from_cost(y, self.cost_matrix, pos_label=self.pos_label)
        sw = sw / sw.sum() if sw.sum() > 0 else sw
        self._clf.fit(X, y, sample_weight=sw)
        return self

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    def predict(self, X):
        return self._clf.predict(X)

    def decision_function(self, X):
        # 若需要与上游接口兼容
        proba = self.predict_proba(X)[:, 1]
        thr = _optimal_threshold_from_cost(self.cost_matrix)
        return proba - thr


def baseline_models(
    cost_matrix: Optional[np.ndarray] = None,
    pos_label: Union[int, str] = 1,
    random_state: int = 0,
) -> Dict[str, Any]:
    models = {
        # 标准模型
        'RF': RandomForestClassifier(random_state=0),
        'GB': GradientBoostingClassifier(random_state=0),
        'AdaBoost': AdaBoostClassifier(random_state=0),

        # 基于采样的管道
        'SMOTE+RF': Pipeline([
            ('smote', SMOTE(random_state=0)),
            ('clf', RandomForestClassifier(random_state=0))
        ]),
        'ADASYN+RF': Pipeline([
            ('adasyn', ADASYN(random_state=0)),
            ('clf', RandomForestClassifier(random_state=0))
        ]),
        'Under+RF': Pipeline([
            ('under', RandomUnderSampler(random_state=0)),
            ('clf', RandomForestClassifier(random_state=0))
        ]),

        # 阈值移动
        'ThresholdWrapperRF': CostSensitiveThresholdWrapper(
            base_estimator=RandomForestClassifier(random_state=0),
            cost_matrix=cost_matrix,   # 需要指定成本矩阵
            pos_label=pos_label,
            use_cost_sample_weight=True,
        ),

        'ThresholdWrapperGB': CostSensitiveThresholdWrapper(
            base_estimator=GradientBoostingClassifier(random_state=0),
            cost_matrix=cost_matrix,
            pos_label=pos_label,
            use_cost_sample_weight=True,
        ),

        # —— AdaCost —— #
        'AdaBS': CostSensitiveAdaBoost(
            n_estimators=300,
            learning_rate=0.5,
            random_state=random_state,
            cost_matrix=cost_matrix,
            pos_label=pos_label,
        ),

        # 不平衡专用集成方法
        'BalancedRF': BalancedRandomForestClassifier(n_estimators=10, random_state=0),
        'EasyEnsemble': EasyEnsembleClassifier(n_estimators=10, random_state=0),
        'RUSBoost': RUSBoostClassifier(n_estimators=10, random_state=0)
    }
    return models

class CostSensitiveThresholdWrapper(BaseEstimator, ClassifierMixin):
    """
    将任意支持 predict_proba 的二分类器，改造成“按成本阈值决策”的模型。
    同时在 fit 时可使用基于成本的 sample_weight。
    """
    def __init__(self, base_estimator: BaseEstimator,
                 cost_matrix: Optional[np.ndarray] = None,
                 pos_label: Union[int, str] = 1,
                 use_cost_sample_weight: bool = True):
        self.base_estimator = base_estimator
        self.cost_matrix = None if cost_matrix is None else np.asarray(cost_matrix, dtype=float)
        self.pos_label = pos_label
        self.use_cost_sample_weight = use_cost_sample_weight
        self._clf = None
        self._threshold = 0.5

    def fit(self, X, y):
        self._clf = clone(self.base_estimator)
        if self.use_cost_sample_weight:
            sw = _sample_weight_from_cost(y, self.cost_matrix, pos_label=self.pos_label)
            self._clf.fit(X, y, sample_weight=sw)
        else:
            self._clf.fit(X, y)
        self._threshold = _optimal_threshold_from_cost(self.cost_matrix)
        return self

    def predict_proba(self, X):
        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(X)
        # 若基学习器不支持 predict_proba，退化为“置信度近似”
        # 这里简单返回二列：neg=1-pos, pos=pos
        scores = self._clf.decision_function(X)
        # MinMax 标定到 [0,1]
        smin, smax = np.min(scores), np.max(scores)
        if smax - smin <= 1e-12:
            p = np.full_like(scores, 0.5, dtype=float)
        else:
            p = (scores - smin) / (smax - smin)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        pos_prob = proba[:, 1]
        return (pos_prob >= self._threshold).astype(int)

    # 兼容性：若需要 decision_function
    def decision_function(self, X):
        proba = self.predict_proba(X)[:, 1]
        return proba - self._threshold


def fit_predict_estimator(est, X_tr, y_tr, X_te) -> Tuple[np.ndarray, np.ndarray]:
    est.fit(X_tr, y_tr)
    y_pred = est.predict(X_te)
    y_proba = None
    if hasattr(est, "predict_proba"):
        try:
            y_proba = est.predict_proba(X_te)
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba = y_proba[:,1]
        except Exception:
            y_proba = None
    return y_pred, y_proba


# ===== 三段式划分：Train_all/Test，然后 Train_base/Eval_for_DEA =====
def split_three_way(X, y, test_size=0.2, eval_frac_of_train=0.25, seed=42):
    X_tr_all, X_te, y_tr_all, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_tr_base, X_eval, y_tr_base, y_eval = train_test_split(
        X_tr_all, y_tr_all, test_size=eval_frac_of_train, stratify=y_tr_all, random_state=seed
    )
    return X_tr_base, y_tr_base, X_eval, y_eval, X_tr_all, y_tr_all, X_te, y_te

# ===== 简单的效率加权集成（若 select 模块未返回成品预测时使用）=====
def efficiency_weighted_predict(selected_ids, eff_map, base_list, X):
    # 汇总概率或硬投票 → 得分矩阵
    # base_list 每个元素需包含 {'learner_id','estimator'}，且 estimator 支持 predict / predict_proba
    classes_ = None
    # 首先确定类集合
    for r in base_list:
        if r["learner_id"] in selected_ids:
            est = r["estimator"]
            try:
                proba = est.predict_proba(X)
                n_classes = proba.shape[1]
                classes_ = np.arange(n_classes)  # 假定类编码为0..C-1
                break
            except Exception:
                # 用一次硬预测推断类别集合
                classes_ = np.unique(est.predict(X))
                break
    if classes_ is None:
        # 兜底：从所有基学习器硬预测联合确定
        labs = set()
        for r in base_list:
            labs |= set(r["estimator"].predict(X).tolist())
        classes_ = np.array(sorted(list(labs)))

    score = np.zeros((X.shape[0], len(classes_)))
    for r in base_list:
        lid = r["learner_id"]
        if lid not in selected_ids:
            continue
        w = eff_map.get(lid, 1.0)
        est = r["estimator"]
        try:
            proba = est.predict_proba(X)
            if proba.ndim == 1:
                # 二分类单列→转两列（假设正类为最大标签）
                p = np.zeros((len(proba), len(classes_)))
                pos_idx = np.argmax(classes_)
                p[:, pos_idx] = proba
                p[:, 1 - pos_idx] = 1 - proba
                score += w * p
            else:
                # 对齐类别索引（若有必要可在训练阶段保存 classes_）
                if proba.shape[1] == len(classes_):
                    score += w * proba
                else:
                    # 简单对齐：截断/填零
                    p = np.zeros_like(score)
                    c = min(proba.shape[1], score.shape[1])
                    p[:, :c] = proba[:, :c]
                    score += w * p
        except Exception:
            # 硬投票
            yp = est.predict(X)
            oh = np.zeros_like(score)
            for i, lab in enumerate(yp):
                oh[i, np.where(classes_ == lab)[0][0]] = 1
            score += w * oh
    # 归一化为“类概率感”的分数（用于AUC）
    denom = np.sum(score, axis=1, keepdims=True) + 1e-12
    p = score / denom
    y_pred = classes_[np.argmax(p, axis=1)]
    return y_pred, p
