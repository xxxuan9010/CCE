# -*- coding: utf-8 -*-
# @Author  : hanyx9010@163.com
# @Time    : 2025/8/19 19:49
# @File    : train_base_classifier.py
# @Description: 构造并训练“抽样法驱动的异质基学习器池”

# 接口契约：
# def train_base_learners(X_train_base, y_train_base, random_state=42) -> list[dict]
# 返回每个基学习器字典：
#   - 'learner_id': 唯一ID（采样法_比例_模型_种子）
#   - 'group':      同质组ID（采样法_比例_模型）
#   - 'estimator':  已拟合的 sklearn/imbalance-learn Pipeline，可 predict / predict_proba

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# 预处理与基分类器
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

def _make_sampler(name: str, ratio: float, seed: int):
    """
    根据名称与比例构造采样器。
    ratio 解释遵循 imbalanced-learn 的 sampling_strategy 语义（针对少数类目标比例等）。
    """
    name = name.upper()
    if name == "SMOTE":
        return SMOTE(sampling_strategy=ratio, random_state=seed)
        # return SMOTE(random_state=seed)
    if name == "ADASYN":
        return ADASYN(sampling_strategy=ratio, random_state=seed)
        # return ADASYN(random_state=seed)
    if name in ("RUS", "RANDOMUNDERSAMPLER", "UNDER"):
        return RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
        # return ADASYN(random_state=seed)
    raise ValueError(f"Unknown sampler: {name}")

def _base_models(seed: int) -> List[Tuple[str, object]]:
    """
    给出一组轻量但异质的基分类器。可按需增删。
    注意：选择 predict_proba 可用的模型优先（便于后续 AUC/加权投票）
    """
    return [
        # ("LR", LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)),
        # ("DT", DecisionTreeClassifier(random_state=seed),)
        ("RF", RandomForestClassifier(random_state=seed)),
        # ("GB", GradientBoostingClassifier(random_state=seed)),
        # ("AdaBoost", AdaBoostClassifier(random_state=seed)),
        # ("KNN", KNeighborsClassifier(n_neighbors=10, weights="distance")),
    ]

def train_base_learners(
    X_train_base: np.ndarray,
    y_train_base: np.ndarray,
    random_state: int = 42,
    samplers: List[str] = ("SMOTE", "RUS", "ADASYN"),
    ratios: List[float] = (0.6, 0.7, 0.8, 0.9, 1.0),
    n_per_group: int = 3,  # 每个“方法-比例-模型”组内，用不同随机种子构造若干同质学习器
) -> List[Dict]:
    """
    训练“抽样法驱动的异质基学习器池”。
    返回：list[dict]，每个元素至少包含:
        - learner_id: 唯一ID，例如 'SMOTE_0.7_RF_003'
        - group:      同质组，例如 'SMOTE_0.7_RF'
        - estimator:  已拟合的 ImbPipeline，可直接用于预测/评估
    说明：
        - 预处理仅做 StandardScaler（数值特征），因为 main.py 传入的是 numpy 数组；
        - 采样在标准化之后、分类器之前（imblearn Pipeline 顺序：scale -> sample -> clf）
        - 为了复现实验的“同质组”，同一(采样法,比例,模型)下改变 seed 构造多个基学习器
    """
    rng = np.random.RandomState(random_state)

    # 标准化作为通用数值预处理；对于稀疏/大规模数据可关掉 with_mean
    scaler = StandardScaler(with_mean=True)

    base_list: List[Dict] = []

    # 遍历“采样方法 × 采样比例 × 基模型”
    for samp in samplers:
        for ratio in ratios:
            for mdl_name, mdl in _base_models(seed=random_state):
                group_id = f"{samp}_{ratio}_{mdl_name}"

                # 组内生成 n_per_group 个同质学习器（仅改变随机种子；KNN无随机性但仍可放入组以保持接口一致）
                for i in range(n_per_group):
                    seed_i = int(rng.randint(0, 10_000))
                    # 深拷贝/复写随机种子（对有 random_state 的模型覆盖）
                    clf = mdl
                    if hasattr(clf, "random_state"):
                        try:
                            clf = clf.__class__(**{**clf.get_params(), "random_state": seed_i})
                        except Exception:
                            pass

                    sampler = _make_sampler(samp, ratio, seed_i)

                    # imblearn 的 Pipeline：先缩放、再采样、再分类
                    pipe = ImbPipeline(steps=[
                        ("scale", scaler),
                        ("samp", sampler),
                        ("clf", clf),
                    ])

                    # 拟合
                    pipe.fit(X_train_base, y_train_base)

                    learner_id = f"{group_id}_{seed_i:04d}"
                    base_list.append({
                        "learner_id": learner_id,
                        "group": group_id,
                        "estimator": pipe,
                    })

    return base_list
