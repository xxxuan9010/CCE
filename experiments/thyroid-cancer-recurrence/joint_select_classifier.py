# -*- coding: utf-8 -*-
# @Author  : hanyx9010@163.com
# @Time    : 2025/8/19
# @File    : joint_select_classifier.py
# @Description: 联合选择基学习器（供 main.py 调用）

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pyomo.environ as pyo


def optimize_selection(
    dea_df: pd.DataFrame,
    diversity_matrix=None,
    A_cov=None,
    k_max: int = 5,
    rho: float = 0.0,
    group_min: dict = None,
    solver: str = "mosek"
):
    """
    从 DEA 结果中选择最优子集
    dea_df: DataFrame, 至少包含 ['learner_id','group','efficiency', u/v权重列]
    diversity_matrix: 可选的外部多样性矩阵（dict 或 DataFrame），若为 None 则自动基于 u/v 权重计算
    A_cov: （暂未使用，留接口）
    k_max: 最多选择的基学习器数
    rho: （暂未使用，留接口）
    group_min: dict {group: min_num}，如果 None 默认每组≥1
    solver: 求解器名称
    """
    assert "learner_id" in dea_df.columns
    assert "group" in dea_df.columns
    assert "efficiency" in dea_df.columns

    # === 效率归一化 ===
    eff_raw = dea_df.set_index("learner_id")["efficiency"]
    theta_min, theta_max = eff_raw.min(), eff_raw.max()
    # eff_norm = (theta_max - eff_raw) / (theta_max - theta_min + 1e-12)
    eff_norm = (eff_raw - theta_min) / (theta_max - theta_min + 1e-12)
    efficiency = eff_norm.to_dict()

    # === 多样性矩阵 ===
    if diversity_matrix is None:
        weight_cols = [c for c in dea_df.columns if c.startswith("u[") or c.startswith("v[")]
        W = dea_df[weight_cols].values
        W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        sim = cosine_similarity(W)
        ids = dea_df["learner_id"].tolist()
        diversity = {}
        pairs = []
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                d_ij = 1.0 - float(sim[i, j])
                diversity[(ids[i], ids[j])] = d_ij
                pairs.append((ids[i], ids[j]))
    else:
        ids = dea_df["learner_id"].tolist()
        diversity = {(i, j): diversity_matrix.loc[i, j] for i in ids for j in ids if i < j}
        pairs = list(diversity.keys())

    # === 分组信息 ===
    groups = {g: sub["learner_id"].tolist() for g, sub in dea_df.groupby("group")}
    if group_min is None:
        group_min = {g: 1 for g in groups}

    # === Pyomo 模型 ===
    model = pyo.ConcreteModel()
    learners = list(efficiency.keys())
    model.K = pyo.Set(initialize=learners)
    model.P = pyo.Set(dimen=2, initialize=pairs)
    model.G = pyo.Set(initialize=list(groups.keys()))

    theta = efficiency
    d = diversity
    lambda_div = 0   # 多样性权重（可调）

    model.z = pyo.Var(model.K, domain=pyo.Binary)  # 是否选择学习器
    model.y = pyo.Var(model.P, domain=pyo.Binary)  # 是否同时选择 (i,j)

    # 目标
    model.obj = pyo.Objective(
        expr=sum(theta[k]*model.z[k] for k in model.K) +
             lambda_div * sum(d[(i,j)]*model.y[(i,j)] for (i,j) in model.P),
        sense=pyo.maximize
    )

    # 规模约束
    model.size_con = pyo.Constraint(expr=sum(model.z[k] for k in model.K) <= k_max)

    # 逻辑约束
    def yz_le_i(m, i, j): return m.y[(i,j)] <= m.z[i]
    def yz_le_j(m, i, j): return m.y[(i,j)] <= m.z[j]
    def yz_ge(m, i, j):   return m.y[(i,j)] >= m.z[i] + m.z[j] - 1
    model.yz1 = pyo.Constraint(model.P, rule=yz_le_i)
    model.yz2 = pyo.Constraint(model.P, rule=yz_le_j)
    model.yz3 = pyo.Constraint(model.P, rule=yz_ge)

    # 组下限
    # def group_min_rule(m, g):
    #     return sum(m.z[k] for k in groups[g]) >= group_min[g]
    # model.group_min = pyo.Constraint(model.G, rule=group_min_rule)

    # === 求解 ===
    solver_obj = pyo.SolverFactory(solver)
    solver_obj.solve(model, tee=False)

    selected_ids = [k for k in model.K if pyo.value(model.z[k]) > 0.5]
    final_selection = dea_df[dea_df["learner_id"].isin(selected_ids)].copy()

    return {
        "selected_ids": selected_ids,
        "final_selection": final_selection
    }
