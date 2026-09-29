# -*- coding: utf-8 -*-
# @Author  : hanyx9010@163.com
# @Time    : 2025/8/19
# @File    : evaluate_base_classifier.py
# @Description: DEA评价函数（供 main.py 调用）

import pandas as pd
import numpy as np
import pyomo.environ as pyo

from typing import Dict, List, Tuple, Any
from numpy.linalg import norm

# ===== DEA配置 =====
INPUT_COLS = ["fnr", "fpr"]
OUTPUT_COLS = ["auc", "f1_macro"]
EPS = 1e-6

def solve_dea_with_groups(df, input_cols, output_cols, target_idx, group_col="group"):
    """输出导向CCR乘子模型，带组约束"""
    groups = df[group_col].tolist()
    X = df[input_cols].values   # (n, m)
    Y = df[output_cols].values  # (n, s)
    n, m = X.shape
    s = Y.shape[1]

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(m)  # 投入维度
    model.R = pyo.RangeSet(s)  # 产出维度
    model.J = pyo.RangeSet(n)  # DMUs
    model.K = sorted(set(groups))

    # 参数
    X_param = {(i+1, j+1): X[j, i] for j in range(n) for i in range(m)}
    Y_param = {(r+1, j+1): Y[j, r] for j in range(n) for r in range(s)}
    group_map = {j+1: groups[j] for j in range(n)}

    # 决策变量
    model.u = pyo.Var(model.R, domain=pyo.NonNegativeReals)  # 产出权重
    model.v = pyo.Var(model.I, domain=pyo.NonNegativeReals)  # 投入权重

    # 目标函数
    model.obj = pyo.Objective(
        expr=sum(model.u[r]*Y_param[r, target_idx+1] for r in model.R),
        sense=pyo.maximize
    )

    # 归一化约束
    model.norm = pyo.Constraint(
        expr=sum(model.v[i]*X_param[i, target_idx+1] for i in model.I) == 1
    )

    # 组约束
    def group_feas_rule(m, k):
        return sum(
            sum(m.u[r]*Y_param[r, j] for r in m.R) -
            sum(m.v[i]*X_param[i, j] for i in m.I)
            for j in m.J if group_map[j]==k
        ) <= 0
    model.group_feas = pyo.Constraint(model.K, rule=group_feas_rule)

    # 求解
    solver = pyo.SolverFactory("mosek")
    solver.solve(model, tee=False)

    eff = pyo.value(model.obj)
    u_vals = [pyo.value(model.u[r]) for r in model.R]
    v_vals = [pyo.value(model.v[i]) for i in model.I]

    return eff, u_vals, v_vals


from typing import Dict, Any, Tuple
import numpy as np
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Reals, Objective, Constraint, SolverFactory, value
)


def solve_fdh_dual(df, input_cols, output_cols, k, group_col="group"):
    """
    依据 FDH 对偶（每个 h 各一套乘子）求解被评价 DMU k 的 z* 与效率 e=1/z*，并返回各 h 的乘子向量。
    模型：
        max z
        s.t.   x_k^T v_h = 1,                             ∀ h
               x_h^T v_h - (y_h - y_k)^T u_h ≥ z,         ∀ h
               u_{h,r} ≥ eps, v_{h,i} ≥ eps               ∀ h,r,i
        z free
    """
    X = df[input_cols].values  # (n, m)
    Y = df[output_cols].values  # (n, s)
    eps = 1e-5

    assert X.ndim == 2 and Y.ndim == 2, "X, Y 必须是二维数组"
    n, m = X.shape
    nY, s = Y.shape
    assert n == nY, "X 与 Y 的样本数量必须一致"
    assert 0 <= k < n, "k 超出 DMU 范围"

    # Pyomo 模型
    M = ConcreteModel()

    # 索引集合
    M.H = Set(initialize=range(n))  # DMU 索引 h
    M.I = Set(initialize=range(m))  # 输入指标 i
    M.R = Set(initialize=range(s))  # 输出指标 r

    # 数据参数（用字典承载便于 rule 调用）
    Xdict = {(h, i): float(X[h, i]) for h in range(n) for i in range(m)}
    Ydict = {(h, r): float(Y[h, r]) for h in range(n) for r in range(s)}
    M.X = Param(M.H, M.I, initialize=Xdict)
    M.Y = Param(M.H, M.R, initialize=Ydict)

    # 变量：对每个 h 各有一组 u_h(.), v_h(.)
    # 注意：严格正的“non-Archimedean”常数用 eps 近似
    M.u = Var(M.H, M.R, domain=pyo.NonNegativeReals, bounds=(eps, None))  # u_{h,r} ≥ eps
    M.v = Var(M.H, M.I, domain=pyo.NonNegativeReals, bounds=(eps, None))  # v_{h,i} ≥ eps
    M.z = Var(domain=pyo.NonNegativeReals)  # 自由变量

    # 约束 1：y_k^T u_h = 1, ∀h //  x_k^T v_h = 1
    def norm_rule(M, h):
        return sum(M.X[k, i] * M.v[h, i] for i in M.I) == 1.0

    M.NormEq = Constraint(M.H, rule=norm_rule)

    # 约束 2：(x_h - x_k)^T v_h - y_h^T u_h + z ≥ 0, ∀h   // x_h^T v_h - (y_h - y_k)^T u_h ≥ z,
    def dom_rule(M, h):
        term_x = sum(M.X[h, i] * M.v[h, i] for i in M.I)
        term_y = sum((M.Y[h, r] - M.Y[k, r]) * M.u[h, r] for r in M.R)
        return M.z <= term_x - term_y

    M.Dominance = Constraint(M.H, rule=dom_rule)

    # 约束 3: 保证域 ***** 这里应该可以传入参数
    def fnr_fpr_ratio_lower_rule(M, h):
        return M.v[h, 0] >= M.v[h, 1]

    M.fnr_fpr_lower_con = pyo.Constraint(M.H, rule=fnr_fpr_ratio_lower_rule)

    # 目标：min z  //  max z
    M.Obj = Objective(expr=M.z, sense=pyo.maximize)

    # 求解
    solver = SolverFactory("mosek")
    res = solver.solve(M, tee=False)

    z_star = value(M.z)
    # eff = 1.0 / z_star if z_star > 0 else float("nan")
    eff = z_star

    # 提取乘子
    u_map: Dict[int, np.ndarray] = {
        h: np.array(
            [
                (M.u[h, r].value if M.v[h, r].value is not None else None)
                for r in M.R
            ],
            dtype=object
        )
        for h in M.H
    }
    u_df = pd.DataFrame(u_map)
    v_map: Dict[int, np.ndarray] = {
        h: np.array(
            [
                (M.v[h, i].value if M.v[h, i].value is not None else None)
                for i in M.I
            ],
            dtype=object  # 用 object 保留 None
        )
        for h in M.H  # 跳过 h=k 的冗余变量
    }
    v_df = pd.DataFrame(v_map)

    return eff, u_df, v_df

def efficiency_evaluate(metrics_df, group_col="group", params=None):
    """核心函数：输入指标DataFrame，输出DEA评价结果"""
    df = metrics_df.copy()

    # 防零
    for c in INPUT_COLS + OUTPUT_COLS:
        if c in df.columns:
            df[c] = np.clip(df[c].values, EPS, None)
        else:
            raise ValueError(f"缺少必要列 {c}，请确保 compute_metrics 输出这些指标")

    # 批量求解
    records = []
    eff_results = []
    for idx in range(len(df)):
        eff, u_map_idx, v_map_idx = solve_fdh_dual(df, INPUT_COLS, OUTPUT_COLS, idx, group_col)
        eff_results.append({
            "k": idx,
            "learner_id": df.loc[idx, "learner_id"],
            "eff": eff,
            "u_map": u_map_idx,  # {h: u_h 或 None}
            "v_map": v_map_idx,  # {h: v_h 或 None}
        })
        row = {"learner_id": df.loc[idx, "learner_id"],
               group_col: df.loc[idx, group_col],
               "efficiency": eff}
        records.append(row)
    dea_df = pd.DataFrame(records)

    weights_k = {}
    for rec in eff_results:
        lid = rec["learner_id"]
        u_map_k = rec["u_map"]
        v_map_k = rec["v_map"]
        weights_k[lid] = pack_weight_vector(
            u_map_k, v_map_k,
            n=u_map_k.shape[0], s=u_map_k.shape[1], m=v_map_k.shape[1], fill=0.0
        )
    C, keys = cosine_similarity_from_weights(weights_k, l2_normalize=True)
    diversity = pd.DataFrame(C, index=keys, columns=keys)

    return dea_df, diversity


def extract_binding_and_support_for_k(
        X: np.ndarray,  # 形状 (n, m)
        Y: np.ndarray,  # 形状 (n, s)
        u_map_k: Dict[int, np.ndarray],  # {h -> u_h ∈ R^s}（针对被评 k 求解得到的那一套乘子）
        v_map_k: Dict[int, np.ndarray],  # {h -> v_h ∈ R^m}
        tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    对固定的被评DMU k，计算每个 h 的约束剩余与 binding 标志，并返回支撑集。
    约束形式：u_h^T y_h - v_h^T x_h <= 0 （你的FDH对偶截图/实现中就是这类“可行性”约束）
    slack_h := v_h·x_h - u_h·y_h >= 0；当 slack_h ≈ 0 视为 binding

    返回:
        slacks:  shape (n,)   各 h 的 slack_h
        is_binding: shape (n,) 布尔数组
        support_h: List[int]   {h | is_binding[h] = True}
    """
    n, m = X.shape
    _, s = Y.shape
    slacks = np.full(n, np.nan, dtype=float)
    is_binding = np.zeros(n, dtype=bool)
    for h in range(n):
        u_h = u_map_k.get(h, None)
        v_h = v_map_k.get(h, None)
        if any(val is None for val in u_h) or any(val is None for val in v_h):
            # 例如你前面提到的 k==h 不可行时置 None 的情形
            continue
        # v_h·x_h - u_h·y_h
        val = float(np.dot(v_h, X[h, :]) - np.dot(u_h, Y[h, :]))
        slacks[h] = val
        is_binding[h] = (abs(val) <= tol)
    support_h = np.where(is_binding)[0].tolist()
    return slacks, is_binding, support_h


def build_support_incidence(
        per_k_results: List[Dict[str, Any]],
        n: int,
        tol: float = 1e-4
) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """
    汇总所有被评 k 的支撑集，形成 n×n 的0/1“参照发生矩阵”S：
      S[k, h] = 1  表示 h 是 k 的支撑/参照（约束在 h 处 binding）

    per_k_results: 列表，元素形如
        {
          "k": int,
          "u_map": {h: u_h (np.ndarray length s) or None},
          "v_map": {h: v_h (np.ndarray length m) or None},
          "X": X, "Y": Y  # 也可统一外部传入，这里为了接口自由度
        }
    """
    S = np.zeros((n, n), dtype=int)
    support_dict: Dict[int, List[int]] = {}
    for rec in per_k_results:
        k = rec["k"]
        X = rec["X"];
        Y = rec["Y"]
        u_map_k = rec["u_map"];
        v_map_k = rec["v_map"]
        _, _, supp = extract_binding_and_support_for_k(X, Y, u_map_k, v_map_k, tol=tol)
        support_dict[k] = supp
        if len(supp) > 0:
            S[k, np.array(supp, dtype=int)] = 1
    S_df = pd.DataFrame(S, index=[f"k={i}" for i in range(n)], columns=[f"h={j}" for j in range(n)])
    return S_df, support_dict


# -----------------------------------
# 2) 多样性矩阵（Jaccard 与 余弦相似）
# -----------------------------------
def jaccard_similarity_from_support(S_bin: np.ndarray) -> np.ndarray:
    """
    输入：二值参照发生矩阵 S (n×n), 第 i 行表示 DMU i 的支撑集指示
    输出：Jaccard 相似度矩阵 J (n×n)，J[i,j] = |Si ∩ Sj| / |Si ∪ Sj|
    多样性 = 1 - J
    """
    n = S_bin.shape[0]
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        Si = S_bin[i]
        for j in range(i, n):
            Sj = S_bin[j]
            inter = np.sum((Si == 1) & (Sj == 1))
            union = np.sum((Si == 1) | (Sj == 1))
            J[i, j] = J[j, i] = (inter / union) if union > 0 else 0.0
    return J


def cosine_similarity_from_weights(
        weights_k: Dict[int, np.ndarray],
        l2_normalize: bool = True
) -> np.ndarray:
    """
    基于“乘子权重矩阵”计算余弦相似度。
    weights_k: {k -> w_k}，w_k 是把 (u_map_k, v_map_k) 统一展开后的向量。
               例如先按 h 逐行拼接 u_h 再拼接 v_h：w_k = vec([u_0,...,u_{n-1}, v_0,...,v_{n-1}])
               （确保对所有 k 维度一致；没有值的用0或np.nan->0）
    返回：Cosine 相似度矩阵 C (n×n)，多样性可取 1 - C。
    """
    keys = sorted(weights_k.keys())
    W = []
    for k in keys:
        w = weights_k[k]
        if w is None:
            W.append(np.zeros(1, dtype=float))
        else:
            W.append(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0))
    W = np.vstack(W)  # (n, d)
    if l2_normalize:
        norms = np.maximum(norm(W, axis=1, keepdims=True), 1e-12)
        W = W / norms
    C = W @ W.T  # 归一化后内积即为余弦
    # 数值清理
    C = np.clip(C, -1.0, 1.0)
    return C, keys


def pack_weight_vector(
        u_map_k: Dict[int, np.ndarray],
        v_map_k: Dict[int, np.ndarray],
        n: int,
        s: int,
        m: int,
        fill: float = 0.0
) -> np.ndarray:
    """
    把某个 k 的 {h->u_h}, {h->v_h} 展为固定长度向量，便于做余弦相似。
    展开规则（可与队里固定约定）：
      w_k = [u_0(1..s), u_1(1..s), ..., u_{n-1}(1..s), v_0(1..m), ..., v_{n-1}(1..m)]
    若某 h 的 u_h 或 v_h 为 None，则用 fill 填充（建议 0）。
    """
    buf = []
    for h in range(n):
        uh = u_map_k.get(h, None)
        if uh is None:
            buf.append(np.full(s, fill, dtype=float))
        else:
            buf.append(np.asarray(uh, dtype=float).reshape(-1))
    for h in range(n):
        vh = v_map_k.get(h, None)
        if vh is None:
            buf.append(np.full(m, fill, dtype=float))
        else:
            buf.append(np.asarray(vh, dtype=float).reshape(-1))
    return np.concatenate(buf, axis=0)

def build_diversity_matrices(
    per_k_results: List[Dict[str, Any]],
    n: int, m: int, s: int,
    tol: float = 1e-4
) -> Dict[str, Any]:
    """
    输入 per_k_results（包含每个 k 的 u_map, v_map, 以及同一组 X,Y），
    输出：
      - 支撑发生矩阵 S_df
      - Jaccard 相似/多样性矩阵（DataFrame）
      - 余弦相似/多样性矩阵（DataFrame）
    """
    # 参照发生矩阵 + 支撑集
    S_df, support_dict = build_support_incidence(per_k_results, n=n, tol=tol)
    S_bin = S_df.values.astype(int)
    # Jaccard
    J = jaccard_similarity_from_support(S_bin)
    J_df = pd.DataFrame(J, index=S_df.index, columns=S_df.index)
    J_div_df = 1.0 - J_df

    # 余弦（基于展开的权重）
    weights_k = {}
    for rec in per_k_results:
        k = rec["k"]
        u_map_k = rec["u_map"]; v_map_k = rec["v_map"]
        weights_k[k] = pack_weight_vector(u_map_k, v_map_k, n=n, s=s, m=m, fill=0.0)
    C, keys = cosine_similarity_from_weights(weights_k, l2_normalize=True)
    # 对齐索引名
    idx = [f"k={k}" for k in keys]
    C_df = pd.DataFrame(C, index=idx, columns=idx)
    C_div_df = 1.0 - C_df

    return {
        "support_matrix": S_df,          # 0/1，谁是参照
        "support_sets": support_dict,    # 每个k的参照索引列表
        "jaccard_sim": J_df,
        "jaccard_div": J_div_df,
        "cosine_sim": C_df,
        "cosine_div": C_div_df,
    }

if __name__ == "__main__":
    # 一个简小例子（随机构造，演示用）
    rng = np.random.default_rng(0)
    n = 20

    # Generate strictly positive inputs using a lognormal distribution
    x1 = rng.lognormal(mean=0.0, sigma=0.7, size=n)
    x2 = rng.lognormal(mean=0.0, sigma=0.7, size=n)

    y1 = rng.lognormal(mean=0.0, sigma=0.7, size=n)
    y2 = rng.lognormal(mean=0.0, sigma=0.7, size=n)

    df = pd.DataFrame({
        "DMU": [f"DMU_{i:03d}" for i in range(1, n + 1)],
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2,
    })

    input_cols = ["x1", "x2"]
    output_cols = ["y1", "y2"]

    eff_results = []
    for idx in range(n):
        eff, u_map_idx, v_map_idx = solve_fdh_dual(df, input_cols, output_cols, idx)
        eff_results.append({
            "k": idx,
            "eff": eff,
            "u_map": u_map_idx,  # {h: u_h 或 None}
            "v_map": v_map_idx,  # {h: v_h 或 None}
        })
    res = build_diversity_matrices(eff_results, n=X.shape[0], m=X.shape[1], s=Y.shape[1], tol=1e-4)

