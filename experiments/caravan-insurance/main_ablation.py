# -*- coding: utf-8 -*-
# @Author  : hanyx9010@163.com
# @Time    : 2025/9/7 12:42
# @File    : main_ablation.py
# @Description    :
import os
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import train_base_classifier as base_mod
import evaluate_base_classifier as eval_mod
import joint_select_classifier as sel_mod
from utils import compute_metrics, baseline_models, split_three_way, efficiency_weighted_predict


def run_once(args, random_state: int):
    """
    用不同的random_state做一次“划分→训练→评估”，
    返回：List[Dict]，每个Dict形如 {"method": 方法名, "ACC": ..., "F1": ..., "AUPRC": ..., "BAC": ...}
    """
    # 1) 读数据
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"目标列 {args.target} 不在数据中！可用列: {list(df.columns)[:10]} ...")
    y = df[args.target].values
    X = df.drop(columns=[args.target]).values

    # 2) 数据集划分
    X_tr_base, y_tr_base, X_eval, y_eval, X_tr_all, y_tr_all, X_te, y_te = split_three_way(
        X, y, test_size=args.test_size, eval_frac_of_train=args.eval_frac_of_train, seed=random_state
    )

    base_list = base_mod.train_base_learners(
        X_tr_base, y_tr_base, random_state=random_state
    )

    # ===== 记录基学习器的单个性能 --> CAR-DEA 得到效率和权重 =====
    eval_rows = []
    for r in base_list:
        est = r["estimator"]
        # 统一得到预测
        try:
            proba = est.predict_proba(X_eval)
            yp = np.argmax(proba, axis=1) if proba.ndim == 2 and proba.shape[1] > 1 else (proba >= 0.5).astype(int)
        except Exception:
            yp = est.predict(X_eval)
            proba = None
        m = compute_metrics(y_eval, yp, proba)
        row = {"learner_id": r["learner_id"], "group": r.get("group", "G0")}
        row.update(m)
        eval_rows.append(row)
    metrics_df = pd.DataFrame(eval_rows)
    # —— CAR-FDH 用于评价 ——
    car_fdh_eff, car_fdh_diversity = eval_mod.efficiency_evaluate(metrics_df, group_col="group", params=None)

    # ===== 比较四种评价模型：CAR-FDH / CAR-DEA / FDH / DEA =====
    variants = [
        {"label": "CCE[CAR-FDH]", "model": "car_fdh"},
        {"label": "CCE[CAR-DEA]", "model": "car_dea"},
        {"label": "CCE[FDH]",     "model": "fdh"},
        {"label": "CCE[DEA]",     "model": "dea"},
    ]

    # 支持一次性比较多个 Kmax（例如 [5,10,15,20]）
    kmax_list = getattr(args, "kmax_list", None)
    if kmax_list is None:
        kmax_list = [args.kmax]

    results: list[dict] = []

    def evaluate_variant(label: str, model_key: str, k_max: int):
        """
        给定一种评价模型（model_key）与 Kmax，执行“评价→联合选择→集成预测→评估→记录”
        """
        # —— 评价（得到效率表与多样性矩阵）——
        dea_df, diversity = eval_mod.efficiency_evaluate(
            metrics_df,
            group_col="group",
            params={"model": model_key}  # 在 eval_mod 内部据此选择 CAR-FDH/CAR-DEA/FDH/DEA
        )

        # —— 联合选择 ——（注意传入与该评价对应的 dea_df / diversity）
        sel_res = sel_mod.optimize_selection(
            dea_df=dea_df,
            diversity_matrix=diversity,
            A_cov=None,
            k_max=k_max,
            rho=0.0,
            group_min=None,
            solver="mosek"
        )
        selected_ids = set(sel_res.get("selected_ids", []))

        # —— Test 集成预测 ——（用效率作为权重；若你有别的加权方案，在此替换）
        eff_map = dict(zip(dea_df["learner_id"], dea_df["efficiency"]))
        y_pred_ens, p_ens = efficiency_weighted_predict(selected_ids, eff_map, base_list, X_te)

        # —— 评估 ——（和你全局一致的 compute_metrics）
        mt_ens = compute_metrics(
            y_te,
            (np.argmax(p_ens, axis=1) if hasattr(p_ens, "ndim") and p_ens is not None and p_ens.ndim == 2
             else (p_ens >= 0.5).astype(int) if p_ens is not None else y_pred_ens),
            p_ens
        )

        # —— 记录一行 —— 可按需附加更多信息（如所选基分类器数量、IDs 等）
        out_row = {"method": label, "Kmax": k_max, **mt_ens}
        # 若你的 compute_metrics 没有返回 EFF，这里可顺手添加“集成加权均值效率”之类的概要量
        # out_row["EFF_mean_sel"] = float(np.mean([eff_map[i] for i in selected_ids])) if selected_ids else np.nan
        results.append(out_row)

    # 逐变体 × 逐 Kmax 执行
    for var in variants:
        for k in kmax_list:
            evaluate_variant(var["label"], var["model"], k)

    return results

def pos_int(x):
    x = int(x)
    if x <= 0:
        raise argparse.ArgumentTypeError("kmax must be > 0.")
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="UCI_Credit_Card.csv", help="CSV 数据路径，如 data.csv")
    parser.add_argument("--target", type=str, default="default.payment.next.month", help="标签列名，如 label/target/y")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--eval_frac_of_train", type=float, default=0.25)
    parser.add_argument("--kmax", type=int, default=10, help="联合选择的最大选择数")
    parser.add_argument("--kmax_list", dest="kmax_list", type=pos_int, nargs="+",help="Multiple Kmax values, e.g., --kmax-list 5 8 10 12")
    parser.add_argument("--diversity", type=float, default=0.0)
    parser.add_argument("--n_runs", type=int, default=1, help="重复实验次数")
    parser.add_argument("--n_jobs", type=int, default=1, help="并行进程数（>1时并行）")
    parser.add_argument("--seed_initial", type=int, default=0, help="起始随机种子，后续自动递增")
    parser.add_argument("--out_path", type=str, default="./result", help="结果输出目录")
    parser.add_argument("--out", type=str, default="results_ablation_repeat.csv", help="逐次运行的原始结果表")
    parser.add_argument("--out_summary", type=str, default="results_ablation_summary.csv", help="按方法聚合(均值±标准差)")
    args = parser.parse_args()

    out_dir = Path(args.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed_initial + i for i in range(args.n_runs)]

    # 可选：并行加速
    all_rows = []
    if args.n_jobs == 1:
        for s in seeds:
            rows = run_once(args, random_state=s)
            for r in rows:
                r["run_seed"] = s
            metrics_df = pd.DataFrame(rows)
            metrics_df = metrics_df.rename(columns={"method": "learner_id"})
            dea_df, _ = eval_mod.efficiency_evaluate(metrics_df, group_col="run_seed", params=None)
            id_col = "learner_id"
            eff_col = "efficiency" if "efficiency" in dea_df.columns else next(
                c for c in dea_df.columns if c.lower().startswith("eff"))
            # 把效率并回到rows上
            rows_df = metrics_df.merge(dea_df[[id_col, eff_col]], on=id_col, how="left").rename(
                columns={eff_col: "eff"})
            # 如果你还需要把rows回成list[dict]
            rows = rows_df.to_dict(orient="records")
            all_rows.extend(rows)
    else:
        from joblib import Parallel, delayed
        bag = Parallel(n_jobs=args.n_jobs, prefer="processes")(delayed(run_once)(args, random_state=s) for s in seeds)
        for s, rows in zip(seeds, bag):
            for r in rows:
                r["run_seed"] = s
            metrics_df = pd.DataFrame(rows)
            metrics_df = metrics_df.rename(columns={"method": "learner_id"})
            dea_df, _ = eval_mod.efficiency_evaluate(metrics_df, group_col="run_seed", params=None)

            id_col = "learner_id"
            eff_col = "efficiency" if "efficiency" in dea_df.columns else next(
                c for c in dea_df.columns if c.lower().startswith("eff"))

            rows_df = metrics_df.merge(dea_df[[id_col, eff_col]], on=id_col, how="left").rename(
                columns={eff_col: "eff"})
            rows = rows_df.to_dict(orient="records")
            all_rows.extend(rows)

    # 整理为DataFrame并落盘
    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"learner_id": "method"})

    # 分组键：按 method 和 Kmax 汇总
    group_keys = ["method", "Kmax"]

    # 仅选择需要聚合的数值指标（排除分组键与 run_seed）
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    metric_cols = [c for c in num_cols if c not in (set(group_keys) | {"run_seed"})]

    # 统计量（均值、标准差、方差、样本量）
    agg_dict = {}
    for c in metric_cols:
        agg_dict[f"{c}_mean"] = (c, "mean")
        agg_dict[f"{c}_std"] = (c, "std")  # ddof=1，pandas 默认就是样本标准差
        agg_dict[f"{c}_var"] = (c, "var")
    # 额外给一个计数，方便查看每组的样本量
    agg_dict["n"] = (metric_cols[0], "count") if metric_cols else ("Kmax", "count")

    summary = df.groupby(group_keys, as_index=False).agg(**agg_dict)

    # 落盘
    df.to_csv(out_dir / args.out, index=False)
    summary.to_csv(out_dir / args.out_summary, index=False)

    # 友好打印
    print(f"\n原始结果已保存：{args.out}；汇总结果已保存：{args.out_summary}")
