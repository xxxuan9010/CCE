# -*- coding: utf-8 -*-
# @Author  : hanyx9010@163.com
# @Time    : 2025/8/19 16:51
# @File    : main.py
# @Description    :

# 用法示例：
#   python main.py --datasets data1.csv data2.csv \
#                  --n_splits 5 --random_state 42 \
#                  --kmax 10 --coverage 0.9 \
#                  --out_dir ./runs

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
    # 你原本的数据加载
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
    # —— CAR-DEA 用于评价 ——
    dea_df, diversity = eval_mod.efficiency_evaluate(metrics_df, group_col="group", params=None)

    # ===== 联合选择（你的 select_classifier.py）=====
    sel_res = sel_mod.optimize_selection(
        dea_df=dea_df, diversity_matrix=diversity, A_cov=None,
        k_max=args.kmax, rho=0.0, group_min=None, solver="mosek"
    )
    selected_ids = set(sel_res.get("selected_ids", []))

    # ===== Test 上的集成预测 =====
    eff_map = dict(zip(dea_df["learner_id"], dea_df["efficiency"]))
    y_pred_ens, p_ens = efficiency_weighted_predict(selected_ids, eff_map, base_list, X_te)
    # 评估
    mt_ens = compute_metrics(
        y_te,
        (np.argmax(p_ens, axis=1) if p_ens.ndim == 2 else (p_ens >= 0.5).astype(int)),
        p_ens
    )

    # 把 CCE 集成结果包装成一行 DataFrame
    results: list[dict] = []
    results.append({"method": f"CCE", **mt_ens})

    # ===== Baseline：在 Train_all 上训练 → Test 上评估（公平对比）=====
    for name, est in baseline_models().items():
        est.fit(X_tr_all, y_tr_all)
        try:
            proba = est.predict_proba(X_te)
            yp = np.argmax(proba, axis=1) if proba.ndim == 2 and proba.shape[1] > 1 else (proba >= 0.5).astype(int)
        except Exception:
            yp = est.predict(X_te)
            proba = None
        m = compute_metrics(y_te, yp, proba)
        results.append({"method": name, **m})

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="UCI_Credit_Card.csv", help="CSV 数据路径，如 data.csv")
    parser.add_argument("--target", type=str, default="default.payment.next.month", help="标签列名，如 label/target/y")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--eval_frac_of_train", type=float, default=0.1)
    parser.add_argument("--kmax", type=int, default=10, help="联合选择的最大选择数")
    parser.add_argument("--n_runs", type=int, default=2, help="重复实验次数")
    parser.add_argument("--n_jobs", type=int, default=2, help="并行进程数（>1时并行）")
    parser.add_argument("--seed_initial", type=int, default=0, help="起始随机种子，后续自动递增")
    parser.add_argument("--out_path", type=str, default="./result", help="结果输出目录")
    parser.add_argument("--out", type=str, default="results_repeat.csv", help="逐次运行的原始结果表")
    parser.add_argument("--out_summary", type=str, default="results_summary.csv", help="按方法聚合(均值±标准差)")
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
    # 建议列名里只保留“method”和若干数值指标，避免字符串列参与聚合报错
    metric_cols = [c for c in df.columns if c not in ["method", "run_seed"]]
    # 计算均值与标准差
    mean_df = df.groupby("method", as_index=False)[metric_cols].mean().rename(
        columns={c: f"{c}_mean" for c in metric_cols}
    )
    std_df = df.groupby("method", as_index=False)[metric_cols].std(ddof=1).rename(
        columns={c: f"{c}_std" for c in metric_cols}
    )
    summary = pd.merge(mean_df, std_df, on="method", how="inner")

    df.to_csv(out_dir / args.out, index=False)
    summary.to_csv(out_dir / args.out_summary, index=False)

    # 友好打印
    print(f"\n原始结果已保存：{args.out}；汇总结果已保存：{args.out_summary}")




