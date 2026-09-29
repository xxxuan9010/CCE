# -*- coding: utf-8 -*-
# @Author  : hanyx9010@163.com
# @Time    : 2025/9/8 20:41
# @File    : main_robust.py
# @Description    :
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import train_base_classifier as base_mod
import evaluate_base_classifier as eval_mod
import joint_select_classifier as sel_mod
from utils import compute_metrics, split_three_way, efficiency_weighted_predict


def run_once(args, random_state: int):
    """
    单次运行（给定 random_state）：
      1) 数据划分 → 训练基学习器
      2) 在验证集上评估单个基学习器指标
      3) 用 CAR-FDH 评价得到效率与多样性
      4) 针对 (rho, Kmax) 网格做联合选择与测试集评估
    返回：List[Dict]，每个 Dict 含 method, rho, Kmax 及一组指标
    """
    # 1) 读数据
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"目标列 {args.target} 不在数据中！可用列: {list(df.columns)[:10]} ...")
    y = df[args.target].values
    X = df.drop(columns=[args.target]).values

    # 2) 三分数据（base-train / eval / test）
    X_tr_base, y_tr_base, X_eval, y_eval, X_tr_all, y_tr_all, X_te, y_te = split_three_way(
        X, y, test_size=args.test_size, eval_frac_of_train=args.eval_frac_of_train, seed=random_state
    )

    # 3) 训练基学习器池
    base_list = base_mod.train_base_learners(
        X_tr_base, y_tr_base, random_state=random_state
    )

    # 4) 在 eval 上计算各基学习器的指标
    eval_rows = []
    for r in base_list:
        est = r["estimator"]
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

    # 5) 仅用 CAR-FDH 做评价，得到效率表与多样性矩阵（与 rho 无关）
    dea_df, diversity = eval_mod.efficiency_evaluate(
        metrics_df, group_col="group", params={"model": "car_fdh"}
    )

    # 6) 遍历 rho × Kmax，做联合选择与测试集评估
    kmax_list = args.kmax_list if args.kmax_list else [args.kmax]
    rho_list = args.rho_list if args.rho_list else [args.rho]

    results: list[dict] = []
    eff_map = dict(zip(dea_df["learner_id"], dea_df["efficiency"]))

    for rho in rho_list:
        for k_max in kmax_list:
            sel_res = sel_mod.optimize_selection(
                dea_df=dea_df,
                diversity_matrix=diversity,
                A_cov=None,
                k_max=k_max,
                rho=float(rho),     # 多样性权重
                group_min=None,
                solver="mosek"
            )
            selected_ids = set(sel_res.get("selected_ids", []))

            # 集成预测：效率加权
            y_pred_ens, p_ens = efficiency_weighted_predict(selected_ids, eff_map, base_list, X_te)

            # 评估
            y_hat = (np.argmax(p_ens, axis=1) if p_ens is not None and hasattr(p_ens, "ndim") and p_ens.ndim == 2
                     else (p_ens >= 0.5).astype(int) if p_ens is not None else y_pred_ens)
            mt = compute_metrics(y_te, y_hat, p_ens)

            results.append({
                "method": "CCE[CAR-FDH]",
                "rho": float(rho),
                "Kmax": int(k_max),
                **mt,
                # 可按需附加选择规模、平均效率等摘要量
                "n_selected": len(selected_ids),
                "eff_mean_sel": float(np.mean([eff_map[i] for i in selected_ids])) if selected_ids else np.nan
            })

    return results


def pos_int(x):
    x = int(x)
    if x <= 0:
        raise argparse.ArgumentTypeError("kmax must be > 0.")
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="UCI_Credit_Card.csv", help="CSV 数据路径")
    parser.add_argument("--target", type=str, default="default.payment.next.month", help="标签列名")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--eval_frac_of_train", type=float, default=0.25)

    # Kmax（Z_max）
    parser.add_argument("--kmax", type=pos_int, default=10, help="联合选择的最大选择数 (Z_max)")
    parser.add_argument("--kmax_list", dest="kmax_list", type=pos_int, nargs="+",
                        help="多个 Kmax 值，如：--kmax_list 5 10")

    # 多样性权重 rho
    parser.add_argument("--rho", type=float, default=0.0, help="多样性权重（单值）")
    parser.add_argument("--rho_list", dest="rho_list", type=float, nargs="+",
                        help="多个多样性权重，如：--rho_list 0 0.01")

    parser.add_argument("--n_runs", type=int, default=2, help="重复实验次数")
    parser.add_argument("--n_jobs", type=int, default=2, help="并行进程数（>1时并行）")
    parser.add_argument("--seed_initial", type=int, default=0, help="起始随机种子，后续自动递增")
    parser.add_argument("--out_path", type=str, default="./result", help="结果输出目录")
    parser.add_argument("--out", type=str, default="results_carfdh_grid.csv", help="逐次运行的原始结果表")
    parser.add_argument("--out_summary", type=str, default="results_carfdh_grid_summary.csv",
                        help="(rho,Kmax) 聚合(均值±标准差)")
    args = parser.parse_args()

    out_dir = Path(args.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed_initial + i for i in range(args.n_runs)]

    # 串行/并行
    all_rows = []
    if args.n_jobs == 1:
        for s in seeds:
            rows = run_once(args, random_state=s)
            for r in rows:
                r["run_seed"] = s
            all_rows.extend(rows)
    else:
        from joblib import Parallel, delayed

        bag = Parallel(n_jobs=args.n_jobs, prefer="processes")(
            delayed(run_once)(args, random_state=s) for s in seeds
        )
        for s, rows in zip(seeds, bag):
            for r in rows:
                r["run_seed"] = s
            all_rows.extend(rows)

    # DataFrame 化并落盘
    df = pd.DataFrame(all_rows)

    # 聚合：按 (rho, Kmax)
    group_keys = ["rho", "Kmax"]
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    metric_cols = [c for c in num_cols if c not in (set(group_keys) | {"run_seed"})]

    agg_dict = {}
    for c in metric_cols:
        agg_dict[f"{c}_mean"] = (c, "mean")
        agg_dict[f"{c}_std"] = (c, "std")
        agg_dict[f"{c}_var"] = (c, "var")
    agg_dict["n"] = (metric_cols[0], "count") if metric_cols else ("Kmax", "count")

    summary = df.groupby(group_keys, as_index=False).agg(**agg_dict)

    df.to_csv(out_dir / args.out, index=False)
    summary.to_csv(out_dir / args.out_summary, index=False)

    print(f"\n原始结果已保存：{out_dir / args.out}")
    print(f"汇总结果已保存：{out_dir / args.out_summary}")