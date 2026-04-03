from pathlib import Path

from fastapi import APIRouter, Query, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

from database import get_db
from models import SiteData, AfterData

router = APIRouter(tags=["Visualize"])


# ===============================
# JSON 安全處理
# ===============================
def safe_json(obj):
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        v = obj.item()
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    elif isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj
    return obj


# ===============================
# 共用：依 site_id + file_name 抓原始資料
# ===============================
def get_site_entries(db: Session, site_id: int, file_name: str):
    entries = (
        db.query(SiteData)
        .filter(
            SiteData.site_id == site_id,
            SiteData.data_name == file_name,
        )
        .order_by(SiteData.the_date, SiteData.the_hour, SiteData.data_id)
        .all()
    )
    return entries


# ===============================
# 共用：檢查資料一致性
# ===============================
def validate_entries(entries, site_id: int, file_name: str):
    if not entries:
        raise HTTPException(
            status_code=404,
            detail=f"找不到資料：site_id={site_id}, file_name={file_name}",
        )

    entry_site_ids = {e.site_id for e in entries}
    if len(entry_site_ids) != 1 or site_id not in entry_site_ids:
        raise HTTPException(
            status_code=400,
            detail="查到的 site_data 與指定 site_id 不一致，請檢查資料表內容",
        )


# ===============================
# 共用：把 SiteData 轉成 DataFrame
# ===============================
def build_raw_dataframe(entries):
    df = pd.DataFrame(
        [
            {
                "data_id": e.data_id,
                "site_id": e.site_id,
                "data_name": e.data_name,
                "EAC": e.eac,
                "GI": e.gi,
                "TM": e.tm,
                "the_date": e.the_date,
                "hour": e.the_hour,
            }
            for e in entries
        ]
    )

    if df.empty:
        return df

    df = df.drop_duplicates(subset=["the_date", "hour"], keep="first")
    df["the_date"] = pd.to_datetime(df["the_date"])
    df["month"] = df["the_date"].dt.month
    df["day"] = df["the_date"].dt.day
    return df


# ===============================
# 共用：Stage 1 + Stage 2 清理
# ===============================
def clean_dataframe(
    df: pd.DataFrame,
    *,
    apply_gi_tm: bool = True,
    outlier_method: str = "none",
    iqr_factor: float = 1.5,
    z_threshold: float = 3.0,
    isolation_contamination: float = 0.1,
    remove_outliers: bool = False,
):
    if df.empty:
        return df.copy(), pd.Series(dtype=bool)

    # Stage 1：GI / TM 清理
    df1 = df.copy()

    if apply_gi_tm:
        df1 = df1[df1["GI"] > 0].copy()
        df1.loc[df1["TM"] <= 0, "TM"] = np.nan
        df1 = df1.sort_values(["the_date", "hour"])

        if df1["TM"].notna().sum() >= 2:
            df1["TM"] = df1["TM"].interpolate("linear", limit_direction="both")

    # Stage 2：離群值
    df2 = df1.copy()
    outlier_mask = pd.Series(False, index=df2.index)

    cols = ["EAC", "GI", "TM"]
    if outlier_method == "iqr_single":
        cols = ["EAC"]

    if outlier_method != "none":
        if outlier_method.startswith("iqr"):
            for col in cols:
                s = df2[col].dropna()
                if len(s) < 10:
                    continue
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - iqr_factor * iqr
                upper = q3 + iqr_factor * iqr
                outlier_mask |= (df2[col] < lower) | (df2[col] > upper)

        elif outlier_method == "zscore":
            for col in cols:
                s = df2[col].dropna()
                if len(s) == 0 or s.std() == 0:
                    continue
                z = np.abs((df2[col] - s.mean()) / s.std())
                outlier_mask |= z > z_threshold

        elif outlier_method == "isolation_forest":
            sub = df2[cols].dropna()
            if len(sub) > 20:
                iso = IsolationForest(
                    contamination=isolation_contamination,
                    random_state=42,
                )
                pred = iso.fit_predict(sub)
                mask = pd.Series(pred == -1, index=sub.index)
                outlier_mask.loc[mask.index] = mask

        elif outlier_method != "none":
            raise HTTPException(status_code=400, detail=f"不支援的 outlier_method: {outlier_method}")

    if outlier_method != "none" and remove_outliers:
        df2.loc[outlier_mask, cols] = np.nan
        df2 = df2.sort_values(["the_date", "hour"])
        df2[cols] = df2[cols].interpolate("linear", limit_direction="both")

    return df2, outlier_mask


# ===============================
# 圖表資料產生器
# ===============================
def build_plots(
    df: pd.DataFrame,
    outlier_mask: pd.Series | None = None,
    *,
    remove_outliers: bool = False,
    correlation_heatmap: dict | None = None,
    correlation_heatmap_full: dict | None = None,
):
    if outlier_mask is None:
        outlier_mask = pd.Series(False, index=df.index)
    outlier_mask = outlier_mask.reindex(df.index, fill_value=False)

    variables = ["EAC", "GI", "TM"]
    hist = {}

    # Histogram
    for v in variables:
        s = df[v].dropna()
        if len(s) < 5:
            hist[v] = {"bins": [], "counts": []}
            continue
        counts, bins = np.histogram(s, bins=10)
        hist[v] = {
            "bins": bins.tolist(),
            "counts": counts.tolist(),
        }

    # Scatter
    pairs = {}
    for x in variables:
        for y in variables:
            if x == y:
                continue
            sub = df[[x, y]].dropna()
            pairs[f"{x}__{y}"] = {
                "x": sub[x].tolist(),
                "y": sub[y].tolist(),
                "is_outlier": outlier_mask.loc[sub.index].tolist(),
            }

    # Boxplot
    def build_box(group_col: str, show_outliers: bool = True):
        result = {}
        for g, sub in df.groupby(group_col):
            v = sub["EAC"].dropna()
            if len(v) == 0:
                continue

            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                lower = upper = v.median()
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

            inside = v[(v >= lower) & (v <= upper)]
            whisker_min = inside.min() if not inside.empty else v.min()
            whisker_max = inside.max() if not inside.empty else v.max()

            outliers = v[(v < lower) | (v > upper)].tolist()

            result[str(g)] = {
                "min": float(v.min()),
                "q1": float(q1),
                "median": float(v.median()),
                "q3": float(q3),
                "max": float(v.max()),
                "whisker_min": float(whisker_min),
                "whisker_max": float(whisker_max),
                "outliers": outliers if show_outliers else [],
            }
        return result

    show_outliers = not remove_outliers

    return {
        "scatter_matrix": {
            "variables": variables,
            "pairs": pairs,
            "hist": hist,
        },
        "boxplot_by_month": build_box("month", show_outliers=show_outliers),
        "boxplot_by_day": build_box("day", show_outliers=show_outliers),
        "boxplot_by_hour": build_box("hour", show_outliers=show_outliers),
        "boxplot_by_batch": {},
        "correlation_heatmap": correlation_heatmap,
        "correlation_heatmap_full": correlation_heatmap_full,
    }


# ===============================
# 主 API
# ===============================
@router.get("/visualize-data/")
def visualize_data(
    site_id: int = Query(...),
    file_name: str = Query(...),
    apply_gi_tm: bool = Query(True),
    outlier_method: str = Query("none"),
    iqr_factor: float = Query(1.5),
    z_threshold: float = Query(3.0),
    isolation_contamination: float = Query(0.1),
    remove_outliers: bool = Query(False),
    db: Session = Depends(get_db),
):
    # ---------- 撈資料 ----------
    entries = get_site_entries(db, site_id, file_name)
    validate_entries(entries, site_id, file_name)

    # ---------- 原始 df ----------
    df = build_raw_dataframe(entries)
    if df.empty:
        raise HTTPException(status_code=404, detail="原始資料為空")

    # ---------- correlation ----------
    corr_data = df.copy()
    if apply_gi_tm:
        corr_data = corr_data[corr_data["GI"] > 0].copy()
        corr_data.loc[corr_data["TM"] <= 0, "TM"] = np.nan
        corr_data = corr_data.sort_values(["the_date", "hour"])
        if corr_data["TM"].notna().sum() >= 2:
            corr_data["TM"] = corr_data["TM"].interpolate("linear", limit_direction="both")

    corr_vars = ["EAC", "GI", "TM", "day", "hour", "month"]
    corr_base = corr_data[corr_vars].dropna()

    if len(corr_base) >= 2:
        corr_heatmap = {
            "variables": ["EAC", "GI", "TM"],
            "matrix": corr_base[["EAC", "GI", "TM"]].corr().values.tolist(),
        }
        corr_heatmap_full = {
            "variables": corr_vars,
            "matrix": corr_base.corr().values.tolist(),
        }
    else:
        corr_heatmap = {"variables": ["EAC", "GI", "TM"], "matrix": []}
        corr_heatmap_full = {"variables": corr_vars, "matrix": []}

    # ---------- Stage 1 ----------
    df1 = df.copy()
    if apply_gi_tm:
        df1 = df1[df1["GI"] > 0].copy()
        df1.loc[df1["TM"] <= 0, "TM"] = np.nan
        df1 = df1.sort_values(["the_date", "hour"])
        if df1["TM"].notna().sum() >= 2:
            df1["TM"] = df1["TM"].interpolate("linear", limit_direction="both")

    # ---------- Stage 0 ----------
    plots_raw = build_plots(
        df,
        outlier_mask=pd.Series(False, index=df.index),
        remove_outliers=remove_outliers,
        correlation_heatmap=corr_heatmap,
        correlation_heatmap_full=corr_heatmap_full,
    )

    # ---------- Stage 2 ----------
    df2 = df1.copy()
    outlier_mask_stage2 = pd.Series(False, index=df2.index)

    cols = ["EAC", "GI", "TM"]
    if outlier_method == "iqr_single":
        cols = ["EAC"]

    if outlier_method != "none":
        if outlier_method.startswith("iqr"):
            for col in cols:
                s = df2[col].dropna()
                if len(s) < 10:
                    continue
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - iqr_factor * iqr
                upper = q3 + iqr_factor * iqr
                outlier_mask_stage2 |= (df2[col] < lower) | (df2[col] > upper)

        elif outlier_method == "zscore":
            for col in cols:
                s = df2[col].dropna()
                if len(s) == 0 or s.std() == 0:
                    continue
                z = np.abs((df2[col] - s.mean()) / s.std())
                outlier_mask_stage2 |= z > z_threshold

        elif outlier_method == "isolation_forest":
            sub = df2[cols].dropna()
            if len(sub) > 20:
                iso = IsolationForest(
                    contamination=isolation_contamination,
                    random_state=42,
                )
                pred = iso.fit_predict(sub)
                mask = pd.Series(pred == -1, index=sub.index)
                outlier_mask_stage2.loc[mask.index] = mask

        elif outlier_method != "none":
            raise HTTPException(status_code=400, detail=f"不支援的 outlier_method: {outlier_method}")

    outlier_mask_stage1 = outlier_mask_stage2.reindex(df1.index, fill_value=False)
    outlier_mask_raw = outlier_mask_stage2.reindex(df.index, fill_value=False)

    plots_raw = build_plots(
        df,
        outlier_mask=outlier_mask_raw,
        remove_outliers=remove_outliers,
        correlation_heatmap=corr_heatmap,
        correlation_heatmap_full=corr_heatmap_full,
    )

    plots_stage1 = build_plots(
        df1,
        outlier_mask=outlier_mask_stage1,
        remove_outliers=remove_outliers,
        correlation_heatmap=corr_heatmap,
        correlation_heatmap_full=corr_heatmap_full,
    )

    if outlier_method != "none" and remove_outliers:
        df2.loc[outlier_mask_stage2, cols] = np.nan
        df2 = df2.sort_values(["the_date", "hour"])
        df2[cols] = df2[cols].interpolate("linear", limit_direction="both")
        plots_stage2 = build_plots(
            df2,
            outlier_mask=pd.Series(False, index=df2.index),
            remove_outliers=remove_outliers,
            correlation_heatmap=corr_heatmap,
            correlation_heatmap_full=corr_heatmap_full,
        )
    else:
        plots_stage2 = build_plots(
            df2,
            outlier_mask=outlier_mask_stage2,
            remove_outliers=remove_outliers,
            correlation_heatmap=corr_heatmap,
            correlation_heatmap_full=corr_heatmap_full,
        )

    return safe_json(
        {
            "site_id": site_id,
            "file_name": file_name,
            "stages": {
                "raw": plots_raw,
                "after_gi_tm": plots_stage1,
                "after_outlier": plots_stage2,
            },
        }
    )


# ===============================
# 儲存清理後資料 + 匯出 CSV
# ===============================
@router.post("/save-cleaned-data/")
def save_cleaned_data(payload: dict, db: Session = Depends(get_db)):
    site_id = payload.get("site_id")
    file_name = payload.get("file_name")
    apply_gi_tm = payload.get("apply_gi_tm", True)
    outlier_method = payload.get("outlier_method", "none")
    remove_outliers = payload.get("remove_outliers", True)
    iqr_factor = float(payload.get("iqr_factor", 1.5))
    z_threshold = float(payload.get("z_threshold", 3.0))
    isolation_contamination = float(payload.get("isolation_contamination", 0.1))

    if site_id is None:
        raise HTTPException(status_code=400, detail="缺少 site_id")

    try:
        site_id = int(site_id)
    except Exception:
        raise HTTPException(status_code=400, detail="site_id 必須是整數")

    if not file_name:
        raise HTTPException(status_code=400, detail="缺少 file_name")

    entries = get_site_entries(db, site_id, file_name)
    validate_entries(entries, site_id, file_name)

    df = build_raw_dataframe(entries)
    if df.empty:
        raise HTTPException(status_code=404, detail="找不到原始資料")

    before_rows = len(df)

    # Stage 1
    if apply_gi_tm:
        df = df[df["GI"] > 0].copy()
        df.loc[df["TM"] <= 0, "TM"] = np.nan
        df = df.sort_values(["the_date", "hour"])
        if df["TM"].notna().sum() >= 2:
            df["TM"] = df["TM"].interpolate("linear", limit_direction="both")

    # Stage 2
    cols = ["EAC", "GI", "TM"]
    if outlier_method == "iqr_single":
        cols = ["EAC"]

    if outlier_method != "none" and remove_outliers:
        outlier_mask = pd.Series(False, index=df.index)

        if outlier_method.startswith("iqr"):
            for col in cols:
                s = df[col].dropna()
                if len(s) < 10:
                    continue
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - iqr_factor * iqr
                upper = q3 + iqr_factor * iqr
                outlier_mask |= (df[col] < lower) | (df[col] > upper)

        elif outlier_method == "zscore":
            for col in cols:
                s = df[col].dropna()
                if len(s) == 0 or s.std() == 0:
                    continue
                z = np.abs((df[col] - s.mean()) / s.std())
                outlier_mask |= z > z_threshold

        elif outlier_method == "isolation_forest":
            sub = df[cols].dropna()
            if len(sub) > 20:
                iso = IsolationForest(
                    contamination=isolation_contamination,
                    random_state=42,
                )
                pred = iso.fit_predict(sub)
                mask = pd.Series(pred == -1, index=sub.index)
                outlier_mask.loc[mask.index] = mask

        elif outlier_method != "none":
            raise HTTPException(status_code=400, detail=f"不支援的 outlier_method: {outlier_method}")

        df.loc[outlier_mask, cols] = np.nan
        df = df.sort_values(["the_date", "hour"])
        df[cols] = df[cols].interpolate("linear", limit_direction="both")

    after_rows = len(df)

    # 匯出 CSV
    processed_dir = Path(__file__).resolve().parent.parent / "processed_data" / str(site_id)
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = f"{Path(file_name).stem}_site_{site_id}_cleaned.csv"
    csv_path = processed_dir / csv_filename
    df.to_csv(csv_path, index=False)

    # outlier params
    outlier_params = None
    if outlier_method.startswith("iqr"):
        outlier_params = {"iqr_factor": iqr_factor}
    elif outlier_method == "zscore":
        outlier_params = {"z_threshold": z_threshold}
    elif outlier_method == "isolation_forest":
        outlier_params = {"contamination": isolation_contamination}

    # 注意：after_data.data_id 目前仍沿用舊設計，只能先暫存第一筆來源 data_id
    source_data_id = int(entries[0].data_id)

    after = AfterData(
        site_id=site_id,
        data_id=source_data_id,
        after_name=f"{file_name}_site_{site_id}_cleaned",
        before_rows=before_rows,
        after_rows=after_rows,
        removed_ratio=(before_rows - after_rows) / before_rows if before_rows > 0 else 0,
        outlier_method=outlier_method if outlier_method != "none" else None,
        gi_tm_applied=apply_gi_tm,
        outlier_params=outlier_params,
        file_path=str(csv_path),
    )

    db.add(after)
    db.commit()
    db.refresh(after)

    return safe_json(
        {
            "message": "清理完成並已儲存 CSV",
            "site_id": site_id,
            "file_name": file_name,
            "before_rows": before_rows,
            "after_rows": after_rows,
            "removed_ratio": round(
                (before_rows - after_rows) / before_rows if before_rows > 0 else 0,
                3,
            ),
            "after_id": after.after_id,
            "file_path": str(csv_path),
        }
    )