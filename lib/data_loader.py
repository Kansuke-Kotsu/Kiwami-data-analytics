import pandas as pd
import numpy as np
import re
from io import BytesIO

def _normalize_header(row0):
    h = row0.astype(str).str.replace(r"\s+", " ", regex=True).str.replace("\n", " ").str.strip()
    return h

def _pick_col(candidates, cols):
    for name in candidates:
        matches = [c for c in cols if str(c).strip() == name]
        if matches:
            return matches[0]
    for name in candidates:
        matches = [c for c in cols if name.lower() in str(c).lower()]
        if matches:
            return matches[0]
    return None

TEXT_CANDIDATES = ["台本データ", "台本", "本文", "script", "text"]
PROFIT_CANDIDATES = ["合算粗利", "粗利", "収益", "広告収益", "売上", "利益"]
CV_CANDIDATES = ["CV率", "CV", "conversion", "コンバージョン率"]

def load_excel_guess_columns(file_like):
    # 読み込み（最も列数が多いシートを対象に）
    xls = pd.ExcelFile(file_like)
    best = None
    best_df = None
    for sn in xls.sheet_names:
        tmp = pd.read_excel(xls, sheet_name=sn, header=None)
        if best is None or tmp.shape[1] > best:
            best = tmp.shape[1]
            best_df = tmp.copy()

    header = _normalize_header(best_df.iloc[0])
    df = best_df.iloc[1:].copy()
    df.columns = header
    df = df.reset_index(drop=True)

    cols = list(df.columns)
    col_text = _pick_col(TEXT_CANDIDATES, cols)
    col_profit = _pick_col(PROFIT_CANDIDATES, cols)
    col_cv = _pick_col(CV_CANDIDATES, cols)

    # 型調整
    def to_float(s):
        if pd.isna(s):
            return np.nan
        if isinstance(s, (int, float)):
            return float(s)
        s = str(s).replace(",", "").replace("%", "").strip()
        try:
            return float(s)
        except:
            return np.nan

    if col_profit:
        df[col_profit] = df[col_profit].apply(to_float)
    if col_cv:
        df[col_cv] = df[col_cv].apply(to_float)

    meta = {
        "sheet_used": "auto-selected",
        "text_col": col_text,
        "profit_col": col_profit,
        "cv_col": col_cv,
        "rows": len(df),
        "cols": len(df.columns),
    }
    return df, meta

def summarize_text_stats(df, text_col):
    s = df[text_col].fillna("").astype(str)
    out = pd.DataFrame({
        "件数": [len(s)],
        "文字数_中央値": [int(s.str.len().median())],
        "文字数_平均": [round(s.str.len().mean(), 1)],
        "行数_中央値": [int(s.str.count("\n").median())],
        "感嘆符_中央値": [int(s.str.count("!").add(s.str.count("！")).median())],
        "疑問符_中央値": [int(s.str.count("\?").add(s.str.count("？")).median())],
        "数字文字_中央値": [int(s.apply(lambda x: sum(ch.isdigit() for ch in x)).median())],
    })
    return out
