import streamlit as st
import pandas as pd
from lib.data_loader import load_excel_guess_columns, summarize_text_stats
import numpy as np

st.set_page_config(page_title="台本データ 分析ハブ", page_icon="📊", layout="wide")

st.title("📊 台本データ 分析ハブ")
st.write("Excelファイルをアップロードし、プレビューを確認してから各ページで分析します。")

uploaded = st.file_uploader("Excelファイル（.xlsx）をアップロード", type=["xlsx"])

if uploaded:
    with st.spinner("読み込み中..."):
        df, meta = load_excel_guess_columns(uploaded)
        st.session_state["df"] = df
        st.session_state["meta"] = meta
        st.success("✅ ファイルが正常にアップロードされました！")
        
        # ファイル基本情報
        st.subheader("📋 ファイル情報")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ファイル名", uploaded.name)
        with col2:
            st.metric("行数", f"{meta['rows']:,}")
        with col3:
            st.metric("列数", f"{meta['cols']:,}")
        with col4:
            file_size = len(uploaded.getvalue()) / 1024 / 1024
            st.metric("ファイルサイズ", f"{file_size:.1f} MB")
        
        # 検出された列の情報
        st.subheader("🔍 自動検出された列")
        detection_info = []
        if meta.get("text_col"):
            detection_info.append({"項目": "台本テキスト列", "検出列名": meta["text_col"], "ステータス": "✅ 検出済み"})
        else:
            detection_info.append({"項目": "台本テキスト列", "検出列名": "-", "ステータス": "⚠️ 未検出"})
            
        if meta.get("profit_col"):
            detection_info.append({"項目": "収益列", "検出列名": meta["profit_col"], "ステータス": "✅ 検出済み"})
        else:
            detection_info.append({"項目": "収益列", "検出列名": "-", "ステータス": "⚠️ 未検出"})
            
        if meta.get("cv_col"):
            detection_info.append({"項目": "CV率列", "検出列名": meta["cv_col"], "ステータス": "✅ 検出済み"})
        else:
            detection_info.append({"項目": "CV率列", "検出列名": "-", "ステータス": "⚠️ 未検出"})
        
        st.dataframe(pd.DataFrame(detection_info), hide_index=True)
        
        # データプレビュー
        st.subheader("👀 データプレビュー")
        
        # 表示行数の選択
        preview_rows = st.select_slider(
            "表示行数を選択",
            options=[5, 10, 20, 50, 100],
            value=10,
            key="preview_rows"
        )
        
        # プレビューテーブル
        preview_df = df.head(preview_rows)
        st.dataframe(preview_df, use_container_width=True)
        
        # 各列の統計情報
        with st.expander("📊 各列の統計情報", expanded=False):
            stats_list = []
            for col in df.columns:
                col_data = df[col]
                null_count = col_data.isnull().sum()
                null_pct = (null_count / len(df)) * 100
                
                if pd.api.types.is_numeric_dtype(col_data):
                    col_type = "数値"
                    unique_count = col_data.nunique()
                    try:
                        mean_val = col_data.mean()
                        summary = f"平均: {mean_val:.2f}" if not pd.isna(mean_val) else "-"
                    except:
                        summary = "-"
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    col_type = "日時"
                    unique_count = col_data.nunique()
                    summary = "-"
                else:
                    col_type = "文字列"
                    unique_count = col_data.nunique()
                    try:
                        avg_length = col_data.astype(str).str.len().mean()
                        summary = f"平均文字数: {avg_length:.1f}" if not pd.isna(avg_length) else "-"
                    except:
                        summary = "-"
                
                stats_list.append({
                    "列名": col,
                    "データ型": col_type,
                    "欠損数": f"{null_count} ({null_pct:.1f}%)",
                    "ユニーク数": unique_count,
                    "統計": summary
                })
            
            st.dataframe(pd.DataFrame(stats_list), hide_index=True)

        # テキスト統計（従来の機能）
        if meta.get("text_col"):
            stats = summarize_text_stats(df, meta["text_col"])
            st.subheader("📝 テキスト詳細統計")
            st.dataframe(stats, hide_index=True)

