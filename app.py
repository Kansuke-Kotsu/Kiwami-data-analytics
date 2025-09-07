import streamlit as st
import pandas as pd
from lib.data_loader import load_excel_guess_columns, summarize_text_stats

st.set_page_config(page_title="台本データ 分析ハブ", page_icon="📊", layout="wide")

st.title("📊 台本データ 分析ハブ")
st.write("Excel（先ほどのファイル）をアップロードし、各ページで分析します。")

uploaded = st.file_uploader("Excelファイル（.xlsx）をアップロード", type=["xlsx"])

if uploaded:
    with st.spinner("読み込み中..."):
        df, meta = load_excel_guess_columns(uploaded)
        st.session_state["df"] = df
        st.session_state["meta"] = meta
        st.success("読み込み完了！")
        st.write("検出情報:", meta)
        st.write("先頭5件", df.head())

        # 概況（テキスト統計）
        if meta.get("text_col"):
            stats = summarize_text_stats(df, meta["text_col"])
            st.subheader("テキスト概況")
            st.dataframe(stats)

st.markdown("---")
st.info("左サイドバーからページを選択してください：\n\n- ① キーワード抽出（収益に効くn-gramを可視化）\n\n今後、ページを追加していけます。")
