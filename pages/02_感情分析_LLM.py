import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from scipy.stats import pearsonr, spearmanr
import io

st.set_page_config(page_title="② 感情分析 (LLM)", page_icon="🧠", layout="wide")
st.title("🧠 感情分析による収益相関分析 (LLM版)")

# セッション状態チェック
if "df" not in st.session_state:
    st.warning("まずトップページでExcelをアップロードしてください。")
    st.stop()

df = st.session_state["df"].copy()
meta = st.session_state.get("meta", {})

# Anthropic API キー確認
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("🔑 ANTHROPIC_API_KEY環境変数が設定されていません。")
    st.info("Streamlitの.streamlit/secrets.tomlファイルまたは環境変数でAPIキーを設定してください。")
    st.stop()

# 列選択
st.subheader("📋 データ列選択")
col1, col2 = st.columns(2)

with col1:
    script_col = st.selectbox(
        "台本データ列",
        options=list(df.columns),
        index=list(df.columns).index(meta.get("text_col")) if meta.get("text_col") in df.columns else 0
    )

with col2:
    revenue_col = st.selectbox(
        "収益データ列", 
        options=[c for c in df.columns if c != script_col],
        index=list(df.columns).index(meta.get("profit_col")) if meta.get("profit_col") in df.columns else 0
    )

# データの前処理
df_clean = df.copy()
df_clean[script_col] = df_clean[script_col].fillna("").astype(str)
df_clean[revenue_col] = pd.to_numeric(df_clean[revenue_col], errors="coerce")

# 有効なデータのみ抽出
valid_data = df_clean[
    (df_clean[script_col].str.strip() != "") & 
    (df_clean[revenue_col].notna())
].copy()

if len(valid_data) == 0:
    st.error("有効な台本データと収益データのペアが見つかりません。")
    st.stop()

st.info(f"📊 有効データ件数: {len(valid_data):,}件 / 全体: {len(df):,}件")

# 感情分析設定
st.subheader("⚙️ 感情分析設定")

col1, col2 = st.columns(2)
with col1:
    analysis_type = st.selectbox(
        "感情分析タイプ",
        ["基本感情（5分類）", "詳細感情（10分類）"]
    )

with col2:
    max_samples = st.slider(
        "最大分析件数（API使用量調整）",
        min_value=10,
        max_value=min(500, len(valid_data)),
        value=min(100, len(valid_data)),
        step=10
    )

# サンプリング
if len(valid_data) > max_samples:
    sample_data = valid_data.sample(n=max_samples, random_state=42)
    st.info(f"🎯 {max_samples}件をランダムサンプリングしました")
else:
    sample_data = valid_data
    st.info(f"📋 全{len(sample_data)}件を分析します")

# 感情分析実行
if st.button("🚀 感情分析実行", type="primary"):
    
    # Anthropic API client のセットアップ
    try:
        import anthropic
        client = anthropic.Client(api_key=api_key)
    except ImportError:
        st.error("anthropicライブラリがインストールされていません。requirements.txtを確認してください。")
        st.stop()
    except Exception as e:
        st.error(f"Anthropic APIクライアントの初期化に失敗しました: {e}")
        st.stop()
    
    # 感情分析プロンプト設定
    if analysis_type == "基本感情（5分類）":
        emotion_categories = ["ポジティブ", "ネガティブ", "中性", "興奮", "不安"]
        system_prompt = """あなたは感情分析の専門家です。提供された日本語テキストを分析し、以下の5つの感情カテゴリーそれぞれについて0-10のスコア（小数点1桁まで）で評価してください。

感情カテゴリー:
- ポジティブ: 喜び、希望、満足、楽観的な感情
- ネガティブ: 悲しみ、失望、不満、悲観的な感情  
- 中性: 客観的、事実的、感情的でない内容
- 興奮: エネルギッシュ、活発、刺激的な感情
- 不安: 心配、恐れ、緊張、不確実性への懸念

JSON形式で回答してください:
{"ポジティブ": X.X, "ネガティブ": X.X, "中性": X.X, "興奮": X.X, "不安": X.X}"""
        
    else:  # 詳細感情（10分類）
        emotion_categories = ["喜び", "悲しみ", "怒り", "恐怖", "驚き", "嫌悪", "期待", "信頼", "中性", "混合"]
        system_prompt = """あなたは感情分析の専門家です。提供された日本語テキストを分析し、以下の10つの感情カテゴリーそれぞれについて0-10のスコア（小数点1桁まで）で評価してください。

感情カテゴリー:
- 喜び: 幸福、喜悦、楽しさ
- 悲しみ: 憂鬱、失望、悲哀
- 怒り: 憤り、いらだち、怒り
- 恐怖: 恐れ、不安、懸念
- 驚き: 驚愕、意外感、びっくり
- 嫌悪: 不快、拒絶、嫌悪感
- 期待: 希望、期待感、楽しみ
- 信頼: 安心、信頼感、確信
- 中性: 客観的、事実的内容
- 混合: 複数の感情が混在

JSON形式で回答してください:
{"喜び": X.X, "悲しみ": X.X, "怒り": X.X, "恐怖": X.X, "驚き": X.X, "嫌悪": X.X, "期待": X.X, "信頼": X.X, "中性": X.X, "混合": X.X}"""
    
    # プログレスバー
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    sentiment_results = []
    
    for idx, (_, row) in enumerate(sample_data.iterrows()):
        try:
            status_text.text(f"分析中... {idx+1}/{len(sample_data)} ({(idx+1)/len(sample_data)*100:.1f}%)")
            
            text = str(row[script_col])[:2000]  # テキスト長制限
            
            # API呼び出し
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"以下のテキストを感情分析してください:\n\n{text}"}
                ]
            )
            
            # JSONパース
            try:
                sentiment_scores = json.loads(response.content[0].text)
                sentiment_scores["revenue"] = row[revenue_col]
                sentiment_scores["text_sample"] = text[:100] + "..."
                sentiment_results.append(sentiment_scores)
            except json.JSONDecodeError:
                st.warning(f"行{idx+1}: JSON解析エラー - スキップします")
                continue
            
            progress_bar.progress((idx + 1) / len(sample_data))
            
            # API呼び出し間隔調整
            time.sleep(0.1)
            
        except Exception as e:
            st.warning(f"行{idx+1}: API呼び出しエラー - {e}")
            continue
    
    if not sentiment_results:
        st.error("感情分析結果が得られませんでした。")
        st.stop()
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(sentiment_results)
    
    status_text.text("✅ 感情分析完了！")
    progress_bar.progress(1.0)
    
    st.success(f"🎉 {len(results_df)}件の感情分析が完了しました！")
    
    # 相関分析
    st.subheader("📈 感情-収益相関分析")
    
    correlation_results = []
    
    for emotion in emotion_categories:
        if emotion in results_df.columns:
            # ピアソン相関
            pearson_corr, pearson_p = pearsonr(results_df[emotion], results_df["revenue"])
            # スピアマン相関
            spearman_corr, spearman_p = spearmanr(results_df[emotion], results_df["revenue"])
            
            correlation_results.append({
                "感情": emotion,
                "ピアソン相関": pearson_corr,
                "ピアソンp値": pearson_p,
                "スピアマン相関": spearman_corr,
                "スピアマンp値": spearman_p,
                "統計的有意性": "有意" if pearson_p < 0.05 else "非有意"
            })
    
    corr_df = pd.DataFrame(correlation_results).sort_values("ピアソン相関", key=abs, ascending=False)
    
    # 結果表示
    st.dataframe(
        corr_df.style.format({
            "ピアソン相関": "{:.3f}",
            "ピアソンp値": "{:.3f}",
            "スピアマン相関": "{:.3f}",
            "スピアマンp値": "{:.3f}"
        }),
        use_container_width=True
    )
    
    # 可視化
    st.subheader("📊 相関可視化")
    
    # 相関バープロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ピアソン相関
    ax1.barh(corr_df["感情"], corr_df["ピアソン相関"])
    ax1.set_xlabel("相関係数")
    ax1.set_title("感情-収益 ピアソン相関")
    ax1.axvline(0, color="black", linestyle="-", alpha=0.5)
    
    # スピアマン相関
    ax2.barh(corr_df["感情"], corr_df["スピアマン相関"])
    ax2.set_xlabel("相関係数")
    ax2.set_title("感情-収益 スピアマン相関")
    ax2.axvline(0, color="black", linestyle="-", alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 感情スコア分布
    st.subheader("📊 感情スコア分布")
    
    emotion_means = results_df[emotion_categories].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(emotion_means.index, emotion_means.values)
    ax.set_ylabel("平均スコア")
    ax.set_title("感情別平均スコア")
    ax.set_ylim(0, 10)
    plt.xticks(rotation=45)
    
    # バーの色を相関の強さで色分け
    colors = ['red' if corr_df.loc[corr_df["感情"] == emotion, "ピアソン相関"].iloc[0] > 0 else 'blue' 
              for emotion in emotion_means.index]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 最高相関感情の散布図
    if len(corr_df) > 0:
        best_emotion = corr_df.iloc[0]["感情"]
        best_corr = corr_df.iloc[0]["ピアソン相関"]
        
        st.subheader(f"🎯 最高相関感情: {best_emotion} (r={best_corr:.3f})")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(results_df[best_emotion], results_df["revenue"], alpha=0.6)
        ax.set_xlabel(f"{best_emotion}スコア")
        ax.set_ylabel("収益")
        ax.set_title(f"{best_emotion}スコア vs 収益")
        
        # トレンドライン
        z = np.polyfit(results_df[best_emotion], results_df["revenue"], 1)
        p = np.poly1d(z)
        ax.plot(results_df[best_emotion], p(results_df[best_emotion]), "r--", alpha=0.8)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 結果要約
    st.subheader("📋 分析結果要約")
    
    significant_emotions = corr_df[corr_df["統計的有意性"] == "有意"]
    
    if len(significant_emotions) > 0:
        st.success(f"✅ {len(significant_emotions)}個の感情で統計的に有意な相関を発見")
        
        for _, row in significant_emotions.head(3).iterrows():
            correlation_strength = "強い" if abs(row["ピアソン相関"]) > 0.5 else "中程度" if abs(row["ピアソン相関"]) > 0.3 else "弱い"
            correlation_direction = "正の" if row["ピアソン相関"] > 0 else "負の"
            
            st.write(f"• **{row['感情']}**: {correlation_direction}{correlation_strength}相関 (r={row['ピアソン相関']:.3f}, p={row['ピアソンp値']:.3f})")
    else:
        st.warning("⚠️ 統計的に有意な相関は見つかりませんでした")
    
    # 収益分布
    st.subheader("💰 収益分布")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(results_df["revenue"], bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("収益")
    ax.set_ylabel("頻度")
    ax.set_title("収益分布")
    plt.tight_layout()
    st.pyplot(fig)
    
    # CSVダウンロード
    st.subheader("💾 結果ダウンロード")
    
    # 詳細結果CSV
    detailed_csv = results_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📁 詳細分析結果をCSVダウンロード",
        data=detailed_csv,
        file_name=f"sentiment_analysis_detailed_{analysis_type.replace('（', '_').replace('）', '')}.csv",
        mime="text/csv"
    )
    
    # 相関結果CSV
    correlation_csv = corr_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📊 相関分析結果をCSVダウンロード",
        data=correlation_csv,
        file_name=f"sentiment_correlation_{analysis_type.replace('（', '_').replace('）', '')}.csv",
        mime="text/csv"
    )

# 使用方法説明
with st.expander("ℹ️ 使用方法とヒント"):
    st.markdown("""
    ### 🎯 機能概要
    - **感情分析**: Anthropic Claude APIを使用して台本テキストの感情を分析
    - **相関分析**: 感情スコアと収益の相関関係を統計的に検証
    - **可視化**: 相関関係、感情分布、散布図を表示
    - **エクスポート**: 分析結果をCSV形式でダウンロード可能
    
    ### ⚙️ 設定のヒント
    - **分析タイプ**: 基本感情（5分類）は簡潔、詳細感情（10分類）は精密
    - **最大分析件数**: API使用量とコストを考慮して調整
    - **統計的有意性**: p値 < 0.05 で相関が統計的に意味あり
    
    ### 📊 結果の解釈
    - **相関係数の目安**: |r| > 0.5 (強い)、|r| > 0.3 (中程度)、|r| > 0.1 (弱い)
    - **正の相関**: 感情スコアが高いほど収益も高い
    - **負の相関**: 感情スコアが高いほど収益は低い
    """)

st.markdown("---")
st.caption("🧠 Anthropic Claude APIによる感情分析 | 📊 台本データ分析ハブ")