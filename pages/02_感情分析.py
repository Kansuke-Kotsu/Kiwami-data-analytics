import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
import anthropic
from typing import List, Dict
import time

st.set_page_config(page_title="② 感情分析", page_icon="😊", layout="wide")
st.title("😊 感情分析（収益との相関分析）")

if "df" not in st.session_state:
    st.warning("まずトップページでExcelをアップロードしてください。")
    st.stop()

df = st.session_state["df"].copy()
meta = st.session_state.get("meta", {})

# Anthropic API キーの確認
if 'ANTHROPIC_API_KEY' not in os.environ:
    st.error("Anthropic API キーが設定されていません。環境変数 'ANTHROPIC_API_KEY' を設定してください。")
    st.stop()

# 列選択
st.subheader("📊 データ選択")
text_col = st.selectbox(
    "台本テキスト列", 
    options=list(df.columns), 
    index=list(df.columns).index(meta.get("text_col")) if meta.get("text_col") in df.columns else 0
)
profit_col = st.selectbox(
    "収益列", 
    options=[c for c in df.columns if c != text_col], 
    index=[i for i, c in enumerate(df.columns) if c != text_col and c == meta.get("profit_col")][0] if meta.get("profit_col") in df.columns else 0
)

# データ前処理
df["_text"] = df[text_col].fillna("").astype(str)
df["_profit"] = pd.to_numeric(df[profit_col], errors="coerce")

# 有効データの確認
valid_data = df[(df["_text"].str.len() > 10) & df["_profit"].notna()].copy()
if len(valid_data) < 3:
    st.error("有効なデータが不足しています。テキストが10文字以上で、収益データが数値である行が3行以上必要です。")
    st.stop()

st.success(f"✅ 分析対象データ: {len(valid_data)}件")

# サンプリング設定
st.subheader("⚙️ 分析設定")
col1, col2 = st.columns(2)

with col1:
    max_samples = st.slider(
        "分析するサンプル数（API使用量調整）",
        min_value=3,
        max_value=min(100, len(valid_data)),
        value=min(20, len(valid_data)),
        help="多いほど正確だが、API使用量とコストが増加します"
    )

with col2:
    sentiment_model = st.selectbox(
        "感情分析の詳細度",
        options=["基本感情（5分類）", "詳細感情（10分類）"],
        index=0,
        help="詳細にするほどAPI使用量が増加します"
    )

# サンプルデータの準備
if len(valid_data) > max_samples:
    # 収益の分布を考慮したサンプリング
    sample_data = valid_data.sample(n=max_samples, random_state=42)
else:
    sample_data = valid_data.copy()

# 感情分析の実行
def analyze_sentiment_batch(texts: List[str], model_type: str) -> List[Dict]:
    """テキストの感情分析を実行"""
    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    
    if model_type == "基本感情（5分類）":
        emotions = ["ポジティブ", "ネガティブ", "中性", "興奮", "不安"]
        prompt_template = """以下の台本テキストの感情を分析してください。

テキスト: "{text}"

以下の5つの感情カテゴリーそれぞれについて、0-10のスコアで評価してください:
- ポジティブ（喜び、希望、満足など）
- ネガティブ（悲しみ、怒り、不満など）  
- 中性（落ち着き、平常、客観的など）
- 興奮（驚き、高揚、熱狂など）
- 不安（心配、恐怖、緊張など）

以下のJSON形式で回答してください:
{{"ポジティブ": スコア, "ネガティブ": スコア, "中性": スコア, "興奮": スコア, "不安": スコア}}"""
    else:
        emotions = ["喜び", "悲しみ", "怒り", "恐怖", "驚き", "嫌悪", "期待", "信頼", "中性", "混合"]
        prompt_template = """以下の台本テキストの感情を詳細分析してください。

テキスト: "{text}"

以下の10の感情カテゴリーそれぞれについて、0-10のスコアで評価してください:
- 喜び、悲しみ、怒り、恐怖、驚き、嫌悪、期待、信頼、中性、混合

以下のJSON形式で回答してください:
{{"喜び": スコア, "悲しみ": スコア, "怒り": スコア, "恐怖": スコア, "驚き": スコア, "嫌悪": スコア, "期待": スコア, "信頼": スコア, "中性": スコア, "混合": スコア}}"""
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text in enumerate(texts):
        try:
            status_text.text(f"感情分析中... ({i+1}/{len(texts)})")
            
            # テキストが長すぎる場合は先頭部分を使用
            text_to_analyze = text[:1000] if len(text) > 1000 else text
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": prompt_template.format(text=text_to_analyze)
                }]
            )
            
            # レスポンスのパース
            response_text = response.content[0].text
            # JSONの抽出を試行
            import json
            try:
                # JSONブロックを探す
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    sentiment_scores = json.loads(json_str)
                else:
                    raise ValueError("JSON not found")
            except:
                # パースに失敗した場合はデフォルト値
                sentiment_scores = {emotion: 5.0 for emotion in emotions}
                st.warning(f"テキスト {i+1} の感情分析結果をパースできませんでした。デフォルト値を使用します。")
            
            results.append(sentiment_scores)
            progress_bar.progress((i + 1) / len(texts))
            
            # API制限を考慮した短い待機
            time.sleep(0.5)
            
        except Exception as e:
            st.error(f"感情分析でエラーが発生しました (テキスト {i+1}): {str(e)}")
            # エラーの場合はデフォルト値
            sentiment_scores = {emotion: 5.0 for emotion in emotions}
            results.append(sentiment_scores)
    
    status_text.text("感情分析完了!")
    progress_bar.progress(1.0)
    return results

# 実行ボタン
if st.button("🚀 感情分析を実行", type="primary"):
    with st.spinner("感情分析を実行中..."):
        # テキストリストの準備
        texts_to_analyze = sample_data["_text"].tolist()
        
        # 感情分析の実行
        sentiment_results = analyze_sentiment_batch(texts_to_analyze, sentiment_model)
        
        # 結果をDataFrameに統合
        sentiment_df = pd.DataFrame(sentiment_results)
        analysis_df = pd.concat([
            sample_data.reset_index(drop=True),
            sentiment_df
        ], axis=1)
        
        # 結果の保存
        st.session_state["sentiment_analysis_results"] = analysis_df
        st.session_state["sentiment_emotions"] = list(sentiment_df.columns)
        
        st.success("✅ 感情分析が完了しました！")

# 結果の表示
if "sentiment_analysis_results" in st.session_state:
    analysis_df = st.session_state["sentiment_analysis_results"]
    emotions = st.session_state["sentiment_emotions"]
    
    st.subheader("📈 分析結果")
    
    # 基本統計
    st.markdown("#### 感情スコア統計")
    emotion_stats = analysis_df[emotions].describe().round(2)
    st.dataframe(emotion_stats)
    
    # 相関分析
    st.markdown("#### 感情と収益の相関分析")
    
    correlations = []
    for emotion in emotions:
        try:
            pearson_r, pearson_p = pearsonr(analysis_df[emotion], analysis_df["_profit"])
            spearman_r, spearman_p = spearmanr(analysis_df[emotion], analysis_df["_profit"])
            
            correlations.append({
                "感情": emotion,
                "ピアソン相関": round(pearson_r, 3),
                "ピアソンp値": round(pearson_p, 4),
                "スピアマン相関": round(spearman_r, 3),
                "スピアマンp値": round(spearman_p, 4),
                "有意性": "有意" if pearson_p < 0.05 else "非有意"
            })
        except:
            correlations.append({
                "感情": emotion,
                "ピアソン相関": 0,
                "ピアソンp値": 1,
                "スピアマン相関": 0,
                "スピアマンp値": 1,
                "有意性": "計算エラー"
            })
    
    corr_df = pd.DataFrame(correlations)
    st.dataframe(corr_df)
    
    # 可視化
    st.markdown("#### 📊 可視化")
    
    # 相関係数のヒートマップ
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.rcParams['font.family'] = ['DejaVu Sans']
    
    # 1. 相関係数バープロット
    axes[0, 0].barh(corr_df["感情"], corr_df["ピアソン相関"])
    axes[0, 0].set_title("感情と収益の相関係数")
    axes[0, 0].set_xlabel("ピアソン相関係数")
    axes[0, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 2. 感情スコア分布
    emotion_means = analysis_df[emotions].mean()
    axes[0, 1].bar(range(len(emotion_means)), emotion_means.values)
    axes[0, 1].set_title("平均感情スコア")
    axes[0, 1].set_xticks(range(len(emotions)))
    axes[0, 1].set_xticklabels(emotions, rotation=45, ha='right')
    axes[0, 1].set_ylabel("平均スコア")
    
    # 3. 散布図（最も相関の高い感情）
    best_emotion = corr_df.loc[corr_df["ピアソン相関"].abs().idxmax(), "感情"]
    axes[1, 0].scatter(analysis_df[best_emotion], analysis_df["_profit"], alpha=0.6)
    axes[1, 0].set_xlabel(f"{best_emotion} スコア")
    axes[1, 0].set_ylabel("収益")
    axes[1, 0].set_title(f"{best_emotion}と収益の散布図")
    
    # 4. 収益分布
    axes[1, 1].hist(analysis_df["_profit"], bins=10, alpha=0.7)
    axes[1, 1].set_title("収益分布")
    axes[1, 1].set_xlabel("収益")
    axes[1, 1].set_ylabel("頻度")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # データ詳細表示
    with st.expander("📋 詳細データ", expanded=False):
        display_cols = [text_col, profit_col] + emotions
        st.dataframe(analysis_df[display_cols])
    
    # 結果のダウンロード
    csv_data = analysis_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 分析結果をCSVダウンロード",
        data=csv_data,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv"
    )
    
    # 主要な発見の要約
    st.markdown("#### 🔍 主要な発見")
    
    # 最も相関の高い感情
    max_corr_idx = corr_df["ピアソン相関"].abs().idxmax()
    max_corr_emotion = corr_df.iloc[max_corr_idx]
    
    if abs(max_corr_emotion["ピアソン相関"]) > 0.3:
        correlation_strength = "強い"
    elif abs(max_corr_emotion["ピアソン相関"]) > 0.1:
        correlation_strength = "中程度"
    else:
        correlation_strength = "弱い"
    
    correlation_direction = "正の" if max_corr_emotion["ピアソン相関"] > 0 else "負の"
    
    st.write(f"""
    - **最も収益と関連の高い感情**: {max_corr_emotion["感情"]} (相関係数: {max_corr_emotion["ピアソン相関"]})
    - **相関の強さ**: {correlation_strength}{correlation_direction}相関
    - **統計的有意性**: {max_corr_emotion["有意性"]}
    - **分析サンプル数**: {len(analysis_df)}件
    """)
    
    if max_corr_emotion["有意性"] == "有意":
        direction_text = "高くなる" if max_corr_emotion["ピアソン相関"] > 0 else "低くなる"
        st.info(f"💡 **示唆**: 台本の「{max_corr_emotion['感情']}」スコアが高いほど、収益が{direction_text}傾向があります。")
    else:
        st.warning("⚠️ **注意**: 統計的に有意な相関は検出されませんでした。サンプル数を増やすか、他の要因を検討することをお勧めします。")

else:
    st.info("👆 上記の設定を完了後、「感情分析を実行」ボタンを押してください。")