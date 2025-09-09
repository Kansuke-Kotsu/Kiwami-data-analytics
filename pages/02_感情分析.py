import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import re
import io

st.set_page_config(page_title="② 感情分析 ", page_icon="💖", layout="wide")
st.title("💖 感情分析による収益相関分析")

# セッション状態チェック
if "df" not in st.session_state:
    st.warning("まずトップページでExcelをアップロードしてください。")
    st.stop()

df = st.session_state["df"].copy()
meta = st.session_state.get("meta", {})

# oseti ライブラリの確認とインポート
try:
    import oseti
    
    # MeCab設定エラーのハンドリング
    try:
        analyzer = oseti.Analyzer()
    except RuntimeError as e:
        if "mecabrc" in str(e).lower():
            st.warning("⚠️ MeCabの設定ファイルが見つかりません。代替方法を試します...")
            
            # 複数の代替設定を試行
            mecab_configs = [
                "",  # デフォルト
                "-r ''",  # 空のrc設定
                "-r /dev/null",  # 無効化
                "-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd",  # 辞書パス指定
                "-d /usr/local/lib/mecab/dic/ipadic",  # 標準辞書
            ]
            
            analyzer = None
            for config in mecab_configs:
                try:
                    if config == "":
                        # MeCabのパッケージが正しくインストールされていない場合の警告
                        st.info("🔧 MeCab設定を自動調整中...")
                        analyzer = oseti.Analyzer(mecab_args="-r ''")
                    else:
                        analyzer = oseti.Analyzer(mecab_args=config)
                    st.success("✅ MeCab設定が正常に構成されました！")
                    break
                except:
                    continue
            
            if analyzer is None:
                st.error("💔 MeCabの設定に失敗しました。以下の手順をお試しください：")
                st.code("""
# macOSの場合:
brew install mecab mecab-ipadic

# Linuxの場合:
sudo apt-get install mecab mecab-ipadic-utf8

# MeCab辞書の再インストール:
pip uninstall mecab-python3
pip install mecab-python3
                """)
                st.info("💡 または、代替として感情分析LLM版をご利用ください。")
                st.stop()
        else:
            # その他のMeCabエラー
            st.error(f"💔 MeCabエラー: {str(e)}")
            st.info("💡 感情分析LLM版のご利用をお勧めします。")
            st.stop()
    
except ImportError:
    st.error("💔 osetiライブラリがインストールされていません。")
    st.code("pip install oseti")
    st.info("requirements.txtにosetiを追加して再起動してください。")
    st.stop()
except Exception as e:
    st.error(f"💔 予期しないエラーが発生しました: {str(e)}")
    st.info("💡 感情分析LLM版のご利用をお勧めします。")
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
    revenue_options = [c for c in df.columns if c != script_col]
    default_idx = revenue_options.index(meta.get("profit_col")) if meta.get("profit_col") in revenue_options else 0
    revenue_col = st.selectbox(
        "収益データ列",
        options=revenue_options,
        index=default_idx
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

# 分析設定
st.subheader("⚙️ 感情分析設定")

col1, col2 = st.columns(2)
with col1:
    max_samples = st.slider(
        "最大分析件数（処理速度調整）",
        min_value=10,
        max_value=min(2000, len(valid_data)),
        value=min(500, len(valid_data)),
        step=50
    )

with col2:
    text_preprocessing = st.selectbox(
        "テキスト前処理",
        ["基本前処理", "詳細前処理"]
    )

# サンプリング
if len(valid_data) > max_samples:
    sample_data = valid_data.sample(n=max_samples, random_state=42)
    st.info(f"🎯 {max_samples}件をランダムサンプリングしました")
else:
    sample_data = valid_data
    st.info(f"📋 全{len(sample_data)}件を分析します")

# テキスト前処理関数
def preprocess_text(text, mode="basic"):
    """テキストの前処理"""
    if mode == "詳細前処理":
        # URL、メール、数字を除去
        text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
        text = re.sub(r'\d+', '', text)
        # 記号の一部を除去
        text = re.sub(r'[【】「」『』（）()[\]{}]', '', text)
        # 改行とタブを空白に
        text = re.sub(r'[\r\n\t]+', ' ', text)
        # 連続する空白を1つに
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# 感情分析関数
def analyze_sentiment_batch(texts, preprocessing_mode="basic"):
    """バッチ感情分析"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, text in enumerate(texts):
        try:
            # テキスト前処理
            processed_text = preprocess_text(text, preprocessing_mode)
            
            if len(processed_text.strip()) == 0:
                # 空のテキストの場合はニュートラルスコア
                sentiment_scores = {
                    "positive": 0.0,
                    "negative": 0.0, 
                    "neutral": 1.0,
                    "compound": 0.0
                }
            else:
                # oseti による感情分析
                scores = analyzer.analyze(processed_text)
                # scores は list（各文のスコア）なので、全体の複合スコアを平均で集約
                if isinstance(scores, (list, tuple, np.ndarray)):
                    if len(scores) == 0:
                        compound_score = 0.0
                    else:
                        compound_score = float(np.mean(scores))
                else:
                    # 稀に単一数値が返ってきても安全に処理
                    compound_score = float(scores)

                # compound_score は -1〜1 を取りうる想定
                # シンプルに「正／負／中立」を割り当て（合計が1になるように）
                if compound_score > 0.1:
                    positive = compound_score          # 例: 0.7 → positive=0.7, neutral=0.3
                    negative = 0.0
                    neutral  = 1.0 - positive
                elif compound_score < -0.1:
                    positive = 0.0
                    negative = -compound_score         # 例: -0.6 → negative=0.6, neutral=0.4
                    neutral  = 1.0 - negative
                else:
                    positive = 0.0
                    negative = 0.0
                    neutral  = 1.0

                sentiment_scores = {
                    "positive": positive,
                    "negative": negative,
                    "neutral": neutral,
                    "compound": compound_score
                }
            
            results.append(sentiment_scores)
            
            # プログレス更新
            progress = (idx + 1) / len(texts)
            progress_bar.progress(progress)
            status_text.text(f"分析中... {idx+1}/{len(texts)} ({progress*100:.1f}%)")
            
        except Exception as e:
            # エラーの場合はニュートラルスコア
            st.warning(f"テキスト{idx+1}の分析でエラー: {e}")
            results.append({
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "compound": 0.0
            })
            continue
    
    status_text.text("✅ 感情分析完了！")
    progress_bar.progress(1.0)
    
    return results

# 感情分析実行
if st.button("🚀 感情分析実行", type="primary"):
    
    st.info("💖 osetiライブラリによる感情分析を開始します")
    
    # テキストリスト準備
    texts = sample_data[script_col].astype(str).tolist()
    
    # バッチ感情分析実行
    sentiment_results = analyze_sentiment_batch(texts, text_preprocessing)
    
    # 結果をDataFrameに結合
    results_df = pd.DataFrame(sentiment_results)
    results_df["revenue"] = sample_data[revenue_col].values
    results_df["text_sample"] = [text[:100] + "..." for text in texts]
    
    # 感情判定（ポジティブ/ネガティブ）の追加
    results_df["sentiment_label"] = results_df["compound"].apply(
        lambda x: "ポジティブ" if x > 0.1 else "ネガティブ" if x < -0.1 else "中性"
    )
    
    st.success(f"🎉 {len(results_df)}件の感情分析が完了しました！")
    
    # 各台本データの感情判定一覧表示
    st.subheader("📋 各台本データの感情判定一覧")
    
    # 感情判定結果のサマリー
    sentiment_counts = results_df["sentiment_label"].value_counts()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ポジティブ", f"{sentiment_counts.get('ポジティブ', 0)}件", 
                 f"{sentiment_counts.get('ポジティブ', 0) / len(results_df) * 100:.1f}%")
    with col2:
        st.metric("ネガティブ", f"{sentiment_counts.get('ネガティブ', 0)}件",
                 f"{sentiment_counts.get('ネガティブ', 0) / len(results_df) * 100:.1f}%")
    with col3:
        st.metric("中性", f"{sentiment_counts.get('中性', 0)}件",
                 f"{sentiment_counts.get('中性', 0) / len(results_df) * 100:.1f}%")
    
    # 台本データと感情判定結果の一覧表
    display_df = pd.DataFrame({
        "番号": range(1, len(results_df) + 1),
        "台本データ（抜粋）": results_df["text_sample"],
        "感情判定": results_df["sentiment_label"],
        "総合スコア": results_df["compound"].round(3),
        "ポジティブ": results_df["positive"].round(3),
        "ネガティブ": results_df["negative"].round(3),
        "収益": results_df["revenue"]
    })
    
    # 感情別の色分けを適用
    def highlight_sentiment(row):
        if row["感情判定"] == "ポジティブ":
            return ['background-color: #e6ffe6'] * len(row)
        elif row["感情判定"] == "ネガティブ":
            return ['background-color: #ffe6e6'] * len(row)
        else:
            return ['background-color: #f5f5f5'] * len(row)
    
    st.dataframe(
        display_df.style.apply(highlight_sentiment, axis=1),
        use_container_width=True,
        height=400
    )
    
    # フィルタリング機能
    st.subheader("🔍 感情別フィルタリング")
    
    filter_emotion = st.selectbox(
        "表示する感情を選択:",
        options=["全て"] + list(sentiment_counts.index)
    )
    
    if filter_emotion != "全て":
        filtered_df = display_df[display_df["感情判定"] == filter_emotion]
        st.write(f"**{filter_emotion}の台本データ ({len(filtered_df)}件):**")
        st.dataframe(
            filtered_df.style.apply(highlight_sentiment, axis=1),
            use_container_width=True,
            height=300
        )
        
        # フィルタリング結果の統計
        if len(filtered_df) > 0:
            avg_revenue = filtered_df["収益"].mean()
            avg_score = filtered_df["総合スコア"].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{filter_emotion}の平均収益", f"{avg_revenue:.2f}")
            with col2:
                st.metric(f"{filter_emotion}の平均感情スコア", f"{avg_score:.3f}")
    
    # 基本統計表示
    st.subheader("📊 感情スコア基本統計")
    
    emotion_stats = results_df[["positive", "negative", "neutral", "compound"]].describe()
    st.dataframe(emotion_stats.round(3))
    
    # 相関分析
    st.subheader("📈 感情-収益相関分析")
    
    emotion_cols = ["positive", "negative", "neutral", "compound"]
    correlation_results = []
    
    for emotion in emotion_cols:
        # データの有効性をチェック
        emotion_data = results_df[emotion].values
        revenue_data = results_df["revenue"].values
        
        # 定数配列やNaN値をチェック
        if (np.std(emotion_data) == 0 or np.std(revenue_data) == 0 or 
            np.isnan(emotion_data).all() or np.isnan(revenue_data).all()):
            # 定数配列の場合は相関係数を0とする
            pearson_corr, pearson_p = 0.0, 1.0
            spearman_corr, spearman_p = 0.0, 1.0
        else:
            try:
                # ピアソン相関
                pearson_corr, pearson_p = pearsonr(emotion_data, revenue_data)
                # スピアマン相関  
                spearman_corr, spearman_p = spearmanr(emotion_data, revenue_data)
                
                # NaN値の処理
                if np.isnan(pearson_corr):
                    pearson_corr, pearson_p = 0.0, 1.0
                if np.isnan(spearman_corr):
                    spearman_corr, spearman_p = 0.0, 1.0
                    
            except Exception as e:
                st.warning(f"相関計算エラー ({emotion}): {str(e)}")
                pearson_corr, pearson_p = 0.0, 1.0
                spearman_corr, spearman_p = 0.0, 1.0
        
        emotion_names = {
            "positive": "ポジティブ",
            "negative": "ネガティブ", 
            "neutral": "中性",
            "compound": "総合感情"
        }
        
        correlation_results.append({
            "感情": emotion_names[emotion],
            "ピアソン相関": pearson_corr,
            "ピアソンp値": pearson_p,
            "スピアマン相関": spearman_corr,
            "スピアマンp値": spearman_p,
            "統計的有意性": "有意" if pearson_p < 0.05 else "非有意"
        })
    
    corr_df = pd.DataFrame(correlation_results).sort_values("ピアソン相関", key=abs, ascending=False)
    
    # 相関結果表示
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
    bars1 = ax1.barh(corr_df["感情"], corr_df["ピアソン相関"])
    ax1.set_xlabel("相関係数")
    ax1.set_title("感情-収益 ピアソン相関")
    ax1.axvline(0, color="black", linestyle="-", alpha=0.5)
    
    # バーの色付け
    for bar, corr in zip(bars1, corr_df["ピアソン相関"]):
        bar.set_color("red" if corr > 0 else "blue")
        bar.set_alpha(0.7)
    
    # スピアマン相関
    bars2 = ax2.barh(corr_df["感情"], corr_df["スピアマン相関"])
    ax2.set_xlabel("相関係数")
    ax2.set_title("感情-収益 スピアマン相関")
    ax2.axvline(0, color="black", linestyle="-", alpha=0.5)
    
    # バーの色付け
    for bar, corr in zip(bars2, corr_df["スピアマン相関"]):
        bar.set_color("red" if corr > 0 else "blue")
        bar.set_alpha(0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 感情スコア分布
    st.subheader("📊 感情スコア分布")
    
    emotion_means = results_df[["positive", "negative", "neutral", "compound"]].mean()
    emotion_names_jp = ["ポジティブ", "ネガティブ", "中性", "総合感情"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(emotion_names_jp, emotion_means.values)
    ax.set_ylabel("平均スコア")
    ax.set_title("感情別平均スコア")
    ax.set_ylim(0, 1)
    
    # バーの色を相関の強さで色分け
    colors = ['red' if corr_df.iloc[i]["ピアソン相関"] > 0 else 'blue' 
              for i in range(len(emotion_means))]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 最高相関感情の散布図
    if len(corr_df) > 0:
        best_emotion_jp = corr_df.iloc[0]["感情"]
        
        # 感情名から英語カラム名へのマッピング
        emotion_name_mapping = {
            "ポジティブ": "positive",
            "ネガティブ": "negative", 
            "中性": "neutral",
            "総合感情": "compound"
        }
        best_emotion_col = emotion_name_mapping[best_emotion_jp]
        best_corr = corr_df.iloc[0]["ピアソン相関"]
        
        st.subheader(f"🎯 最高相関感情: {best_emotion_jp} (r={best_corr:.3f})")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(results_df[best_emotion_col], results_df["revenue"], 
                           alpha=0.6, c=results_df[best_emotion_col], cmap="viridis")
        ax.set_xlabel(f"{best_emotion_jp}スコア")
        ax.set_ylabel("収益")
        ax.set_title(f"{best_emotion_jp}スコア vs 収益")
        
        # トレンドライン（エラーハンドリング付き）
        try:
            x_data = results_df[best_emotion_col].values
            y_data = results_df["revenue"].values
            
            # データの有効性をチェック
            if (np.std(x_data) > 1e-10 and np.std(y_data) > 1e-10 and 
                not np.isnan(x_data).any() and not np.isnan(y_data).any() and 
                len(x_data) > 1):
                
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=2)
            else:
                st.info(f"📝 {best_emotion_jp}データに一定値が多いため、トレンドラインを省略します。")
                
        except Exception as e:
            st.warning(f"トレンドライン描画エラー: {str(e)}")
            st.info("💡 データに数値的な問題があるため、トレンドラインなしで表示します。")
        
        plt.colorbar(scatter, label=f"{best_emotion_jp}スコア")
        plt.tight_layout()
        st.pyplot(fig)
    
    # 感情分布のヒートマップ
    st.subheader("🌡️ 感情スコア相関マトリックス")
    
    emotion_corr = results_df[["positive", "negative", "neutral", "compound", "revenue"]].corr()
    emotion_corr.columns = ["ポジティブ", "ネガティブ", "中性", "総合感情", "収益"]
    emotion_corr.index = ["ポジティブ", "ネガティブ", "中性", "総合感情", "収益"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(emotion_corr, annot=True, cmap="coolwarm", center=0, 
                square=True, fmt=".3f", ax=ax)
    ax.set_title("感情スコア相関マトリックス")
    plt.tight_layout()
    st.pyplot(fig)
    
    # 結果要約
    st.subheader("📋 分析結果要約")
    
    significant_emotions = corr_df[corr_df["統計的有意性"] == "有意"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("分析件数", f"{len(results_df):,}件")
        st.metric("有意な相関数", f"{len(significant_emotions)}個")
        
    with col2:
        strongest_corr = corr_df.iloc[0]
        st.metric("最強相関", f"{strongest_corr['感情']}")
        st.metric("相関係数", f"{strongest_corr['ピアソン相関']:.3f}")
    
    if len(significant_emotions) > 0:
        st.success(f"✅ {len(significant_emotions)}個の感情で統計的に有意な相関を発見")
        
        for _, row in significant_emotions.iterrows():
            correlation_strength = "強い" if abs(row["ピアソン相関"]) > 0.5 else "中程度" if abs(row["ピアソン相関"]) > 0.3 else "弱い"
            correlation_direction = "正の" if row["ピアソン相関"] > 0 else "負の"
            
            st.write(f"• **{row['感情']}**: {correlation_direction}{correlation_strength}相関 (r={row['ピアソン相関']:.3f}, p={row['ピアソンp値']:.3f})")
    else:
        st.warning("⚠️ 統計的に有意な相関は見つかりませんでした")
    
    # 感情別収益分析
    st.subheader("💰 感情別収益分析")
    
    # 総合感情スコアで上位・下位を分類
    compound_median = results_df["compound"].median()
    
    high_sentiment = results_df[results_df["compound"] >= compound_median]
    low_sentiment = results_df[results_df["compound"] < compound_median]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "高感情グループ平均収益", 
            f"{high_sentiment['revenue'].mean():.2f}",
            f"{high_sentiment['revenue'].mean() - results_df['revenue'].mean():.2f}"
        )
        st.write(f"件数: {len(high_sentiment)}件")
        
    with col2:
        st.metric(
            "低感情グループ平均収益",
            f"{low_sentiment['revenue'].mean():.2f}",
            f"{low_sentiment['revenue'].mean() - results_df['revenue'].mean():.2f}"
        )
        st.write(f"件数: {len(low_sentiment)}件")
    
    # 収益分布比較
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist([high_sentiment["revenue"], low_sentiment["revenue"]], 
            bins=20, alpha=0.7, label=["高感情", "低感情"], color=["red", "blue"])
    ax.set_xlabel("収益")
    ax.set_ylabel("頻度")
    ax.set_title("感情グループ別収益分布")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # CSVダウンロード
    st.subheader("💾 結果ダウンロード")
    
    # 詳細結果CSV（感情判定ラベルを含む）
    download_df = results_df.copy()
    download_df = download_df[["text_sample", "sentiment_label", "compound", "positive", "negative", "neutral", "revenue"]]
    download_df.columns = ["台本データ（抜粋）", "感情判定", "総合スコア", "ポジティブスコア", "ネガティブスコア", "中性スコア", "収益"]
    
    detailed_csv = download_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📁 詳細分析結果をCSVダウンロード",
        data=detailed_csv,
        file_name="sentiment_analysis_detailed_with_labels.csv",
        mime="text/csv"
    )
    
    # 感情判定一覧CSV
    sentiment_list_csv = display_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📋 感情判定一覧をCSVダウンロード",
        data=sentiment_list_csv,
        file_name="sentiment_judgment_list.csv",
        mime="text/csv"
    )
    
    # 相関結果CSV
    correlation_csv = corr_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📊 相関分析結果をCSVダウンロード", 
        data=correlation_csv,
        file_name="sentiment_correlation_free.csv",
        mime="text/csv"
    )

# 使用方法説明
with st.expander("ℹ️ 使用方法とヒント"):
    st.markdown("""
    ### 🎯 機能概要
    - **感情分析**: osetiライブラリによる日本語感情分析
    - **相関分析**: 感情スコアと収益の相関関係を統計的に検証
    - **可視化**: 相関関係、感情分布、ヒートマップを表示
    - **エクスポート**: 分析結果をCSV形式でダウンロード可能
    
    ### 💖 osetiライブラリについて
    - 日本語専用の感情分析ライブラリ
    - APIキー不要、完全無料で利用可能
    - -1（ネガティブ）から1（ポジティブ）のスコアを出力
    - 辞書ベースの感情分析手法
    
    ### ⚙️ 設定のヒント
    - **基本前処理**: 軽微なクリーニングのみ
    - **詳細前処理**: URL、記号、数字を除去してより精密に分析
    - **最大分析件数**: 処理速度を考慮して調整（osetiは高速）
    
    ### 📊 結果の解釈
    - **positive**: ポジティブ感情の強さ（0-1）
    - **negative**: ネガティブ感情の強さ（0-1）
    - **neutral**: 中性的感情の強さ（0-1）
    - **compound**: 総合感情スコア（-1から1）
    - **統計的有意性**: p値 < 0.05 で相関が統計的に意味あり
    
    ### 🆚 LLM版との比較
    - **LLM版**: より複雑な感情分析、API料金が発生
    - **無料版**: 高速処理、API料金なし、基本的な感情分析
    """)

st.markdown("---")
st.caption("💖 oseti ライブラリによる感情分析 | 📊 台本データ分析ハブ")