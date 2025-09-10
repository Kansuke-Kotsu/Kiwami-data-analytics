import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import re
import io

# MeCab不要の簡易感情分析クラス
class SimpleSentimentAnalyzer:
    """MeCabに依存しない簡易感情分析クラス"""
    
    def __init__(self):
        # 基本的な感情語彙辞書（日本語）
        self.positive_words = {
            '嬉しい': 0.8, 'うれしい': 0.8, '楽しい': 0.7, 'たのしい': 0.7,
            '良い': 0.6, 'よい': 0.6, 'いい': 0.6, 'すごい': 0.7, 'すばらしい': 0.9,
            'すてき': 0.7, '素敵': 0.7, '最高': 0.9, '素晴らしい': 0.9, '感動': 0.8,
            '愛': 0.8, '好き': 0.7, '幸せ': 0.8, '成功': 0.7, '勝利': 0.8,
            'ありがとう': 0.7, 'がんばる': 0.6, '頑張る': 0.6, '笑顔': 0.7, 'おめでとう': 0.8,
            '面白い': 0.6, 'おもしろい': 0.6, 'かわいい': 0.6, 'きれい': 0.6, '美しい': 0.7,
            '安心': 0.6, '満足': 0.7, '充実': 0.7, '快適': 0.6, '平和': 0.6
        }
        
        self.negative_words = {
            '悲しい': -0.8, 'かなしい': -0.8, '辛い': -0.7, 'つらい': -0.7,
            '悪い': -0.6, 'わるい': -0.6, 'だめ': -0.6, 'ダメ': -0.6, '最悪': -0.9,
            '嫌い': -0.7, 'きらい': -0.7, '嫌': -0.6, '怒り': -0.8, '腹立つ': -0.7,
            '失敗': -0.7, '困る': -0.6, '不安': -0.7, '心配': -0.6, '疲れる': -0.5,
            '病気': -0.6, '痛い': -0.6, 'つまらない': -0.6, '退屈': -0.5, '面倒': -0.5,
            '危険': -0.7, '問題': -0.6, '困難': -0.7, '苦しい': -0.8, 'むかつく': -0.7,
            '絶望': -0.9, 'ストレス': -0.6, '不満': -0.6, '後悔': -0.7, '恐怖': -0.8
        }
        
        # 強調語の重み調整
        self.intensifiers = {
            'とても': 1.5, 'すごく': 1.4, '本当に': 1.3, 'めちゃくちゃ': 1.6,
            'かなり': 1.3, 'ものすごく': 1.5, '非常に': 1.4, '超': 1.4,
            '少し': 0.7, 'ちょっと': 0.6, 'やや': 0.8, '若干': 0.7
        }
        
        # 否定語
        self.negators = ['ない', 'ず', 'ぬ', 'でも', 'けど', 'が']
    
    def analyze(self, text):
        """簡易感情分析を実行"""
        if not text or not isinstance(text, str):
            return 0.0
        
        text = text.lower().strip()
        words = re.findall(r'[ひらがなカタカナ漢字一-龯]+', text)
        
        score = 0.0
        word_count = 0
        
        for i, word in enumerate(words):
            # ポジティブ語彙のチェック
            if word in self.positive_words:
                word_score = self.positive_words[word]
                word_count += 1
                
            # ネガティブ語彙のチェック  
            elif word in self.negative_words:
                word_score = self.negative_words[word]
                word_count += 1
            else:
                continue
            
            # 強調語の調整
            if i > 0 and words[i-1] in self.intensifiers:
                word_score *= self.intensifiers[words[i-1]]
            
            # 否定語の調整（簡易）
            negation_context = ' '.join(words[max(0, i-2):i])
            for neg in self.negators:
                if neg in negation_context:
                    word_score *= -0.8
                    break
            
            score += word_score
        
        # 正規化（-1から1の範囲に調整）
        if word_count == 0:
            return 0.0
        
        normalized_score = score / word_count
        return max(-1.0, min(1.0, normalized_score))

st.set_page_config(page_title="② 感情分析 ", page_icon="", layout="wide")
st.title("感情分析による収益相関分析")

# セッション状態チェック
if "df" not in st.session_state:
    st.warning("まずトップページでExcelをアップロードしてください。")
    st.stop()

df = st.session_state["df"].copy()
meta = st.session_state.get("meta", {})

# 感情分析器の初期化
analyzer = None
use_simple_mode = st.session_state.get("use_simple_sentiment", False)

# 簡易モードでの強制実行チェック
if "force_simple_mode" in st.query_params:
    use_simple_mode = True
    st.session_state.use_simple_sentiment = True

if use_simple_mode:
    # 簡易感情分析モードを使用
    analyzer = SimpleSentimentAnalyzer()
    st.info("🚀 簡易感情分析モード (MeCab不要) で動作中")
else:
    # oseti ライブラリの確認とインポート
    try:
        import oseti
        
        # MeCab設定エラーの堅牢なハンドリング
        def initialize_oseti_analyzer():
            """より堅牢なoseti Analyzer初期化"""
            
            # Streamlit環境用の拡張MeCab設定リスト
            mecab_configs = [
                # 最もシンプルな設定から試行
                "-r ''",                    # 空のrc設定
                "-r /dev/null",             # rc無効化
                "",                         # デフォルト設定
                "-Owakati",                 # 分かち書きモード
                "-r /etc/mecabrc",          # 一般的なLinux設定
                "-r /usr/local/etc/mecabrc",# macOS Homebrew設定
                "-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd",  # Ubuntu辞書
                "-d /usr/local/lib/mecab/dic/ipadic",  # macOS標準辞書
                "-d /usr/share/mecab/dic/ipadic",      # Debian/Ubuntu標準
                "-F%m ",                    # 最小出力フォーマット
            ]
            
            # 進捗表示
            progress_placeholder = st.empty()
            progress_placeholder.info("🔧 MeCab設定を自動調整中...")
            
            for i, config in enumerate(mecab_configs):
                try:
                    progress_placeholder.info(f"🔧 MeCab設定を試行中... ({i+1}/{len(mecab_configs)}) {config or 'デフォルト'}")
                    
                    if config == "":
                        test_analyzer = oseti.Analyzer()
                    else:
                        test_analyzer = oseti.Analyzer(mecab_args=config)
                    
                    # 簡単な動作テスト
                    test_result = test_analyzer.analyze("テスト")
                    if test_result is not None:
                        progress_placeholder.success(f"✅ MeCab設定が正常に構成されました！ (設定: {config or 'デフォルト'})")
                        return test_analyzer
                        
                except Exception as e:
                    # エラーログを詳細に記録（デバッグ用）
                    if st.secrets.get("debug_mode", False):
                        st.write(f"設定 `{config}` でエラー: {str(e)}")
                    continue
            
            # 全ての設定が失敗した場合
            progress_placeholder.empty()
            return None
    
    # MeCab初期化実行
    try:
        analyzer = initialize_oseti_analyzer()
        
        if analyzer is None:
            # 全て失敗した場合のフォールバック
            st.error("🚨 MeCabの設定に失敗しました")
            
            # Streamlit環境に特化したソリューション
            st.markdown("""
            ### 💡 Streamlit環境での解決方法
            
            #### **方法1: システムパッケージの確認**
            """)
            
            st.code("""
# 1. MeCabシステムパッケージのインストール確認
# macOS (Homebrew):
brew list mecab mecab-ipadic || brew install mecab mecab-ipadic

# Ubuntu/Debian:
apt list --installed | grep mecab || sudo apt-get install mecab mecab-ipadic-utf8

# CentOS/RHEL:
yum list installed | grep mecab || sudo yum install mecab mecab-ipadic
            """)
            
            st.markdown("#### **方法2: Python環境のリセット**")
            st.code("""
# 2. Python MeCabバインディングの再インストール
pip uninstall -y mecab-python3 oseti
pip install --no-cache-dir mecab-python3==1.0.6
pip install --no-cache-dir oseti==0.2.0
            """)
            
            st.markdown("#### **方法3: 環境変数の設定**")
            st.code("""
# 3. 環境変数でMeCab辞書パスを明示
export MECAB_DICDIR=/usr/local/lib/mecab/dic/ipadic  # macOS
export MECAB_DICDIR=/usr/lib/mecab/dic/ipadic        # Linux
            """)
            
            st.markdown("#### **方法4: Docker環境の場合**")
            st.code("""
# Dockerfile に追加
RUN apt-get update && apt-get install -y mecab mecab-ipadic-utf8 libmecab-dev
ENV MECAB_DICDIR /usr/lib/mecab/dic/ipadic
            """)
            
            # 代替案の提示
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 簡易感情分析モードで続行", type="primary"):
                    st.session_state.use_simple_sentiment = True
                    st.rerun()
                st.info("MeCab不要の基本的な感情分析で続行します")
            
            with col2:
                st.info("📎 **推奨**: 上記の解決方法を試すか、LLM版をご利用ください")
                if st.button("🔄 MeCabの設定を再試行"):
                    st.rerun()
            
            st.stop()
            
        except Exception as e:
            st.error(f"初期化中にエラーが発生: {str(e)}")
            st.info("📝 このエラーをコピーして開発者に報告してください。")
            st.stop()
        
    except ImportError:
        st.error("osetiライブラリがインストールされていません。")
        st.code("pip install oseti==0.2.0")
        st.info("requirements.txtにosetiを追加して再起動してください。")
        st.stop()
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {str(e)}")
        st.info("技術的な問題が発生しています。管理者にお問い合わせください。")
        st.stop()

# アナライザーが正常に初期化されているか確認
if analyzer is None:
    st.error("感情分析器の初期化に失敗しました。")
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
    """バッチ感情分析（oseti または 簡易モード対応）"""
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
                # 感情分析器のタイプに応じて処理を分岐
                if isinstance(analyzer, SimpleSentimentAnalyzer):
                    # 簡易感情分析の場合
                    compound_score = analyzer.analyze(processed_text)
                else:
                    # oseti による感情分析の場合
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
                    positive = abs(compound_score)     # 正の値を正規化
                    negative = 0.0
                    neutral  = 1.0 - positive
                elif compound_score < -0.1:
                    positive = 0.0
                    negative = abs(compound_score)     # 負の値を正の値に変換
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
            
            analyzer_type = "簡易モード" if isinstance(analyzer, SimpleSentimentAnalyzer) else "osetiモード"
            status_text.text(f"{analyzer_type} で分析中... {idx+1}/{len(texts)} ({progress*100:.1f}%)")
            
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
    
    analyzer_type = "簡易感情分析" if isinstance(analyzer, SimpleSentimentAnalyzer) else "oseti感情分析"
    status_text.text(f"✅ {analyzer_type}完了！")
    progress_bar.progress(1.0)
    
    return results

# 感情分析実行
analyzer_name = "簡易感情分析 (MeCab不要)" if isinstance(analyzer, SimpleSentimentAnalyzer) else "oseti感情分析"

if st.button(f"感情分析実行 ({analyzer_name})", type="primary"):
    
    if isinstance(analyzer, SimpleSentimentAnalyzer):
        st.info("簡易感情分析による感情分析を開始します（MeCab不要）")
    else:
        st.info("osetiライブラリによる感情分析を開始します")
    
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
    
    st.success(f" {len(results_df)}件の感情分析が完了しました！")
    
    # 各台本データの感情判定一覧表示
    st.subheader(" 各台本データの感情判定一覧")
    
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
    st.subheader("感情別フィルタリング")
    
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
    st.subheader("感情スコア基本統計")
    
    emotion_stats = results_df[["positive", "negative", "neutral", "compound"]].describe()
    st.dataframe(emotion_stats.round(3))
    
    # 相関分析
    st.subheader("感情-収益相関分析")
    
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
    - **感情分析**: 日本語感情分析（osetiまたは簡易モード）
    - **相関分析**: 感情スコアと収益の相関関係を統計的に検証
    - **可視化**: 相関関係、感情分布、ヒートマップを表示
    - **エクスポート**: 分析結果をCSV形式でダウンロード可能
    
    ### 🔧 感情分析モード
    **osetiモード（推奨）:**
    - MeCabによる高精度な形態素解析
    - 日本語専用の感情分析ライブラリ
    - より詳細で精密な感情分析
    
    **簡易モード（MeCab不要）:**
    - システム設定不要で即座に利用可能
    - 基本的な感情語彙辞書による分析
    - MeCab設定問題の回避策として提供
    
    ### ⚙️ 設定のヒント
    - **基本前処理**: 軽微なクリーニングのみ
    - **詳細前処理**: URL、記号、数字を除去してより精密に分析
    - **最大分析件数**: 処理速度を考慮して調整
    
    ### 📊 結果の解釈
    - **positive**: ポジティブ感情の強さ（0-1）
    - **negative**: ネガティブ感情の強さ（0-1）
    - **neutral**: 中性的感情の強さ（0-1）
    - **compound**: 総合感情スコア（-1から1）
    - **統計的有意性**: p値 < 0.05 で相関が統計的に意味あり
    
    ### 🆚 各版の特徴比較
    - **LLM版**: AI による複雑な感情分析、API料金が発生
    - **oseti版**: 高精度、MeCab要、完全無料
    - **簡易版**: 基本精度、設定不要、完全無料
    """)

st.markdown("---")
if isinstance(analyzer, SimpleSentimentAnalyzer):
    st.caption("🚀 簡易感情分析 (MeCab不要) | 📊 台本データ分析ハブ")
else:
    st.caption("💖 oseti ライブラリによる感情分析 | 📊 台本データ分析ハブ")