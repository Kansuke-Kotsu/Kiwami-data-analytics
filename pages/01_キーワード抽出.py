
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import io

from lib.text_utils import is_noise_token

st.set_page_config(page_title="① キーワード抽出", page_icon="🔑", layout="wide")
st.title("🔑 キーワード抽出（収益寄与の可視化）")

if "df" not in st.session_state:
    st.warning("まずトップページでExcelをアップロードしてください。")
    st.stop()

df = st.session_state["df"].copy()
meta = st.session_state.get("meta", {})

# 列選択
text_col = st.selectbox("台本テキスト列", options=list(df.columns), index=list(df.columns).index(meta.get("text_col")) if meta.get("text_col") in df.columns else 0)
target_col = st.selectbox("収益（目的変数）列", options=[c for c in df.columns if c != text_col], index=list(df.columns).index(meta.get("profit_col")) if meta.get("profit_col") in df.columns else 0)

# ハイパラ
with st.sidebar:
    st.header("抽出パラメータ")
    ngram_min = st.slider("n-gram最小", 2, 4, 2)
    ngram_max = st.slider("n-gram最大", ngram_min, 6, 4)
    min_df = st.slider("min_df（最低出現数）", 3, 50, 10)
    max_features = st.slider("max_features", 1000, 20000, 5000, step=500)
    svd_components = st.slider("SVD成分数（圧縮次元）", 20, 300, 80, step=10)
    top_k = st.slider("上位キーワード数", 5, 100, 30, step=5)
    sample_n = st.slider("サンプル件数（速度優先）", 500, len(df), min(1500, len(df)), step=100)

# 前処理
df["_text"] = df[text_col].fillna("").astype(str)
y_raw = pd.to_numeric(df[target_col], errors="coerce")
use = df[y_raw.notna()].copy()
use["y"] = pd.to_numeric(use[target_col], errors="coerce")
if len(use) < 50:
    st.error("目的変数（収益）に有効な数値が50件未満です。列選択を見直してください。")
    st.stop()

if len(use) > sample_n:
    use = use.sample(sample_n, random_state=42)

# TF-IDF（char n-gram）
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(ngram_min, ngram_max),
    min_df=min_df,
    max_features=max_features
)
X_tfidf = vectorizer.fit_transform(use["_text"].values)
vocab = np.array(vectorizer.get_feature_names_out())

# ノイズn-gramのマスクを作成（後で重要度集計時に除外）
noise_mask = np.array([is_noise_token(tok) for tok in vocab])

# 次元圧縮＋回帰
svd = TruncatedSVD(n_components=svd_components, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

X_train, X_test, y_train, y_test = train_test_split(X_svd, use["y"].values.astype(float), test_size=0.2, random_state=42)
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("モデル指標")
st.write({"R2": round(float(r2), 4), "MAE": round(float(mae), 2), "n_train": int(len(X_train)), "n_test": int(len(X_test))})

# 係数を原語彙空間へ近似逆写像して各n-gramの重要度を推定
coef_svd = model.coef_
V = svd.components_  # (n_components, n_features)
approx_importance = V.T @ coef_svd  # (n_features,)

# ノイズ除外
approx_importance = np.where(noise_mask, 0.0, approx_importance)

order_pos = np.argsort(approx_importance)[::-1]
order_neg = np.argsort(approx_importance)

top_pos = pd.DataFrame({"ngram": vocab[order_pos[:top_k]], "importance": approx_importance[order_pos[:top_k]]})
top_neg = pd.DataFrame({"ngram": vocab[order_neg[:top_k]], "importance": approx_importance[order_neg[:top_k]]})

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 収益に **正** の影響が大きい n-gram")
    fig1 = plt.figure(figsize=(7, 6))
    plt.barh(top_pos["ngram"][::-1], top_pos["importance"][::-1])
    plt.title("正の寄与（上位）")
    plt.tight_layout()
    st.pyplot(fig1)
    st.dataframe(top_pos)

with col2:
    st.markdown("#### 収益に **負** の影響が大きい n-gram")
    fig2 = plt.figure(figsize=(7, 6))
    plt.barh(top_neg["ngram"][::-1], top_neg["importance"][::-1])
    plt.title("負の寄与（上位）")
    plt.tight_layout()
    st.pyplot(fig2)
    st.dataframe(top_neg)

# ダウンロード用
out = pd.DataFrame({
    "ngram": vocab,
    "importance": approx_importance,
    "is_noise": noise_mask
}).sort_values("importance", ascending=False)

st.download_button(
    label="全n-gram重要度をCSVダウンロード",
    data=out.to_csv(index=False).encode("utf-8-sig"),
    file_name="ngram_importance.csv",
    mime="text/csv",
)

st.caption("注：日本語は形態素解析なしでもchar n-gramで安定して傾向が取れます。英数字のみの羅列やURLは自動で除外しています。")
