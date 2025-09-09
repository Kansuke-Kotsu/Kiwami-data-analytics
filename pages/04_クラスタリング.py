import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from lib.text_utils import simple_japanese_tokenize

st.set_page_config(page_title="④ クラスタリング", page_icon="🎯", layout="wide")
st.title("🎯 クラスタリングによるデータ分析")

if "df" not in st.session_state:
    st.warning("まずトップページでExcelをアップロードしてください。")
    st.stop()

df = st.session_state["df"].copy()
meta = st.session_state.get("meta", {})

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")
    
    # 1. データ準備
    st.subheader("1. データ準備")
    
    # DataFrameプレビュー
    with st.expander("📊 DataFrameプレビュー（先頭10行）", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # 欠損値レポート
    with st.expander("❓ 欠損値レポート", expanded=False):
        missing_report = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_report.append({
                "列名": col,
                "欠損数": missing_count,
                "欠損率(%)": f"{missing_pct:.1f}%"
            })
        st.dataframe(pd.DataFrame(missing_report), hide_index=True)
    
    # 2. 特徴量設定
    st.subheader("2. 特徴量設定")
    
    # 数値列の選択
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    selected_numeric_cols = st.multiselect(
        "数値列を選択", 
        options=numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )
    
    # 標準化スイッチ
    standardize_numeric = st.checkbox("数値列を標準化", value=True)
    
    # テキスト列の選択
    text_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    text_col = st.selectbox(
        "テキスト列（1つ）", 
        options=["なし"] + text_cols,
        index=1 if len(text_cols) > 0 else 0
    )
    
    # TF-IDF設定
    if text_col != "なし":
        use_tfidf = st.checkbox("TF-IDF埋め込みを使用", value=True)
        
        if use_tfidf:
            vector_dim = st.slider("ベクトル次元", 100, 2000, 300, step=50)
            min_df = st.number_input("min_df（最小出現頻度）", min_value=1, max_value=100, value=5)
            max_df = st.number_input("max_df（最大出現頻度）", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
            
            # 日本語は指定なし、英語は "english"
            use_stopwords = st.selectbox("ストップワード", ["なし", "english"])
            stopwords = "english" if use_stopwords == "english" else None
            
            # 自動派生特徴
            st.write("**自動派生特徴（テキスト列対象）**")
            derive_char_count = st.checkbox("文字数", value=True)
            derive_word_count = st.checkbox("単語数", value=True) 
            derive_avg_word_len = st.checkbox("平均文長", value=True)
    else:
        use_tfidf = False
    
    # 3. クラスタリング設定
    st.subheader("3. クラスタリング設定")
    
    # クラスタ数設定
    cluster_method = st.radio("クラスタ数の決定方法", ["直接指定", "自動推定（シルエット最大）"])
    
    if cluster_method == "直接指定":
        n_clusters = st.slider("クラスタ数 k", 2, 15, 5)
    else:
        k_range_start = st.slider("探索範囲（最小）", 2, 10, 2)
        k_range_end = st.slider("探索範囲（最大）", k_range_start + 1, 15, 10)
    
    # スケーリング
    use_scaling = st.checkbox("StandardScaler適用", value=True)
    
    # 4. 次元削減設定
    st.subheader("4. 次元削減")
    
    reduction_method = st.selectbox("手法", ["t-SNE", "UMAP"])
    
    if reduction_method == "t-SNE":
        perplexity = st.slider("perplexity", 5, 50, 30)
        learning_rate = st.number_input("learning_rate", 10.0, 1000.0, 200.0, step=50.0)
        n_iter = st.slider("n_iter", 250, 2000, 1000, step=250)
    else:  # UMAP
        n_neighbors = st.slider("n_neighbors", 5, 50, 15)
        min_dist = st.number_input("min_dist", 0.0, 1.0, 0.1, step=0.05)
    
    # 5. 実行ボタン
    st.subheader("5. 実行")
    execute_btn = st.button("🎯 クラスタリング実行", type="primary", use_container_width=True)

# メインエリア
if execute_btn:
    # データ前処理と特徴量作成
    st.header("📊 クラスタリング結果")
    
    with st.status("特徴量を準備中...", expanded=True) as status:
        st.write("数値特徴量を処理中...")
        
        # 数値特徴量の準備
        features_list = []
        feature_names = []
        
        if selected_numeric_cols:
            numeric_data = df[selected_numeric_cols].fillna(0)
            
            if standardize_numeric:
                scaler = StandardScaler()
                numeric_data = pd.DataFrame(
                    scaler.fit_transform(numeric_data),
                    columns=selected_numeric_cols,
                    index=df.index
                )
            
            features_list.append(numeric_data)
            feature_names.extend(selected_numeric_cols)
        
        # テキスト特徴量の準備
        if text_col != "なし" and use_tfidf:
            st.write("テキスト特徴量（TF-IDF）を処理中...")
            
            # テキストデータの前処理
            text_data = df[text_col].fillna("").astype(str)
            
            # TF-IDF
            def japanese_tokenizer(text):
                return simple_japanese_tokenize(text)
            
            vectorizer = TfidfVectorizer(
                tokenizer=japanese_tokenizer,
                max_features=vector_dim,
                min_df=min_df,
                max_df=max_df,
                stop_words=stopwords
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(text_data)
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
                    index=df.index
                )
                features_list.append(tfidf_df)
                feature_names.extend(tfidf_df.columns.tolist())
                
                # 語彙を保存（後でクラスタプロファイル用）
                vocab = vectorizer.get_feature_names_out()
                
            except Exception as e:
                st.error(f"TF-IDF処理でエラーが発生しました: {str(e)}")
                st.stop()
            
            # 自動派生特徴
            if derive_char_count or derive_word_count or derive_avg_word_len:
                st.write("派生特徴量を処理中...")
                derived_features = pd.DataFrame(index=df.index)
                
                if derive_char_count:
                    derived_features["char_count"] = text_data.str.len()
                    feature_names.append("char_count")
                
                if derive_word_count:
                    derived_features["word_count"] = text_data.apply(lambda x: len(simple_japanese_tokenize(x)))
                    feature_names.append("word_count")
                
                if derive_avg_word_len:
                    derived_features["avg_word_len"] = text_data.apply(
                        lambda x: np.mean([len(word) for word in simple_japanese_tokenize(x)]) if simple_japanese_tokenize(x) else 0
                    )
                    feature_names.append("avg_word_len")
                
                if standardize_numeric:
                    scaler_derived = StandardScaler()
                    derived_features = pd.DataFrame(
                        scaler_derived.fit_transform(derived_features),
                        columns=derived_features.columns,
                        index=df.index
                    )
                
                features_list.append(derived_features)
        
        # 全特徴量を結合
        if features_list:
            X = pd.concat(features_list, axis=1)
            st.write(f"特徴量作成完了: {X.shape[1]}次元 × {X.shape[0]}サンプル")
        else:
            st.error("特徴量が選択されていません。")
            st.stop()
        
        status.update(label="特徴量準備完了!", state="complete")
    
    # クラスタ数の決定
    with st.status("最適クラスタ数を探索中...", expanded=True) as status:
        if cluster_method == "自動推定（シルエット最大）":
            st.write("シルエット分析を実行中...")
            
            # 探索範囲でシルエットスコアを計算
            silhouette_scores = []
            k_range = range(k_range_start, k_range_end + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(X)
                sil_score = silhouette_score(X, cluster_labels)
                silhouette_scores.append(sil_score)
                st.write(f"k={k}: シルエットスコア = {sil_score:.3f}")
            
            # 最適なkを選択
            best_idx = np.argmax(silhouette_scores)
            n_clusters = k_range[best_idx]
            best_silhouette = silhouette_scores[best_idx]
            
            st.write(f"**最適クラスタ数: {n_clusters} (シルエットスコア: {best_silhouette:.3f})**")
        
        status.update(label="クラスタ数決定完了!", state="complete")
    
    # クラスタリング実行
    with st.status("K-meansクラスタリング実行中...", expanded=True) as status:
        # 最終的なスケーリング
        if use_scaling:
            scaler_final = StandardScaler()
            X_scaled = scaler_final.fit_transform(X)
        else:
            X_scaled = X.values
        
        # K-means実行
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # シルエットスコア計算
        final_silhouette = silhouette_score(X_scaled, cluster_labels)
        st.write(f"最終シルエットスコア: {final_silhouette:.3f}")
        
        status.update(label="クラスタリング完了!", state="complete")
    
    # 次元削減実行
    with st.status("次元削減実行中...", expanded=True) as status:
        if reduction_method == "t-SNE":
            st.write("t-SNE実行中...")
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=42
            )
        else:
            st.write("UMAP実行中...")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42
            )
        
        X_reduced = reducer.fit_transform(X_scaled)
        
        status.update(label="次元削減完了!", state="complete")
    
    # 結果の可視化
    st.subheader("📈 クラスタ評価")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # エルボー法
        st.write("**エルボー法（SSE vs k）**")
        k_range_elbow = range(2, 16)
        sse_scores = []
        
        for k in k_range_elbow:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            kmeans_temp.fit(X_scaled)
            sse_scores.append(kmeans_temp.inertia_)
        
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range_elbow),
            y=sse_scores,
            mode='lines+markers',
            name='SSE'
        ))
        fig_elbow.update_layout(
            title="エルボー法",
            xaxis_title="クラスタ数 k",
            yaxis_title="SSE (Sum of Squared Errors)",
            height=300
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col2:
        # シルエットスコア
        st.write("**シルエットスコア（k=2-15）**")
        k_range_sil = range(2, 16)
        sil_scores_all = []
        
        for k in k_range_sil:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            labels_temp = kmeans_temp.fit_predict(X_scaled)
            sil_score = silhouette_score(X_scaled, labels_temp)
            sil_scores_all.append(sil_score)
        
        colors = ['red' if k == n_clusters else 'blue' for k in k_range_sil]
        
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=list(k_range_sil),
            y=sil_scores_all,
            marker_color=colors,
            name='シルエットスコア'
        ))
        fig_sil.update_layout(
            title="シルエットスコア比較",
            xaxis_title="クラスタ数 k", 
            yaxis_title="シルエットスコア",
            height=300
        )
        st.plotly_chart(fig_sil, use_container_width=True)
    
    # 最終スコア表示
    st.metric("選択されたk", n_clusters, f"シルエットスコア: {final_silhouette:.3f}")
    
    # 2次元散布図
    st.subheader("🎨 2次元可視化")
    
    # データフレーム作成（散布図用）
    viz_df = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1], 
        'cluster': [f"Cluster {i}" for i in cluster_labels],
        'index': df.index
    })
    
    # 追加情報をツールチップ用に準備
    tooltip_cols = ['index']
    
    # タイトル列があるかチェック
    title_candidates = ['title', 'タイトル', '台本データ', '台本', '本文']
    title_col = None
    for col in title_candidates:
        if col in df.columns:
            title_col = col
            break
    
    if title_col:
        viz_df['title'] = df[title_col].fillna("").astype(str).str[:50]  # 50文字まで表示
        tooltip_cols.append('title')
    
    # revenue列があるかチェック
    revenue_candidates = ['revenue', 'profit', '収益', '売上', '利益', '合算粗利']
    revenue_col = None
    for col in revenue_candidates:
        if col in df.columns:
            revenue_col = col
            break
    
    if revenue_col:
        viz_df['revenue'] = df[revenue_col].fillna(0)
        tooltip_cols.append('revenue')
    
    # 主要特徴の一部を追加
    if selected_numeric_cols:
        for col in selected_numeric_cols[:3]:  # 上位3つまで
            viz_df[f'{col}_feature'] = df[col].fillna(0)
            tooltip_cols.append(f'{col}_feature')
    
    viz_df['cluster_id'] = cluster_labels
    tooltip_cols.append('cluster_id')
    
    # Plotly散布図作成
    fig_scatter = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='cluster',
        hover_data=tooltip_cols,
        title=f"{reduction_method} 2次元可視化 (k={n_clusters})",
        labels={'x': f'{reduction_method}_1', 'y': f'{reduction_method}_2'}
    )
    
    fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
    fig_scatter.update_layout(height=600)
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # クラスタプロファイル
    st.subheader("📋 クラスタプロファイル")
    
    # クラスタ別統計
    cluster_stats = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_data = df[cluster_mask]
        
        stats = {
            'クラスタ': f'Cluster {i}',
            '件数': cluster_mask.sum()
        }
        
        # 収益統計
        if revenue_col:
            revenue_data = pd.to_numeric(cluster_data[revenue_col], errors='coerce')
            if not revenue_data.dropna().empty:
                stats['収益_平均'] = f"{revenue_data.mean():.2f}"
                stats['収益_中央値'] = f"{revenue_data.median():.2f}"
                stats['収益_25%'] = f"{revenue_data.quantile(0.25):.2f}"
                stats['収益_75%'] = f"{revenue_data.quantile(0.75):.2f}"
        
        cluster_stats.append(stats)
    
    cluster_df = pd.DataFrame(cluster_stats)
    st.dataframe(cluster_df, hide_index=True, use_container_width=True)
    
    # 数値特徴の平均比較
    if selected_numeric_cols:
        st.write("**数値特徴の平均値比較**")
        
        feature_means = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_means = df[cluster_mask][selected_numeric_cols].mean()
            cluster_means['cluster'] = f'Cluster {i}'
            feature_means.append(cluster_means)
        
        means_df = pd.DataFrame(feature_means).set_index('cluster')
        
        # レーダーチャート作成
        fig_radar = go.Figure()
        
        for cluster in means_df.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=means_df.loc[cluster].values,
                theta=means_df.columns,
                fill='toself',
                name=cluster
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[means_df.values.min(), means_df.values.max()]
                )
            ),
            showlegend=True,
            title="クラスタ別数値特徴の平均値レーダー"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # テキスト特徴のトップ語彙（TF-IDF使用時）
    if text_col != "なし" and use_tfidf and 'vocab' in locals():
        st.write("**クラスタ別トップ語彙（上位20語）**")
        
        # クラスタごとのTF-IDF特徴量を集計
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_tfidf = tfidf_df[cluster_mask]
            
            if len(cluster_tfidf) > 0:
                # クラスタ内でのTF-IDF平均値を計算
                avg_tfidf = cluster_tfidf.mean()
                top_indices = avg_tfidf.nlargest(20).index
                top_words = []
                top_scores = []
                
                for idx in top_indices:
                    tfidf_idx = int(idx.split('_')[1])  # tfidf_0 -> 0
                    if tfidf_idx < len(vocab):
                        top_words.append(vocab[tfidf_idx])
                        top_scores.append(avg_tfidf[idx])
                
                if top_words:
                    with st.expander(f"Cluster {i} トップ語彙", expanded=False):
                        word_df = pd.DataFrame({
                            '語彙': top_words,
                            'TF-IDF平均': top_scores
                        })
                        
                        # 棒グラフ
                        fig_words = px.bar(
                            word_df.head(10),
                            x='TF-IDF平均',
                            y='語彙',
                            orientation='h',
                            title=f'Cluster {i} トップ10語彙'
                        )
                        fig_words.update_layout(height=400)
                        st.plotly_chart(fig_words, use_container_width=True)
                        
                        # テーブル
                        st.dataframe(word_df, hide_index=True)

    st.success("✅ クラスタリング分析が完了しました！")
    
else:
    st.info("👈 左側のサイドバーで設定を行い、「クラスタリング実行」ボタンをクリックしてください。")