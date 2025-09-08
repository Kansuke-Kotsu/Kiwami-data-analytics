
import re

ASCII_RE = re.compile(r"^[\x00-\x7F]+$")
URL_LIKE_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
# 記号列（日本語が含まれない短い断片など）を弾くための緩い判定
JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")  # ひらがな・カタカナ・漢字

def is_noise_token(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return True
    # URLやASCIIだけの羅列（英数字のみ・記号のみ）は除外
    if URL_LIKE_RE.search(t):
        return True
    if ASCII_RE.match(t):
        # 純ASCIIは基本ノイズ扱い（CTA英語等を許容したい場合は閾値を下げる）
        return True
    # 日本語が1文字もなければ除外
    if not JAPANESE_CHAR_RE.search(t):
        return True
    # 記号だけ・1文字すぎる断片
    if len(t) <= 1:
        return True
    return False

def simple_japanese_tokenize(text: str) -> list:
    """
    簡易的な日本語トークン化
    文字種の変化点で区切りを入れて意味のある単語単位に近づける
    """
    if not text.strip():
        return []
    
    # 文字種パターンを定義
    hiragana_re = re.compile(r'[\u3040-\u309f]+')
    katakana_re = re.compile(r'[\u30a0-\u30ff]+')  
    kanji_re = re.compile(r'[\u3400-\u9fff]+')
    ascii_re = re.compile(r'[a-zA-Z0-9]+')
    punctuation_re = re.compile(r'[。、！？\s\n\r]+')
    
    # まず句読点で大きく分割
    sentences = punctuation_re.split(text)
    
    tokens = []
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        # 文字種の変化点で分割
        current_token = ""
        current_type = None
        
        for char in sentence:
            char_type = None
            if hiragana_re.match(char):
                char_type = "hiragana"
            elif katakana_re.match(char):
                char_type = "katakana"
            elif kanji_re.match(char):
                char_type = "kanji"
            elif ascii_re.match(char):
                char_type = "ascii"
            else:
                char_type = "other"
            
            if current_type is None:
                current_type = char_type
                current_token = char
            elif current_type == char_type:
                current_token += char
            else:
                # 文字種が変わったので前のトークンを保存
                if current_token.strip() and len(current_token) >= 2:
                    tokens.append(current_token.strip())
                current_token = char
                current_type = char_type
        
        # 最後のトークンを追加
        if current_token.strip() and len(current_token) >= 2:
            tokens.append(current_token.strip())
    
    # さらにノイズフィルタリング
    filtered_tokens = []
    for token in tokens:
        if not is_noise_token(token) and len(token) >= 2:
            filtered_tokens.append(token)
    
    return filtered_tokens
