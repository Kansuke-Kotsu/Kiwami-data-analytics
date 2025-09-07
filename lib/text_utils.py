
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
