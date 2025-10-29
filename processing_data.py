# preprocessing_improved.py (chỉ phần text cleaning + helpers)
import re
import unicodedata
import numpy as np

# compile regex at module load
RE_EMOJI = re.compile(
    r'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])'
)
RE_SPECIAL = re.compile(r"�+")
RE_PUNCT = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]+")  # keep letters, numbers, spaces
RE_NUMBER = re.compile(r"\b\d+\b")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\S+@\S+")
RE_HTML = re.compile(r"<.*?>")
RE_MULTI_SPACE = re.compile(r"\s+")

# normalize mapping patterns (compiled)
RE_NV = re.compile(r"\bnv\b", flags=re.IGNORECASE)
RE_KHACH_SAN = re.compile(r"\b(khách ?sạn|khach ?san|ksan|ks)\b", flags=re.IGNORECASE)

def strip_emoji(text: str) -> str:
    return RE_EMOJI.sub("", text)

def remove_special_char(text: str) -> str:
    return RE_SPECIAL.sub("", text)

def remove_punctuation(text: str) -> str:
    return RE_PUNCT.sub(" ", text)

def remove_number(text: str) -> str:
    return RE_NUMBER.sub(" ", text)

def normalize_annotation(text: str) -> str:
    text = RE_NV.sub("nhân viên", text)
    text = RE_KHACH_SAN.sub("khách sạn", text)
    return text

def clean_text_single(text: str) -> str:
    # null-safe
    if text is None:
        return ""
    text = str(text)
    # unicode normalize
    text = unicodedata.normalize("NFC", text)
    # lower
    text = text.lower()
    # remove urls/emails/html
    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)
    text = RE_HTML.sub(" ", text)
    # strip/replace
    text = strip_emoji(text)
    text = remove_special_char(text)
    text = remove_punctuation(text)
    text = remove_number(text)
    text = normalize_annotation(text)
    text = RE_MULTI_SPACE.sub(" ", text).strip()
    return text

# wrapper expecting dataset example dict
def clean_text(example):
    return {"Review": clean_text_single(example.get("Review", ""))}
