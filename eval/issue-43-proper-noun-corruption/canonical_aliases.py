"""Data-only canonical aliases and matching helpers for the #43 evaluation.

This module must stay importable without MLX. The clean-control preflight and the
read-only transition analyzer share it so leakage checks and R-stage matching use the
same alias semantics.
"""
import functools
import re


# Mechanical Revised Romanization is insufficient for historical names: it yields
# forms such as "I Sunsin" and "Gim Busik", while real English text commonly uses
# "Yi Sun-sin" and "Kim Bu-sik".
CONVENTIONAL_ALIASES = {
    "유형원": ["Yu Hyeong-won", "Yu Hyongwon", "Ryu Hyongwon"],
    "이익": ["Yi Ik", "Lee Ik", "I Ik"],
    "박지원": ["Park Ji-won", "Pak Chiwon", "Bak Jiwon"],
    "박제가": ["Park Je-ga", "Pak Chega", "Bak Jega"],
    "홍대용": ["Hong Dae-yong", "Hong Taeyong"],
    "정약용": ["Jeong Yak-yong", "Chong Yagyong", "Jung Yakyong", "Dasan"],
    "유득공": ["Yu Deuk-gong", "Yu Tukkong"],
    "이덕무": ["Yi Deok-mu", "Lee Deokmu", "Yi Tongmu"],
    "김정희": ["Kim Jeong-hui", "Kim Jung-hee", "Kim Chonghui", "Chusa"],
    "세종": ["Sejong", "King Sejong", "Sejong the Great"],
    "정인지": ["Jeong In-ji", "Chong Inji"],
    "신숙주": ["Shin Suk-ju", "Sin Sukchu", "Shin Sukju"],
    "성삼문": ["Seong Sam-mun", "Song Sammun", "Sung Sammun"],
    "최항": ["Choe Hang", "Choi Hang"],
    "박팽년": ["Park Paeng-nyeon", "Pak Paengnyon"],
    "최만리": ["Choe Man-ri", "Choi Man-ri", "Choi Mal-li", "Choe Malli"],
    "정창손": ["Jeong Chang-son", "Chong Changson"],
    "이순신": ["Yi Sun-sin", "Yi Sun-shin", "Lee Sun-shin", "Yi Sunshin"],
    "원균": ["Won Gyun", "Weon Gyun", "Won Kyun"],
    "권율": ["Gwon Yul", "Kwon Yul", "Kwon Yool"],
    "이억기": ["Yi Eok-gi", "Lee Eok-gi", "Yi Okki"],
    "정운": ["Jeong Un", "Chong Un", "Jung Woon"],
    "황진": ["Hwang Jin", "Hwang Chin"],
    "김종직": ["Kim Jong-jik", "Kim Chongjik"],
    "조광조": ["Jo Gwang-jo", "Cho Kwang-jo", "Jo Gwangjo"],
    "김일손": ["Kim Il-son", "Kim Ilson"],
    "유자광": ["Yu Ja-gwang", "Yu Chagwang", "Ryu Jagwang"],
    "남곤": ["Nam Gon", "Nam Kon"],
    "심정": ["Sim Jeong", "Shim Jung", "Sim Chong"],
    "연산군": ["Yeonsangun", "Yonsangun", "Yeonsan-gun", "Prince Yeonsan",
            "King Yeonsan"],
    "중종": ["Jungjong", "Chungjong", "King Jungjong"],
    "김부식": ["Kim Bu-sik", "Kim Busik", "Kim Pusik"],
    "일연": ["Ilyeon", "Iryeon", "Il-yeon"],
    "인종": ["Injong", "King Injong"],
    "각훈": ["Gakhun", "Kakhun", "Gak-hun"],
    "김옥균": ["Kim Ok-gyun", "Kim Okkyun"],
    # Observed in the committed gpt-oss production artifact.
    "박영효": ["Park Yeong-hyo", "Pak Yonghyo", "Park Young-hyo", "Park Yung-hyo"],
    "홍영식": ["Hong Yeong-sik", "Hong Yongsik"],
    "서광범": ["Seo Gwang-beom", "So Kwangbom"],
    "서재필": ["Seo Jae-pil", "So Chaepil", "Philip Jaisohn"],
    "민영익": ["Min Yeong-ik", "Min Yongik"],
    "고종": ["Gojong", "Kojong", "King Gojong", "Emperor Gojong"],
}

_I = ['g','kk','n','d','tt','r','m','b','pp','s','ss','','j','jj','ch','k','t','p','h']
_V = ['a','ae','ya','yae','eo','e','yeo','ye','o','wa','wae','oe','yo','u','wo','we','wi','yu','eu','ui','i']
_F = ['','k','k','k','n','n','n','t','l','l','l','l','l','l','l','l','m','p','p','t','t','ng','t','t','k','t','p','t']


def romanize(text):
    out = []
    for char in text:
        code = ord(char) - 0xAC00
        out.append(_I[code // 588] + _V[(code % 588) // 28] + _F[code % 28]
                   if 0 <= code < 11172 else char)
    return "".join(out)


def romanize_spaced(text):
    """Mechanical romanization with syllable boundaries kept as spaces.

    romanize() alone yields one fused token ("sinsukju"), and contains_alias compiles a
    single-part alias to a bare literal — so the leakage guard only ever matched a
    spelling no English text uses. Spacing the syllables lets the same _NAME_SEP logic
    that serves conventional aliases cover "Sin Sukju", "Sin Suk-ju" and "Sinsukju" too.
    """
    return " ".join(romanize(ch) for ch in text if "가" <= ch <= "힣")


def flat_letters(text):
    """Lowercase, ASCII letters only. Not used by contains_alias, which is word-bounded;
    only for controlled comparisons of short fixed strings in diagnostics and preflight.
    """
    return re.sub(r"[^a-z]", "", text.lower())


# Separators that genuinely occur inside a romanized Korean name, and nothing else.
# gpt-oss writes U+202F narrow no-break space and U+2011 non-breaking hyphen, so both
# must pass. Periods, commas, asterisks, pipes and newlines must not: an earlier
# `\W*` join matched "Yi. Ik", "Yi -- Ik", "Nam, Gon" and — in a real artifact —
# "King** | Jungjong", which spans two Markdown table cells.
_NAME_SPACE = "[ \t\u00a0\u2007\u2009\u202f]"
_NAME_HYPHEN = "[-\u2010\u2011\u2012\u2013\u2014\u2015]"
# At most one hyphen, optionally cushioned by spaces; may also be empty for joined
# forms like "Park Jiwon". Two hyphens ("Yi -- Ik") are a dash, not a name separator.
_NAME_SEP = f"(?:{_NAME_SPACE}*{_NAME_HYPHEN}?{_NAME_SPACE}*)"

# A trailing hyphen+letter extends the name into a different person: "Yi Ik-hwan" is
# 이익환, a fabrication this eval exists to catch, not 이익. A possessive is not an
# extension, so "Yi Ik's" still matches.
_NOT_EXTENDED_RIGHT = f"(?!\\w)(?!{_NAME_HYPHEN}\\w)"
# The mirror of the right guard is word-then-hyphen, not hyphen-then-word: for
# "Hwan-Yi Ik" the two characters before the match are "n-", so a `hyphen\w` lookbehind
# never fires and the guard was a no-op.
_NOT_EXTENDED_LEFT = f"(?<!\\w)(?<!\\w{_NAME_HYPHEN})"


@functools.lru_cache(maxsize=None)
def _alias_pattern(alias):
    parts = re.findall(r"[A-Za-z]+", alias)
    if not parts:
        return None
    body = _NAME_SEP.join(map(re.escape, parts))
    return re.compile(_NOT_EXTENDED_LEFT + body + _NOT_EXTENDED_RIGHT, re.I)


def contains_alias(text, alias):
    pattern = _alias_pattern(alias)
    return bool(pattern and pattern.search(text))


def names_person(text, person):
    if re.search(rf"(?<![가-힣]){re.escape(person)}(?![가-힣])", text):
        return True
    return any(contains_alias(text, alias)
               for alias in CONVENTIONAL_ALIASES.get(person, ()))
