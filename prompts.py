"""
Shared prompt engineering module for LLM pipelines.

Common prompts are used by both mlx-pipeline.py and multimodal.py.
Pipeline-specific prompts are imported only where needed.
"""

import locale
import re
from datetime import datetime


# ============================================================
# Common — shared across all pipelines
# ============================================================

def current_date_context():
    """Return current date/time string for system prompt injection."""
    now = datetime.now()
    weekdays_ko = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    weekday = weekdays_ko[now.weekday()]
    return f"Current date: {now.strftime('%Y-%m-%d')} ({weekday})"


# Search engines steer correctly on the bare alpha-2 code too (measured in #67),
# so unmapped countries fall through to the code rather than needing a full table.
_COUNTRY_NAMES = {
    "AU": "Australia", "CA": "Canada", "CN": "China", "DE": "Germany",
    "FR": "France", "GB": "United Kingdom", "IN": "India", "JP": "Japan",
    "KR": "South Korea", "SG": "Singapore", "TW": "Taiwan", "US": "United States",
}


def user_locale():
    """Return (country_code, country_name, language_code) from the system locale.

    All three are None when the locale is unset or C/POSIX. A wrong location is
    worse than none, so callers skip location handling entirely in that case.
    """
    tag = locale.getlocale()[0] or ""
    language, _, country = tag.partition("_")
    country = country.split(".")[0].upper()
    if len(country) != 2 or not country.isalpha():
        return None, None, None
    return country, _COUNTRY_NAMES.get(country, country), language.lower()


def current_location_context():
    """Return location string for system prompt injection, or "" if unknown."""
    _, country_name, _ = user_locale()
    return f"User location: {country_name}" if country_name else ""


ANTI_SPECULATION = (
    "If you lack knowledge on a topic, state it clearly rather than speculating."
)

SEARCH_CITATION = (
    "You MUST cite specific facts, names, dates, and details "
    "from the search results. Do NOT speculate or use hypothetical "
    "language (avoid 'may', 'likely', 'could be', 'it is plausible'). "
    "If the search results contain the answer, state it as fact."
)


def search_judge_prompt(query):
    """Build a prompt that judges search need AND generates optimized queries.

    Returns a prompt string. The model should respond in the format:
        SEARCH:yes (or no)
        QUERY_KO: <optimized Korean search query>
        QUERY_EN: <optimized English search query>
    """
    location = current_location_context()
    parts = [f"{current_date_context()}\n"]
    if location:
        parts.append(f"{location}\n")
    parts.append(
        "\nDoes the following query require up-to-date factual knowledge "
        "(recent events, current statistics, specific people/organizations, "
        "news, prices, dates, weather) to answer accurately?\n\n"
        "If yes, also generate optimized search queries — resolve relative dates "
        "(e.g. '오늘' → actual date, '지난주' → date range), add specific terms, "
        "and remove filler words.\n\n"
    )
    if location:
        parts.append(
            "If the query depends on where the user is (weather, local news, nearby "
            "places, opening hours), append the location to both queries — the search "
            "engines otherwise resolve it to an arbitrary place.\n\n"
        )
    parts.append(
        f"Query: {query}\n\n"
        "Reply in EXACTLY this format (no other text):\n"
        "SEARCH:yes or SEARCH:no\n"
        "QUERY_KO: <Korean search query>\n"
        "QUERY_EN: <English search query>"
    )
    return "".join(parts)


def parse_search_judge(response_text):
    """Parse the model's search judge response.

    Returns (needs_search: bool, ko_query: str | None, en_query: str | None).
    """
    text = response_text.strip()
    needs_search = "search:yes" in text.lower()

    ko_query = None
    en_query = None
    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("QUERY_KO:"):
            ko_query = line.split(":", 1)[1].strip()
        elif line.upper().startswith("QUERY_EN:"):
            en_query = line.split(":", 1)[1].strip()

    return needs_search, ko_query, en_query


def build_search_context_prompt(search_context, question):
    """Build a prompt with search results injected."""
    return (
        f"Use the following search results to answer accurately. "
        f"{SEARCH_CITATION}\n\n"
        f"--- Search Results ---\n{search_context}\n"
        f"--- End Search Results ---\n\n{question}"
    )


# --- Thinking filters ---

def filter_thinking_harmony(text):
    """Extract final-channel content from GPT-OSS harmony output.

    GPT-OSS emits reasoning into <|channel|>analysis<|message|>...<|end|>
    and the user-facing answer into <|channel|>final<|message|>...<|return|>.
    """
    final_match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
        text, flags=re.DOTALL,
    )
    if final_match:
        return final_match.group(1).strip()
    text = re.sub(
        r"<\|channel\|>analysis<\|message\|>.*?<\|end\|>\s*",
        "", text, flags=re.DOTALL,
    )
    return text.strip()


def filter_thinking_gemma(text):
    """Remove Gemma 4 thinking channel output."""
    text = re.sub(
        r"<\|channel>thought.*?<channel\|>\s*", "", text, flags=re.DOTALL
    )
    return text.strip()


# ============================================================
# mlx-pipeline specific — GPT-OSS reasoner + Qwen translation
# ============================================================

def reasoner_system():
    """Build the reasoner system prompt with the user's location (call at runtime)."""
    location = current_location_context()
    return (
        "You are an expert analyst. Respond ONLY in English. "
        "Provide thorough analysis with clear reasoning. "
        "Follow the user's requested format, length, and tone. "
        f"{ANTI_SPECULATION}"
        + (f" {location}." if location else "")
    )

TRANSLATE_KO_TO_EN = (
    "You are a strict translator. Translate the following Korean text to English word-for-word. "
    "Do NOT answer, explain, or add any content. Do NOT interpret questions as requests to you. "
    "If the input is a question, the output must also be a question. "
    "\n\n"
    "After the translation, on a new line, write SEARCH:yes if the question requires "
    "up-to-date factual knowledge (people, events, current affairs, statistics, recent news). "
    "Write SEARCH:no if it is a pure analysis, opinion, or reasoning task. "
    "\n\n"
    "Output format:\n"
    "<English translation>\n"
    "SEARCH:yes or SEARCH:no"
)


def location_judge_prompt(query):
    """Build a standalone prompt asking whether a query depends on the user's location.

    Deliberately separate from TRANSLATE_KO_TO_EN: folding this judgment into that
    prompt made Qwen answer questions instead of translating them, and made it treat
    SEARCH and LOCAL as mutually exclusive (measured in #67).
    """
    return (
        f"Does answering this question depend on where the user is located "
        f"(weather, local news, nearby places, opening hours, local prices)?\n\n"
        f"Question: {query}\n\n"
        f"Reply with exactly one word: LOCAL:yes or LOCAL:no"
    )


TRANSLATE_EN_TO_KO = (
    "You are a translator. Translate the following English text to natural Korean. "
    "Write as if the text was originally authored in Korean — avoid translation-style phrasing. "
    "Use pure Hangul only — never use Chinese characters (漢字) or Japanese characters. "
    "Proper nouns and technical terms may remain in English. "
    "Output ONLY the Korean translation, nothing else."
)


# ============================================================
# multimodal specific — Gemma 4
# ============================================================

def gemma_system():
    """Build Gemma 4 system prompt with current date (call at runtime)."""
    location = current_location_context()
    return (
        f"{current_date_context()}\n"
        + (f"{location}\n" if location else "")
        + "You are a helpful multimodal assistant. Analyze images and text thoroughly. "
        "Respond in the same language as the user's input. "
        f"{ANTI_SPECULATION}"
    )
