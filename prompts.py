"""
Shared prompt engineering module for LLM pipelines.

Common prompts are used by both mlx-pipeline.py and multimodal.py.
Pipeline-specific prompts are imported only where needed.
"""

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
    return (
        f"{current_date_context()}\n\n"
        f"Does the following query require up-to-date factual knowledge "
        f"(recent events, current statistics, specific people/organizations, "
        f"news, prices, dates, weather) to answer accurately?\n\n"
        f"If yes, also generate optimized search queries — resolve relative dates "
        f"(e.g. '오늘' → actual date, '지난주' → date range), add specific terms, "
        f"and remove filler words.\n\n"
        f"Query: {query}\n\n"
        f"Reply in EXACTLY this format (no other text):\n"
        f"SEARCH:yes or SEARCH:no\n"
        f"QUERY_KO: <Korean search query>\n"
        f"QUERY_EN: <English search query>"
    )


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

REASONER_SYSTEM = (
    "You are an expert analyst. Respond ONLY in English. "
    "Provide thorough analysis with clear reasoning. "
    "Follow the user's requested format, length, and tone. "
    f"{ANTI_SPECULATION}"
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
    return (
        f"{current_date_context()}\n"
        "You are a helpful multimodal assistant. Analyze images and text thoroughly. "
        "Respond in the same language as the user's input. "
        f"{ANTI_SPECULATION}"
    )
