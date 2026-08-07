"""Controlled experiments isolating which stage loses Korean proper nouns (#43).

Observational runs show the wrapper path recalls far fewer canonical entities than the
direct path, and that the effect is reasoner-independent (two reasoners score alike).
They do NOT show which of three steps causes it. These three experiments separate them.

  A. roundtrip   — names only, KO -> EN -> KO. Isolates the translator pair with no
                   reasoner involved. Survival here means translation is not the locus.
  B. back-only   — correct English names fed straight to EN -> KO. Isolates
                   back-translation from any upstream degradation.
  C. reasoner    — one fixed English question to both reasoners. Isolates whether the
                   reasoner invents Korean names unprompted.

Usage:
    python isolate.py a
    python isolate.py b
    python isolate.py c
"""
import json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mlx_lm import load, generate
import prompts

TRANSLATOR = "mlx-community/Qwen3-14B-4bit"
REASONERS = ["mlx-community/gpt-oss-120b-4bit", "mlx-community/Qwen3.6-35B-A3B-4bit"]

# Ground truth for A and B. Romanization follows Revised Romanization.
NAMES = [
    ("유형원", "Yu Hyeong-won"), ("이익", "Yi Ik"), ("박지원", "Bak Ji-won"),
    ("박제가", "Bak Je-ga"), ("홍대용", "Hong Dae-yong"), ("정약용", "Jeong Yak-yong"),
    ("유득공", "Yu Deuk-gong"), ("이덕무", "Yi Deok-mu"), ("신숙주", "Sin Suk-ju"),
    ("성삼문", "Seong Sam-mun"), ("최만리", "Choe Man-ri"), ("정인지", "Jeong In-ji"),
    ("이순신", "Yi Sun-sin"), ("원균", "Won Gyun"), ("권율", "Gwon Yul"),
    ("김종직", "Kim Jong-jik"), ("조광조", "Jo Gwang-jo"), ("김부식", "Kim Bu-sik"),
    ("김옥균", "Kim Ok-gyun"), ("박영효", "Bak Yeong-hyo"),
]

REASONER_Q = ("List the major scholars of the Korean Silhak movement of the late Joseon "
              "dynasty, with their principal works. Give names as they are conventionally "
              "romanized.")


def call(model, tok, system, user, max_tokens=1500, thinking=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    kw = {} if thinking is None else {"enable_thinking": thinking}
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
    t0 = time.time()
    raw = generate(model, tok, p, max_tokens=max_tokens, verbose=False)
    import re
    text = re.split(r"</think>|<\|channel\|>final<\|message\|>|assistantfinal", raw)[-1]
    return text.strip(), round(time.time() - t0, 2)


def exp_a():
    """Names only, KO -> EN -> KO. No reasoner in the loop."""
    m, t = load(TRANSLATOR)
    ko_list = "\n".join(n for n, _ in NAMES)
    en, _ = call(m, t, prompts.TRANSLATE_KO_TO_EN, ko_list, thinking=False)
    import re
    en = re.sub(r"SEARCH:\s*(yes|no)", "", en).strip()
    back, _ = call(m, t, prompts.TRANSLATE_EN_TO_KO, en, thinking=False)
    survived = [ko for ko, _ in NAMES if ko in back]
    return dict(experiment="A roundtrip", input=ko_list, intermediate_en=en, output=back,
                survived=survived, survival=f"{len(survived)}/{len(NAMES)}")


def exp_b():
    """Correct English names -> KO only. Back-translation in isolation."""
    m, t = load(TRANSLATOR)
    en_list = "\n".join(en for _, en in NAMES)
    back, _ = call(m, t, prompts.TRANSLATE_EN_TO_KO, en_list, thinking=False)
    survived = [ko for ko, _ in NAMES if ko in back]
    return dict(experiment="B back-only", input=en_list, output=back,
                survived=survived, survival=f"{len(survived)}/{len(NAMES)}")


def exp_c():
    """Same English question to both reasoners. Do they invent names unprompted?"""
    out = {}
    for repo in REASONERS:
        m, t = load(repo)
        think = True if "Qwen3.6" in repo else None
        text, secs = call(m, t, prompts.REASONER_SYSTEM, REASONER_Q, 2500, thinking=think)
        out[repo] = dict(answer=text, secs=secs)
        del m
    return dict(experiment="C reasoner", question=REASONER_Q, outputs=out)


if __name__ == "__main__":
    which = sys.argv[1].lower()
    res = {"a": exp_a, "b": exp_b, "c": exp_c}[which]()
    dest = f"{HERE}/outputs/isolate_{which}.json"
    json.dump(res, open(dest, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "outputs"},
                     ensure_ascii=False, indent=1)[:2000])
    print(f"\nwrote {dest}")
