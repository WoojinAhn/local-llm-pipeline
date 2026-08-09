"""Contract tests for mlx-pipeline.py — what pipeline() returns, and what renders it.

#71 split computation from rendering so #8/#30/#64 and #65 can consume the pipeline
without going through the terminal. These tests pin that contract:

  - pipeline(query) is silent and returns a PipelineResult carrying its own evidence
  - pipeline_events() emits tokens for every generating stage, final channel only
  - ConsoleRenderer + render_result() reproduce the terminal output exactly (GOLDEN)
  - the pre-#71 drivers (--reasoner-only / --qwen-only / --translate-only) still stream

Plain asserts, no test framework — matching eval/proper-noun-preservation/test_*.py.
No model is loaded and no network is reached: every generation is stubbed, so this runs
in under a second. Run directly:

    .venv/bin/python test_pipeline_contract.py

GOLDEN is a regression guard, not a spec. #64 will deliberately change what the terminal
shows; when it does, re-generate GOLDEN in the same commit rather than loosening the
assertion — the point is that the output never changes by accident.
"""
import importlib.util, io, json, os, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rich.console import Console
import prompts

# --- canned generations -------------------------------------------------------------

KO2EN_SEARCH = "Explain the causes of the Imjin War.\nSEARCH:yes"
KO2EN_NOSEARCH = "Explain the causes of the Imjin War.\nSEARCH:no"
KO2EN_SNIPPETS = "1. [Korean title] a translated snippet"
ENGLISH_QUERY = "Explain the causes of the Imjin War."
ANALYSIS = "## Causes\n\n- Toyotomi's ambition\n- Joseon's unpreparedness"
HIDDEN = "secret chain of thought that must never be shown"
REASON_RAW = (f"<|channel|>analysis<|message|>{HIDDEN}"
              f"<|end|><|start|>assistant<|channel|>final<|message|>{ANALYSIS}")
EN2KO = "## 원인\n\n- 도요토미의 야심\n- 조선의 무방비"
QUERY = "임진왜란의 원인을 설명해줘"

# The renderer's output for the search path, at width 82. Regenerate with
# --print-golden. Trailing whitespace is normalized away before comparing: Rich pads
# Markdown headings out to the full width, and a literal carrying that padding is one
# reformat away from a spurious failure. Everything that carries meaning — line order,
# blank lines, indentation, the rules and their titles — is still pinned.
GOLDEN = """\
[1/4] Translating to English...
  → Explain the causes of the Imjin War.
[2/4] Searching web...
  → 2 results found
  → Translating Korean results to English...

[3/4] GPT-OSS reasoning...
[4/4] Translating to Korean...

──────────────────────────────── English Analysis ────────────────────────────────
Causes

 • Toyotomi's ambition
 • Joseon's unpreparedness
─────────────────────────────── Korean Translation ───────────────────────────────
원인

 • 도요토미의 야심
 • 조선의 무방비
"""


def normalize(text):
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


class Resp:
    def __init__(self, text):
        self.text = text


class FakeTokenizer:
    """The pipeline only ever asks the tokenizer to render a chat template."""

    def apply_chat_template(self, messages, **kw):
        return json.dumps(messages, ensure_ascii=False)


def fake_stream_generate(search_flag):
    """Dispatch on the system prompt, in chunks small enough to split every marker."""
    def stream_generate(model, tokenizer, prompt=None, max_tokens=None, prompt_cache=None):
        msgs = json.loads(prompt)
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next(m["content"] for m in msgs if m["role"] == "user")
        if not system:
            out = "LOCAL:no"  # judge_location sends a user turn only
        elif system == prompts.TRANSLATE_KO_TO_EN:
            out = (KO2EN_SNIPPETS if user.startswith("1. [")
                   else (KO2EN_SEARCH if search_flag else KO2EN_NOSEARCH))
        elif system == prompts.TRANSLATE_EN_TO_KO:
            out = EN2KO
        else:
            out = REASON_RAW
        for i in range(0, len(out), 7):
            yield Resp(out[i:i + 7])
    return stream_generate


def fake_web_search():
    m = types.ModuleType("web_search")
    ko = [{"title": "한국 기사", "url": "https://ko.example/1", "snippet": "한국어 스니펫"}]
    en = [{"title": "EN article", "url": "https://en.example/1", "snippet": "english snippet"}]
    m.search_both = lambda ko_q, en_q, is_local=False: ([dict(r) for r in ko],
                                                        [dict(r) for r in en])
    m.localize_en_query = lambda q, is_local: f"{q} South Korea" if is_local else q
    m.format_search_context = lambda k, e: "context"
    return m


def load_pipeline():
    """mlx-pipeline.py is not an importable name, so load it by path."""
    spec = importlib.util.spec_from_file_location(
        "mlx_pipeline_under_test", f"{HERE}/mlx-pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load_pipeline()


def stub(search_flag=True, width=82):
    """Reset the module to a clean, model-free state. Returns the console buffer."""
    buf = io.StringIO()
    MOD.console = Console(file=buf, width=width, force_terminal=False, highlight=False)
    MOD.stream_generate = fake_stream_generate(search_flag)
    MOD._qwen_model = MOD._reasoner_model = object()
    MOD._qwen_tokenizer = MOD._reasoner_tokenizer = FakeTokenizer()
    MOD._reasoner_cache = None
    MOD._reasoner_history = []
    sys.modules["web_search"] = fake_web_search()
    return buf


class capture_stdout:
    def __enter__(self):
        self._real, sys.stdout = sys.stdout, io.StringIO()
        return self

    def __exit__(self, *exc):
        self.text, sys.stdout = sys.stdout.getvalue(), self._real


def render(search_flag=True, force_search=None):
    """Drive the pipeline exactly as main() does. Returns (console text, result)."""
    buf = stub(search_flag)
    renderer = MOD.ConsoleRenderer()
    try:
        result = MOD.pipeline(QUERY, force_search=force_search, on_event=renderer)
    finally:
        renderer.close()
    MOD.render_result(result)
    return buf.getvalue(), result


# --- checks ---------------------------------------------------------------------------

CHECKS = []


def check(label):
    def register(fn):
        CHECKS.append((label, fn))
        return fn
    return register


@check("pipeline(query) prints nothing at all")
def _():
    buf = stub()
    with capture_stdout() as out:
        MOD.pipeline(QUERY)
    assert buf.getvalue() == "", repr(buf.getvalue())
    assert out.text == "", repr(out.text)


@check("the result carries the English query with the SEARCH line peeled off")
def _():
    _, r = render()
    assert r.english_query == ENGLISH_QUERY, repr(r.english_query)
    assert "SEARCH:" not in r.english_query


@check("the result carries the analysis and the Korean translation")
def _():
    _, r = render()
    assert r.english_analysis == ANALYSIS, repr(r.english_analysis)
    assert r.korean_result == EN2KO, repr(r.korean_result)


@check("the reasoner's analysis channel never reaches the result")
def _():
    _, r = render()
    assert HIDDEN not in r.english_analysis
    assert "<|channel|>" not in r.english_analysis


@check("the result carries the search queries as actually issued, not as typed")
def _():
    _, r = render()
    # judge_location answers LOCAL:no here, so the English query is passed through;
    # the point is that the field reports what search_both received, not english_query.
    assert r.search_queries == {"ko": QUERY, "en": ENGLISH_QUERY}, repr(r.search_queries)


@check("the result keeps Korean source URLs, which the reasoner context drops")
def _():
    _, r = render()
    assert [(s["lang"], s["url"]) for s in r.sources] == [
        ("ko", "https://ko.example/1"), ("en", "https://en.example/1")], repr(r.sources)


@check("the result carries per-stage timings plus a total")
def _():
    _, r = render()
    assert set(r.timings) == {"ko2en", "search", "reason", "en2ko", "total"}, repr(r.timings)
    assert all(isinstance(v, float) for v in r.timings.values()), repr(r.timings)


@check("no search: no queries, no sources, no search timing")
def _():
    _, r = render(search_flag=False)
    assert r.search_performed is False
    assert r.search_queries == {} and r.sources == [], repr((r.search_queries, r.sources))
    assert "search" not in r.timings, repr(r.timings)


@check("/search and /nosearch override the judge in both directions")
def _():
    _, forced = render(search_flag=False, force_search=True)
    _, skipped = render(search_flag=True, force_search=False)
    assert forced.search_performed is True and forced.sources
    assert skipped.search_performed is False and not skipped.sources


@check("token events escape for every generating stage")
def _():
    stub(search_flag=False)
    seen = {}
    for kind, payload in MOD.pipeline_events(QUERY):
        if kind == "token":
            stage, text = payload
            seen.setdefault(stage, []).append(text)
    assert set(seen) == {"ko2en", "reason", "en2ko"}, sorted(seen)


@check("reason tokens stream the final channel only, in pieces")
def _():
    stub(search_flag=False)
    reason = [t for k, (s, t) in ((k, p) for k, p in MOD.pipeline_events(QUERY)
                                  if k == "token") if s == "reason"]
    assert len(reason) > 1, reason
    assert "".join(reason).strip() == ANALYSIS, repr("".join(reason))
    assert HIDDEN not in "".join(reason)


@check("the event stream ends with the result, and only once")
def _():
    stub()
    kinds = [k for k, _ in MOD.pipeline_events(QUERY)]
    assert kinds[-1] == "result", kinds[-3:]
    assert kinds.count("result") == 1, kinds.count("result")


@check("the renderer reproduces the terminal output exactly (GOLDEN)")
def _():
    text = normalize(render()[0])
    assert text == GOLDEN, "\n--- got ---\n" + text + "\n--- want ---\n" + GOLDEN


@check("the renderer drops token events rather than echoing them")
def _():
    buf = stub()
    r = MOD.ConsoleRenderer()
    with capture_stdout() as out:
        r(("token"), ("reason", "should not appear"))
    assert buf.getvalue() == "" and out.text == "", repr((buf.getvalue(), out.text))


@check("the spinner starts and stops around each generating stage")
def _():
    buf = stub()
    events = []
    MOD.console.status = lambda label, **kw: types.SimpleNamespace(
        start=lambda: events.append(f"start {label}"), stop=lambda: events.append("stop"))
    r = MOD.ConsoleRenderer()
    try:
        MOD.pipeline(QUERY, on_event=r)
    finally:
        r.close()
    assert events == ["start [cyan]reasoning...[/]", "stop",
                      "start [cyan]translating...[/]", "stop"], events


@check("close() stops a live spinner and is idempotent")
def _():
    stub()
    stopped = []
    MOD.console.status = lambda label, **kw: types.SimpleNamespace(
        start=lambda: None, stop=lambda: stopped.append(1))
    r = MOD.ConsoleRenderer()
    r("busy", "reasoning...")
    r.close()
    r.close()
    assert stopped == [1], stopped


@check("reasoner history accumulates system, user and assistant turns")
def _():
    render(search_flag=False)
    assert [m["role"] for m in MOD._reasoner_history] == ["system", "user", "assistant"]
    assert MOD._reasoner_history[-1]["content"] == ANALYSIS


@check("--reasoner-only still streams the final channel to stdout")
def _():
    stub(search_flag=False)
    with capture_stdout() as out:
        value = MOD.analyze("hello", stream=True)
    assert out.text == ANALYSIS + "\n", repr(out.text)
    assert value == ANALYSIS, repr(value)


@check("--translate-only still streams and still reports the SEARCH judgment")
def _():
    stub(search_flag=True)
    with capture_stdout() as out:
        text, needs_search = MOD.translate(QUERY, direction="ko2en", stream=True)
    assert out.text == KO2EN_SEARCH + "\n", repr(out.text)
    assert (text, needs_search) == (ENGLISH_QUERY, True), repr((text, needs_search))


@check("--qwen-only still streams raw Qwen output")
def _():
    stub(search_flag=False)
    prompt = json.dumps([{"role": "system", "content": prompts.TRANSLATE_EN_TO_KO},
                         {"role": "user", "content": "x"}])
    with capture_stdout() as out:
        value = MOD._stream_qwen(MOD._qwen_model, MOD._qwen_tokenizer, prompt)
    assert out.text == EN2KO + "\n", repr(out.text)
    assert value == EN2KO, repr(value)


@check("silent generation helpers print nothing")
def _():
    stub(search_flag=False)
    with capture_stdout() as out:
        MOD.translate("x", direction="en2ko")
        MOD.analyze("x", stream=False)
    assert out.text == "", repr(out.text)


def main():
    if "--print-golden" in sys.argv:
        sys.stdout.write(normalize(render()[0]))
        return 0
    failed = 0
    for label, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {label}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {label}\n        {e}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
