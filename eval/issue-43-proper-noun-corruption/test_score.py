"""Regression tests for score.py's canonical matching.

Plain asserts, no test framework — the repo has no test suite and this harness is
self-contained. Run directly:

    python eval/issue-43-proper-noun-corruption/test_score.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import extract_people, find_canonical

CANON = ["세종", "성삼문", "김종직", "연산군", "이익", "김부식", "정약용", "원균",
         "황진", "심정", "인종", "최항"]

# (text, expected matches, why)
CASES = [
    # --- must now match: Korean attaches these directly to the name ---
    ("세종대왕의 재위 기간", {"세종"}, "honorific title 대왕"),
    ("성삼문과 이후 실학운동의 학자들", {"성삼문"}, "particle 과"),
    ("김종직의 추종자들이 숙청되었다", {"김종직"}, "particle 의"),
    ("연산군의 폭정", {"연산군"}, "particle 의"),
    ("정약용은 목민심서를 저술했다", {"정약용"}, "particle 은"),
    ("원균이 이끄는 함대", {"원균"}, "particle 이"),
    ("김부식에게 편찬을 명했다", {"김부식"}, "particle 에게"),
    ("세종께서 창제하셨다", {"세종"}, "honorific particle 께서"),

    # --- must still NOT match: the corruptions this eval exists to catch ---
    ("이익환은 실학자였다", set(), "이익환 is a fabricated name, not 이익 + particle"),
    ("김복식이 편찬했다", set(), "김복식 is 김부식 corrupted; must not match 김부식"),
    ("박영호가 참여했다", set(), "unrelated surface form"),
    ("세종실록에 기록되어", set(), "실록 is not a particle or title — compound work name"),
    ("원균형제", set(), "형제 is not in the suffix list"),

    # --- PARTICLE_AMBIGUOUS: the particle form is not attributable (#48) ---
    ("**황진이**, **유자광**", set(), "황진이 is the poet, a different real person"),
    ("백성의 이익을 위해", set(), "이익 here is the common noun 'profit'"),
    ("학자들의 심정을 이해", set(), "심정 here is the common noun 'feelings'"),
    ("다양한 인종이 공존했다", set(), "인종 here is the common noun 'race'"),
    # ...but the bare and title-bearing forms still count for the same entities
    ("**이익 (1681-1764)**", {"이익"}, "bare name with dates"),
    ("이익(영남 학파)", {"이익"}, "bare name, parenthetical gloss"),
    ("인종(재위 1122-1146)은 왕권을", {"인종"}, "bare name, reign dates"),
    ("황진 장군이 진주성을 지켰다", {"황진"}, "title 장군 still attaches"),
    ("황진, 이억기, 정운", {"황진"}, "bare name in a list"),
    # The cost of the rule, pinned so it is not mistaken for a bug later: the general's
    # own subject form is 황진이, indistinguishable from the poet, so it is given up.
    ("황진이 이끄는 부대가 진주성을 지켰다", set(),
     "genuine 황진 + subject 이 is sacrificed — same string as the poet"),
    # A bare-string homonym is out of reach of any suffix rule, and the deny-list must
    # not be described as covering it: the Goryeo ruler 崔沆 scores as the Joseon
    # scholar 崔恒 either way. Pinned so the limitation stays visible.
    ("최항이 무신정권을 장악했다", {"최항"}, "era confusion still scores — 崔沆 vs 崔恒"),
    ("최항, 집현전 학자", {"최항"}, "bare 최항 counts, as it must"),

    # --- unchanged behaviour ---
    ("이순신과 원균", {"원균"}, "plain particle case, 이순신 not in CANON"),
    ("정약용, 박지원", {"정약용"}, "punctuation boundary still works"),
    ("세종", {"세종"}, "bare name at end of string"),
    ("", set(), "empty text"),
]


# (text, expected outside-canonical candidates, why) — extract_people, not recall.
EXTRACT_CASES = [
    ("세종은 신숙주 대신 신석주가 훈민정음을 편찬했다", "신석주",
     "the 신숙주->신석주 fabrication, particle-attached (#46)"),
    ("김복식의 삼국사기", "김복식", "particle 의 stripped"),
    ("박영호가 참여했다", "박영호", "particle 가 stripped"),
    ("신석주 편찬", "신석주", "bare form was already surfaced"),
    # A multi-syllable suffix pushed a 3-syllable name out of PLAIN's 4-syllable window,
    # so 17 of the 37 declared titles/particles were unreachable — 신석주에게 surfaced
    # only the unrelated 전달했다. PLAIN_SUFFIXED splits the suffix off instead.
    ("신석주에게 전달했다", "신석주", "2-syllable particle 에게"),
    ("신석주께서 말씀하셨다", "신석주", "honorific particle 께서"),
    ("신석주대왕의 명으로", "신석주", "title 대왕 plus particle 의"),
    ("박영호처럼 행동했다", "박영호", "2-syllable particle 처럼"),
    ("이덕주한테 물었다", "이덕주", "2-syllable particle 한테"),
]
# Tokens the stripped path must NOT introduce.
EXTRACT_NEGATIVE = [
    ("구조를 바꾸었다", "구조", "two syllables after stripping is ordinary vocabulary"),
    ("문제가 있었다", "문제", "same"),
    ("체제를 유지하는 세력", "유지하", "verb stem, not a name"),
    ("압박을 강요받은 국왕", "강요받", "verb stem, not a name"),
    ("정권을 전복하려는 시도", "전복하려", "conjugated verb stem in 려"),
    # Attached forms only: bare 선발되어 (no particle) reaches plausible_name directly and
    # has always been surfaced, which these guards do not govern.
    ("대표로 선발되어야 한다", "선발되어", "conjugated verb stem in 어, particle 야"),
]


def main():
    failed = 0
    for text, expected, why in CASES:
        got = find_canonical(text, CANON)
        ok = got == expected
        if not ok:
            failed += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {why}")
        if not ok:
            print(f"        text={text!r}")
            print(f"        expected={sorted(expected)}  got={sorted(got)}")
    total = len(CASES)
    for text, want, why in EXTRACT_CASES:
        got = extract_people(text)
        ok = want in got
        total += 1
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {why}")
        if not ok:
            print(f"        text={text!r}  want {want!r} in got={sorted(got)}")
    for text, unwanted, why in EXTRACT_NEGATIVE:
        got = extract_people(text)
        ok = unwanted not in got
        total += 1
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {why}")
        if not ok:
            print(f"        text={text!r}  unwanted {unwanted!r} in got={sorted(got)}")
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
