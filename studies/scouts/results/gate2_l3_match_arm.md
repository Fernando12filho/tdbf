# Why gate 2 did not trigger: the match arm's 8 L3 tests (2026-11-24)

Match arm (`scouts-dev-match`, 30 bugs): the judge said *match* on 29 of 30 first tests, including
all 8 tests the offline label calls L3. Each L3 test was re-run on the gold-fixed tree (offline
analysis; gold is allowed here) to read why it still fails. Data: `logs/scouts/gate2_l3_match.json`.

| group | bug | the test asserts | the gold fix does | catchable from the issue text? |
|---|---|---|---|---|
| A. invented value | sympy-19254 | `dmp_zz_mignotte_bound(f) == 4` | another bound value | yes (the issue states no value) |
| B. over-specified | sympy-13773 | `2 @ B` raises `ValueError` (NumPy's type, from the issue's analogy) | raises `TypeError` | likely (the issue says only "should fail") |
| B. over-specified | django-14155 | one exact `ResolverMatch.__repr__` string | a different, also informative repr | likely (no format in the issue) |
| B. over-specified | django-11630 | `run_checks()` returns `[]` | returns a warning (W035) instead of the error | likely (the issue only asks "is this correct?") |
| C. fix ≠ issue | django-15695 | backward restores the old auto-generated name (as the issue says) | backward is a no-op | no |
| C. fix ≠ issue | django-11905 | non-boolean `__isnull` raises `ValueError` ("raise an error", issue) | deprecation warning | no |
| C. fix ≠ issue | sympy-17022 | the issue's own example, `Identity(n)` with symbolic `n` | NotImplementedError for symbolic `n` | no |
| C. fix ≠ issue | django-12589 | `GROUP BY (SELECT ...` as in the issue's SQL | groups by the primary key | no |

Reading:

- **Group C (4 of 8) is out of reach of any gold-free check.** The tests follow the issue; the
  benchmark's fix chose a different behaviour. Against the issue they are correct tests.
- **Groups A and B (4 of 8) are judge leniency.** The prompt rejects invented values, yet the judge
  accepted one (it has no tools to compute the right value), and the prompt has no rule against
  asserting an exception type, exact text or format the issue does not state.
- Upper bound for a stricter text-only judge on this arm: 4 more retries, each of which still has to
  produce an L4 test.
