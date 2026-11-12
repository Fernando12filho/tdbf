# Preflight on development round 1 (2026-11-12)

Preflight of the procedure agent (`src/tdbf/agent/preflight.py`), run live with
`scripts/procedure_preflight_live.py`: Docker only, **no model call, no cost, gold-free**.
Bugs: development round 1 (`data/dev/round1.json`, seed 20261112, pool = SWT-bench Lite minus
the 114), plus two development-pool bugs pulled only to check the sphinx (tox) and astropy
(`--tb=no` -> `--tb=short`) runners. None of the 114 is in this table.

Columns: env (trivial passing and failing test gave pass and fail/assertion), runner
(official SWE-bench command or today's pytest fallback), blocks run (issue code and inline
expressions executed on the buggy code), exception seen in run (from those runs; NameError
means the issue's code is a fragment), first inline result, top-ranked code and test
candidates (not yet checked against the fix; that is measured offline later), wall time.

| bug | env | runner | python | blocks run | exception seen in run | first inline result | top code candidate | top test candidate | secs |
|---|---|---|---|---|---|---|---|---|---|
| scikit-learn__scikit-learn-13497 | ok | official | 3.6.13 | 1 | - | discrete_features == 'auto' -> raised NameError: name 'discr | sklearn/feature_selection/mutual_info_.py | sklearn/tests/ | 8.2 |
| django__django-14608 | ok | official | 3.8.20 | 0 | - | - | django/forms/utils.py | tests/forms_tests/tests/test_formsets.py | 7.0 |
| django__django-14855 | ok | official | 3.8.20 | 1 | NameError | - | django/contrib/admin/options.py | tests/admin_utils/test_logentry.py | 8.7 |
| sympy__sympy-19254 | ok | official | 3.9.20 | 1 | - | dup_zz_mignotte_bound(f, K) -> raised NameError: name 'dup_z | sympy/polys/compatibility.py | sympy/polys/tests/test_factortools.py | 11.8 |
| psf__requests-3362 | ok | official | 3.9.20 | 1 | - | next(r.iter_content(16*1024, decode_unicode=True)) -> raised | requests/models.py | tests/test_requests.py | 6.6 |
| django__django-17051 | ok | official | 3.11.10 | 0 | - | - | django/db/models/query.py | tests/basic/tests.py | 7.0 |
| django__django-15695 | ok | official | 3.9.20 | 0 | - | - | django/db/migrations/operations/models.py | tests/migrations/test_operations.py | 6.9 |
| django__django-15498 | ok | official | 3.9.20 | 0 | - | - | - | tests/utils_tests/ | 5.4 |
| sympy__sympy-20212 | ok | official | 3.9.20 | 1 | - | 0**-oo -> 0 | sympy/core/numbers.py | sympy/core/tests/test_numbers.py | 11.8 |
| django__django-12589 | ok | official | 3.6.13 | 3 | NameError | - | django/db/models/fields/related.py | tests/migrations/test_autodetector.py | 25.9 |
| sphinx-doc__sphinx-8474 (runner check) | ok | official | 3.9.20 | 0 | - | - | - | tests/test_build_html.py | 8.0 |
| astropy__astropy-7746 (runner check) | ok | official | 3.6.13 | 1 | InconsistentAxisTypesError | - | astropy/wcs/wcs.py | astropy/wcs/tests/test_wcs.py | 9.6 |

Observations:
- All 12 environments are valid with the **official** command; the fallback was never needed.
- Preflight takes 5–25 s per bug.
- Free reproductions: sympy-20212 (`0**-oo -> 0`, the wrong value in the issue) and
  astropy-7746 (the issue's `InconsistentAxisTypesError`). Most issues in this round carry
  no runnable code; Django issues that define models give NameError (no app to hold them),
  as expected. The model adapts these in the Reproduce phase.
- django-15498 and sphinx-8474 get no code candidate (no identifiers or tracebacks in the
  issue); Locate falls back to the model's own search.
