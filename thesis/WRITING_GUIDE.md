# Writing guide — TDBF research paper

A reference for *how* to write the prose, so the per-section briefs can stay
short. Consult this whenever a paragraph won't come out.

---

## 0. This is a paper, not a thesis

The document was converted from a report-class thesis to an ACM `acmart`
research paper on 2026-09-21 — single-column manuscript style, the layout of
ReProAgent (arXiv:2607.09123) and of ACM software-engineering venues. The
structure is `thesis/main.tex` + `thesis/sections/`, and the differences are
not only formatting: they change what each paragraph is allowed to do.

| Thesis habit | Paper equivalent |
|---|---|
| "This thesis presents…" | "This paper presents…" / "We present…" |
| "Chapter 3 documents…" | "Section~\ref{sec:approach} documents…" — always `\ref`, never a hard-coded number |
| `\chapter` + 5 `\section`s | `\section` + at most 4 `\subsection`s |
| Survey everything relevant | Cite only what shapes the design or the claim |
| Defend a decision over a page | State the decision, then the reason, in one sentence |
| Results and discussion far apart | Section 5 reports, Section 6 interprets — never both in one place |
| Contributions implied | Contributions listed explicitly in the introduction |

**Space is the real constraint.** The target is 12–15 pages in this layout,
excluding references. Every section file opens with its page budget; when a
section runs over, cut a paragraph rather than shrinking the font or the
margins. The per-section briefs in `sections/*.tex` are written to those
budgets, and they sum to about 12.5 pages.

**Layout mechanics.** Single column, so size figures and tables to `\linewidth`
— there is no `figure*`/`table*` distinction here. Sections are numbered in
arabic (1, 2.1, 2.1.1); refer to them with `\ref`, never a hard-coded number.
Citations are numeric ACM Reference Format: `\cite{key}` renders as `[7]`.

**Class options** (`main.tex`, first line). `acmsmall,screen` is what we build
— ACM's journal trim, 6.75in × 10in, with the same margins on every page. Add
`review` to get line numbers when sending the PDF to your advisor for
comments; swap `acmsmall` for `sigconf` if a venue wants the two-column ACM
proceedings layout. Do **not** use `manuscript`: it is a two-sided format, so
it mirrors the margins and every even page sits ~37pt right of every odd one.

**One temporary line.** `\raggedbottom` sits in the preamble because acmsmall
normally stretches a page's vertical glue to justify the bottom edge, which
looks like random gaps while sections 2–7 are still empty. Delete it once the
sections carry prose.

**Placeholders.** `\ph{...}` renders in red brackets and marks a number the
sweep has not produced yet. Grep for `\ph{` before any submission; the paper is
not finished while one remains.

---

## 1. The only paragraph shape you need

Almost every paragraph in a paper is the same four moves. If a paragraph feels
stuck, it is usually missing one of them.

| Move | Job | Length |
|---|---|---|
| **Claim** | The first sentence says the paragraph's point. Not context — the point. | 1 sentence |
| **Unpack** | Explain the claim: define a term, draw the distinction, give the mechanism. | 1–2 sentences |
| **Support** | Cite, or give the concrete instance from your system. | 1 sentence |
| **Consequence** | So what? Why the reader should carry this into the next paragraph. | 1 sentence |

**Rule: the first sentence of a paragraph should be readable on its own.** A
reader skimming only your first sentences should still get the argument. Test
your draft this way — it catches weak paragraphs faster than anything else.

---

## 2. Sentence-level rules

**One idea per sentence.** If you used a comma to join two complete thoughts,
make it a period instead. This is the single biggest fix available to your
current draft.

> ✗ Since, coding and building software are very complex tasks, it is important
>   to understand the phases on programming, because each one of have different
>   priorities and goals, making it hard for All purpose LLMs to give garanteed
>   help on all of them.
>
> ✓ Software construction is not a single activity but a sequence of phases.
>   Each phase has its own goals and its own notion of correctness. A model that
>   performs well in one therefore offers no guarantee of performing well in
>   another.

Three sentences, three ideas. Same content, and now a reader can follow it.

**Put the subject early.** English readers hold a sentence open until they meet
its subject and verb. Front-loaded subordinate clauses ("Since…", "Because…",
"In order to…") make the reader wait. Start with the actor.

> ✗ In order to evaluate whether the ordering matters, an experiment was run.
> ✓ We ran a paired experiment to test whether the ordering matters.

**Cut intensifiers.** *very, really, wildly, extremely, quite, significantly*
(unless statistically significant). They weaken academic prose rather than
strengthen it. Replace with a concrete fact: not "very complex" but "spanning
2,294 real issues across 12 repositories".

**Prefer verbs to noun-phrases.** "We measure X" beats "the measurement of X is
performed".

---

## 3. Hedging — how confident may you sound?

Match the verb to the evidence you actually have. Committees read for this.

| You have | Use | Example |
|---|---|---|
| Published result | *shows, demonstrates, reports* | Cheng et al. **show** that… |
| Your own measured result | *we find, we observe, our results indicate* | We **find** that… |
| A result you have not yet run | *this paper tests whether, we hypothesise* | This paper **tests whether**… |
| A mechanism you reason about | *should, is expected to, plausibly* | Earlier feedback **should** reduce unfocused edits. |

**Right now you have run no sweep.** Until Section 5 carries real numbers, every claim about
the effect of test-first ordering is in row 3. Write "this paper tests whether",
never "this paper shows that". Reviewers punish overclaiming far harder than
they punish modesty.

---

## 4. Patterns to watch in your own drafts

Recurring in the §1.1 draft and in the proposal — worth a targeted pass:

| Pattern | Fix |
|---|---|
| `Since, X, ...` / `Although, X, ...` | No comma after a subordinating conjunction: `Since X, …` |
| `the phases on programming` | `of programming` — prepositions rarely map 1:1 from Portuguese |
| `each one of have` | `each of which has` |
| `All purpose LLMs` | `general-purpose LLMs` — lowercase, hyphenated |
| Sentence starting lowercase after a period | Capitalise |
| `wildly / garanteed / debbuging / planing / maintainance` | widely / guaranteed / debugging / planning / maintenance |
| Comma-chained clauses running 60+ words | Split into 2–3 sentences |

Run a spellcheck pass over every section before showing it to your advisor.
LaTeX hides typos well; readers do not.

---

## 5. Citation mechanics

**Format.** Author, then a tie, then the cite. The tie `~` prevents a line break
between name and bracket:

```latex
Yang et al.~\cite{Yang2024SWEagent} introduced SWE-agent.
```

Never `~yang et al.,~\cite{...}` — that leading tie inserts a stray space, the
name should be capitalised, and the comma does not belong.

**Two ways to cite, and they mean different things:**

- *Integral* — the author is the grammatical subject. Use when the study itself
  matters: `Cheng et al.~\cite{Cheng2025Agentic} showed that…`
- *Non-integral* — the cite sits at the end. Use when the fact matters, not who
  found it: `BRTs improve repair outcomes~\cite{Cheng2025Agentic}.`

Vary them. All-integral prose reads like a list of names; all-non-integral prose
reads like it has no sources.

**Not all sources carry equal weight.** When you cite something that is not
peer-reviewed — a talk, a blog post, a practitioner report — say so in the
sentence. `Akita2026MethAI` is a practitioner account, so frame it as
*practitioner reports describe…*, not as a study. Doing this openly is a
strength: it shows you know the difference. Hiding it is what gets questioned.

---

## 6. Tense

| Where | Tense |
|---|---|
| Prior work | past: *Yang et al. introduced…* |
| Established facts | present: *SWE-bench contains 2,294 issues.* |
| Your system's design | present: *The fix agent receives the validated test.* |
| What you did | past: *We ran both arms on the same sample.* |
| Paper navigation | present: *Section 3 documents the system.* |

The proposal is written in future tense (*we will evaluate*). Every sentence you
port from it needs its tense changed.

---

## 7. When a paragraph will not come

In order, these usually unstick it:

1. **Say it out loud in Portuguese, in one sentence.** Then translate that one
   sentence. That is your claim. The rest of the paragraph supports it.
2. **Write the ugly version first.** Bad prose that says the right thing can be
   fixed; a blank page cannot. Never edit while drafting.
3. **Check the paragraph has a job.** If you cannot name what it does that its
   neighbours do not, it should be merged or cut.
4. **Cut the first sentence.** Drafts often open with a windup. The real claim
   is frequently sentence two.
