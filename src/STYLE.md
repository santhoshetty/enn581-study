# STYLE.md — how every topic tab is written (model: `frag/02-w02.html`)

Read `frag/02-w02.html` fully before writing a tab. Copy its structure and markup exactly; change only the content.

## Audience and tone
A master's student who needs near-full marks in a **written, 2-hour, open-book (printed notes only), calculator-allowed** exam, and wants to understand the subject deeply. Plain, direct English, Australian spelling (optimisation, colour, centre). Short sentences. Explain *why*, then *how*. No filler, no hype, no emoji in the text.

## File and section skeleton (in this order)
1. `<p class="eyebrow">Week N · Topic</p>` + `<h2>` (a plain question or statement the week answers) + `<p class="lead">` (2–3 sentences).
2. Notes box: `div.box.notes-box` with `[data-view="wNN-lec"]`, `[data-dl="notes/wNN-lec.pdf"]`, `a[href="notes/wNN-lec.pdf"]` (copy from W2). Add a second "View" button for a practical sheet if the week has one (`wNN-prac`).
3. `div.box.exam-box` "How this is examined": the matching W13 archetype (cite `<a class="pg" data-doc="w13-lec" data-page="P">W13 p.P</a>`), where marks come from (indicative; "marks not published"), where marks are lost.
4. `div.box.prac-box`: what the week's practical files really do (2–3 lines; read the source; never claim results that are not saved).
5. `div.box.eli5` with `div.cards` (3–5 `div.card`, ≤ 60 words each, each with an everyday or physical analogy; optional `span.inprac`) and `<div class="vids" data-vids="wNN"></div>` (videos are injected by the build — do not write video cards yourself).
6. Optional widget(s) (see Widgets).
7. One `div.sub` per slide subtopic, `id="wN-short"`, `<h3>n · Title <span class="pgs"><a class="pg" data-doc="wNN-lec" data-page="P">pp. P–Q</a></span></h3>`, containing:
   - `div.layer.picture` (tag "Picture"): what it is physically; an inline SVG where a picture teaches (see Diagrams).
   - `div.layer.derive` (tag "Derivation"): every algebra line, symbols defined with units, sign conventions stated, slide page links on each key formula.
   - `div.layer.recipe` (tag "Exam recipe"): an `<ol>` of 4–8 steps to reproduce in the exam, ending with a sanity check.
   - `div.worked` (tag "Worked example"): concrete numbers, the arithmetic shown, the result; `<span class="chk">checked</span>` **only** if `checks/wNN.py` verifies that exact number.
   - `div.angle`: the exam angle (trap, typical question, what to write).
   - `div.box.warn-box` (tag "Slides vs textbook" or "Slides vs notebook") for real errata, only after verifying.
8. `<h3>Guided problems: solve on paper first</h3>` then ≥ 5 `div.gp` (exact markup from W2): head with `<b>Pn.k · title</b>`, chips (`chip derive|calc|sketch|explain|spot`), `<span class="marks" data-id="wNN-pK" data-topic="wNN"></span>`; the question; 2–3 `details.hint`; for numeric answers an `div.ansrow` with `input[data-ans][data-tol][data-label]` (+ optional `data-wrong='[{"v":..,"msg":".."}]'` for a known wrong answer) and `button.check`, then `div.fb`; `details.sol` (full solution, every line, "Common wrong answer"); optional `p.blank` (blank-page test). Include every notebook exercise and practical-sheet question for the week.
   All five skills must appear across the week: derive, calculate, sketch, explain, spot-the-error.
9. `<h3>Mini-quiz</h3>` + 3 short `div.gp` items with ids `wNN-q1..q3`.
10. `<div class="box" data-reading="wNN"></div>` (reading map injected by the build).

## Maths
MathJax: inline `\( … \)`, display `\[ … \]`. Use `\mathbf T` for transforms, `{}^A\mathbf T_B` for "pose of B in A", `c\theta`/`s\theta` shorthand only after defining it. No raw `<` inside maths (use `\lt`). Long chains go in display maths, never inline (inline maths cannot wrap). Units always (m, rad, rad/s, N·m). Angles: say degrees or radians every time.

## Diagrams (inline SVG)
Draw the real mechanism to scale where numbers are given (state the scale in the caption), otherwise say "schematic, not to scale". Colours only through CSS variables: `var(--ink)`, `var(--muted)`, `var(--accent)`, `var(--exam)`, and frame axes `var(--axx)` (x red), `var(--axy)` (y green), `var(--axz)` (z blue). Right-handed frames; z out of the page for planar drawings (draw ⊙). Subscripts with `<tspan dy="4" font-size="10">i</tspan><tspan dy="-4">…</tspan>` — never write `z_{i}` as text. Font `IBM Plex Mono, monospace`, 12–13 px. Give every shape an explicit `fill` (use `fill="none"` for lines/arcs). Keep labels inside the viewBox. `figure.svg` + `figcaption`.

## Widgets
Use an existing kind (`data-w3d="chain"` with `data-presets`, `dh`, `ik13`) or register a new one in a `<script data-after>` block inside your fragment:
```html
<div class="widget" data-w3d="myKind" id="w5-traj"><div class="wh"><b>Title</b><span class="eyebrow">what it shows</span></div>
  <div class="wbody"><div class="stage"></div><div class="side"></div></div><div class="try"><b>Try:</b> …</div></div>
<script data-after>
W3D.kinds.myKind = function (root) {           // called once, when the tab is first shown
  const stage = root.querySelector('.stage'), side = root.querySelector('.side');
  // 2D: create a <canvas>, size it to stage.clientWidth × 340 with devicePixelRatio (≤2), draw with colours from W3D.css('--ink') etc.
  // 3D: new W3D.Scene3D(stage, {...}) and W3D.drawChain(...)
  // Controls: W3D.sliderRow(id, label, min, max, value, step, unit) → HTML string.
  // Expose for tests: window.__state[root.id] = {...numbers shown...};  W3D.instances[root.id] = { set(...) { ...; return window.__state[root.id]; } };
};
</script>
```
Every number a widget displays must come from the same function that draws it, and that function must reproduce the check-script numbers at the test inputs you list in `checks/wNN.py` (write them into `checks/wNN_vectors.json`). Redraw on `window` event `enn-theme`.

## Checks
`checks/wNN.py` imports `common.py` (`check`, `close`, `report`, sympy/numpy elementary transforms). Every number shown with `chk` must be computed there. It must end with `report()` and print `ALL CHECKS PASSED`.

## Never
Invent marks, numbers or slide content; quote textbook text; name the student; read `Assessments/`, `Mid-Sem/`, `Other-Sub-Cheatsheets/`, or student-made files in `Week-10*/Practical/week 10 practicals/` and `Week-11*/Practical/week 11 practicals/`.
