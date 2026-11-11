# ENN581 Robot Motion, Control and Planning — exam-prep site

A single-page study site and a printable revision booklet for QUT ENN581 (W1–W12), built for an open-book, written, 2-hour exam.

**Live site:** https://santhoshetty.github.io/enn581-study/

## What is here

| Path | Contents |
|---|---|
| `index.html` | The site: study plan, 12 topic tabs (with 3D widgets and a PDF viewer), the W13 sample questions solved, two mock exams, 72 practice questions, and the booklet |
| `booklet.pdf` | 25-page A4 landscape revision booklet (print at 100 %, double-sided, flip on the short edge) |
| `mock-*-questions.pdf`, `mock-*-solutions.pdf` | Two mock papers to do on paper, with solutions |
| `notes/` | Lecture decks and practical sheets (metadata stripped) |
| `src/` | Build scripts, page fragments, weekly digests, `checks/` (Python scripts that verify every number), and `tests/` (Playwright) |
| `booklet/` | LaTeX source of the booklet (`tectonic -X compile main.tex`) and its check scripts |

## Conventions

Craig's proximal (modified) DH, ᴬT_B = pose of {B} in {A}, right-handed and z-up, radians. Content beyond the slides is marked "added".

## Rebuild

```
python3 src/build.py          # writes dist/index.html and dist/gh-index.html
python3 src/checks/w02.py     # one check script per week, plus mock.py, qa1.py, qa2.py, w13.py
```

Lecture material © QUT and its lecturers; textbooks are cited but not reproduced.
