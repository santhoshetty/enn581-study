"""Rewords MCQ framing (the exam is written) across fragments. Idempotent."""
import glob, re
R = [("Definition MCQs here are", "Definition questions here are"), ("Typical MCQ distractors:", "Typical traps:"),
     ("Typical MCQ:", "Typical short question:"), ("Possible MCQ:", "Possible short question:"),
     ("A multiple-choice version lists 9 and 3 among the options. The distractor is the plain-G–K value.", "A short-answer version expects you to say whether 9 or 3 is wanted. The trap is stopping at the plain-G–K value."),
     ("Absolute-word distractors live here", "Absolute-word traps live here"), ("is a distractor", "is a trap"),
     ("In an MCQ the roots’", "In a short answer, the roots’"), ("quotes the slide formula in a distractor", "quotes the slide formula in a statement to judge"),
     ("spot-the-error or MCQ on", "spot-the-error or short question on"), ("Distractor:", "Trap:"), ("distractors", "traps"), ("distractor", "trap"),
     ("MCQ traps", "Traps"), ("an MCQ", "a short question"), ("MCQ", "short question"), ("multiple-choice", "short-answer"), ("Multiple-choice", "Short-answer")]
for f in sorted(glob.glob('frag/*.html')):
    s = open(f).read(); o = s
    # only touch text outside <script> blocks
    parts = re.split(r'(<script.*?</script>)', s, flags=re.S)
    for i in range(0, len(parts), 2):
        for a, b in R: parts[i] = parts[i].replace(a, b)
    s = ''.join(parts)
    if s != o: open(f, 'w').write(s); print('reworded', f)
