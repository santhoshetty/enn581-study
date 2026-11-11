import re, sys, html
s = open(sys.argv[1]).read()
s = re.sub(r'<script.*?</script>', '', s, flags=re.S)
s = re.sub(r'<svg.*?</svg>', '[SVG]', s, flags=re.S)
s = re.sub(r'<(h[234])[^>]*>', r'\n## ', s); s = re.sub(r'</h[234]>', '\n', s)
s = re.sub(r'<(p|li|tr|div)[^>]*>', '\n', s); s = re.sub(r'<(td|th)[^>]*>', ' | ', s)
s = re.sub(r'<[^>]+>', '', s); s = html.unescape(s)
s = re.sub(r'[ \t]+', ' ', s); s = re.sub(r'\n\s*\n+', '\n', s)
print(s)
