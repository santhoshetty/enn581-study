#!/bin/zsh
# usage: render.sh <doc-key> <page> [page2]  -> stacks 1-2 pages at 100 dpi into src/img/<key>-pA-B.png
cd "$(dirname $0)"
key=$1; a=$2; b=${3:-$2}
pdf=$(grep -m1 "^## $key " extract/index.md | sed 's/.* — //')
for p in $a $b; do pdftoppm -r ${DPI:-100} -f $p -l $p -png "../../$pdf" img/tmp-$key-$p; done
python3 - "$key" "$a" "$b" <<'PY'
import sys, glob
from PIL import Image
k,a,b=sys.argv[1:]
ims=[Image.open(sorted(glob.glob(f'img/tmp-{k}-{p}*.png'))[0]) for p in dict.fromkeys([a,b])]
W=max(i.width for i in ims); H=sum(i.height for i in ims)
c=Image.new('RGB',(W,H),'white'); y=0
for i in ims: c.paste(i,(0,y)); y+=i.height
c.save(f'img/{k}-p{a}-{b}.png'); print(f'img/{k}-p{a}-{b}.png', c.size)
PY
rm -f img/tmp-$key-*
