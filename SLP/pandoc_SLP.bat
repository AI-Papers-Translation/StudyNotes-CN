REM pandoc -d slp/pandoc-tex.yaml
pandoc --defaults=slp/pandoc-tex.yaml --verbose



REM 增加了结束前的提醒声音
ECHO **
Code tmp/slp.tex
REM sumatrapdf slp/slp-notes.pdf