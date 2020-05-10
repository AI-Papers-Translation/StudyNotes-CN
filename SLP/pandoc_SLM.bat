REM pandoc -d slp/pandoc-tex.yaml
pandoc --defaults=slm/pandoc-tex.yaml --verbose



REM 增加了结束前的提醒声音
ECHO **
Code tmp/slm.tex
REM sumatrapdf slm/slm-notes.pdf