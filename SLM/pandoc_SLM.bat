REM pandoc -d slm/pandoc-tex.yaml
pandoc --defaults=slm/pandoc-pdf.yaml --verbose



REM 增加了结束前的提醒声音
ECHO **
REM Code tmp/slm.tex
sumatrapdf slm/slm-notes.pdf