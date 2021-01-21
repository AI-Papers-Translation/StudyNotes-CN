REM pandoc PRML/preface.md PRML/C_01.md PRML/C_02.md PRML/C_03.md -o tmp/PRML_tmp.pdf --log=tmp/PRML_tmp.log --template=template/zYxTom.tex --pdf-engine=xelatex -V CJKmainfont="Microsoft YaHei" -V mainfont="Arial Unicode MS" -V monofont="Cambria Math"
REM pandoc PRML/C_03.md -o tmp/PRML_tmp.tex --log=tmp/PRML_tmp.log --template=template/zYxTom.tex --pdf-engine=xelatex -V CJKmainfont="Microsoft YaHei" -V mainfont="Arial Unicode MS" -V monofont="Cambria Math" --reference-location="document"

REM pandoc -d prml/prml-tex.yaml
pandoc --defaults=prml/prml-pdf.yaml



REM 增加了结束前的提醒声音
ECHO **
REM Code tmp/PRML_tmp.tex
sumatrapdf prml/prml-notes.pdf