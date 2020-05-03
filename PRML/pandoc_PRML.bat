REM pandoc PRML/preface.md PRML/C_01.md PRML/C_02.md PRML/C_03.md -o tmp/PRML_tmp.pdf --log=tmp/PRML_tmp.log --template=template/zYxTom.tex --pdf-engine=xelatex -V CJKmainfont="Microsoft YaHei" -V mainfont="Arial Unicode MS" -V monofont="Cambria Math"

REM pandoc -d prml/prml-tex.yaml
REM Code tmp/prml_tmp.tex

REM pandoc PRML/C_04.md -o tmp/PRML_tmp.tex --log=tmp/PRML_tmp.log --template=template/zYxTom.tex --pdf-engine=xelatex -V CJKmainfont="Microsoft YaHei" -V mainfont="Arial Unicode MS" -V monofont="Cambria Math"

pandoc -d prml/prml-pdf.yaml
sumatrapdf tmp/prml_tmp.pdf