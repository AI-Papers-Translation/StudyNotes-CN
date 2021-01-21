REM pandoc ML/metadata.yaml ML/Outline.md ML/Regression.md ML/reference.md -o tmp/ML_PPT.pptx --reference-doc=template/reference.pptx
REM pandoc ML/Regression.md ML/reference.md -o tmp/ML_PPT.pptx --reference-doc=template/mathematics.pptx
pandoc ML/ML-Architecture.md -o tmp/ML_PPT.pptx --reference-doc=template/mathematics.pptx
powerpnt tmp/ML_PPT.pptx

REM -------------------------------------------------------
REM pandoc tmp/sample.md -o tmp/sample.pptx --reference-doc=template/reference.pptx
REM powerpnt tmp/sample.pptx

REM -------------------------------------------------------
REM pandoc tmp/test.md -o tmp/test.pptx --reference-doc=template/reference.pptx
REM powerpnt tmp/test.pptx

REM -------------------------------------------------------
REM pandoc ML/metadata.yaml ML/Outline.md ML/reference.md -t beamer -o tmp/ML_PPT.pdf --template=template/beamer-template.tex --pdf-engine=xelatex -V CJKmainfont="SimSun" -V mainfont="Arial"
REM sumatrapdf tmp/ML_PPT.pdf
REM TeXworks tmp/ML_PPT.tex