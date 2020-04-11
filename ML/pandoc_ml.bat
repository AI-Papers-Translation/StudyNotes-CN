REM -------------------------------------------------------
cls
REM pandoc ML/metadata.yaml ML/Outline.md ML/Regression.md -o tmp/ML_PPT.pptx --reference-doc=template/reference.pptx

pandoc ML/Regression.md -o tmp/ML_PPT.pptx --reference-doc=template/reference.pptx

powerpnt tmp/ML_PPT.pptx
REM -------------------------------------------------------
cls
REM pandoc tmp/sample.md -o tmp/sample.pptx --reference-doc=template/reference.pptx
REM powerpnt pandoc/sample.pptx