#!/bin/bash

results='./results'
utils='./utils'

echo "SVM"
python ${utils}/summarize_validation.py ${results}/results_Li2014_{147..162}/validation.csv \
                               ${results}/summary_svm_Li2014.csv

echo "GMM"
python ${utils}/summarize_validation.py ${results}/results_Li2014_{163..177}/validation.csv \
                               ${results}/summary_gmm_Li2014.csv

echo "GMM3"
python ${utils}/summarize_validation.py ${results}/results_Li2014_{178..182}/validation.csv \
                               ${results}/summary_gmm3_Li2014.csv

echo "Tax kernel bandwidth 0.05"
python ${utils}/summarize_validation.py ${results}/results_Tax2008_{3..15}/validation.csv \
                               ${results}/summary_kernel_Tax2008_bw_005.csv

echo "Tax kernel bandwidth 0.1"
python ${utils}/summarize_validation.py ${results}/results_Tax2008_{16..31}/validation.csv \
                               ${results}/summary_kernel_Tax2008_bw_01.csv

echo "Tax kernel missing bandwidth 0.05"
python ${utils}/summarize_validation.py ${results}/results_Tax2008_{3..15}/validation.csv \
                               ${results}/results_Tax2008_{32..37}/validation.csv \
                               ${results}/results_Tax2008_{39..40}/validation.csv \
                               ${results}/summary_kernel_Tax2008_missing_bw_005.csv
