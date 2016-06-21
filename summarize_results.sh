#!/bin/bash
# Store the results in the outputfile.out with a pipe:
# e.g. summarize_results.sh > outputfile.out

results='./results'
utils='./utils'

echo "SVM"
python ${utils}/summarize_validation.py ${results}/results_Li2014_{147..162}/validation.csv \
                                        ${results}/summary_svm_Li2014.csv

#echo "GMM"
#python ${utils}/summarize_validation.py ${results}/results_Li2014_{163..177}/validation.csv \
#                               ${results}/summary_gmm_Li2014.csv
#
#echo "GMM3"
#python ${utils}/summarize_validation.py ${results}/results_Li2014_{178..182}/validation.csv \
#                               ${results}/summary_gmm3_Li2014.csv
#
#echo "Tax kernel bandwidth 0.05"
#python ${utils}/summarize_validation.py ${results}/results_Tax2008_{3..15}/validation.csv \
#                               ${results}/summary_kernel_Tax2008_bw_005.csv
#
#echo "Tax kernel bandwidth 0.1"
#python ${utils}/summarize_validation.py ${results}/results_Tax2008_{16..31}/validation.csv \
#                               ${results}/summary_kernel_Tax2008_bw_01.csv


# I removed lung-cancer Tax2008_32 as it stoped}
echo "Tax kernel bandwidth 0.05"
python ${utils}/summarize_validation.py \
       ${results}/results_Tax2008_{3..9}/validation.csv \
       ${results}/results_Tax2008_11/validation.csv \
       ${results}/results_Tax2008_{13..15}/validation.csv \
       ${results}/results_Tax2008_{32..33}/validation.csv \
       ${results}/results_Tax2008_{34..36}/validation.csv \
       ${results}/results_Tax2008_{39..40}/validation.csv \
       ${results}/results_Tax2008_{58..63}/validation.csv \
       ${results}/summary_kernel_Tax2008_bw_005.csv
