#!/bin/bash
# ===========================================
# --coding:UTF-8 --
# file: start.sh
# author: Qingli Wang
# date:2022-10-26
#===========================================

module load anaconda/2020.11
source activate fe

# train for ARNet and TSN
# bash ./scripts_train_new.sh

# inference 
# bash ./full_test.sh

# =======================================IRTE===================================
# train ucf101
# bash ./train_ucf101.sh


# train hmdb51
bash ./train_hmdb.sh

# test for irte 
# bash ./test.sh

# python test.py > test.log

echo "Done"