# Usage: $./extract_losses_aucs.sh logfile

# 1 time 1638929475.6382291 loss 0.6981884837150574
# 1 eval_loss 0.6460890740156173 eval_auc 0.5129028989457659

logfile=${1}
grep time ${logfile} | cut -d " " -f 5 > losses.tmp
grep eval_auc: ${logfile} | cut -d " " -f 5 > aucs.tmp
paste losses.tmp aucs.tmp > ${logfile}.losses_aucs
echo "extract loss and AUC to ${logfile}.losses_aucs"
rm losses.tmp aucs.tmp
