# Usage: $./extract_losses_aucs.sh logfile
logfile=${1}
grep Iter: ${logfile} | cut -d " " -f 8 > losses
grep AUC: ${logfile} | cut -d " " -f 4 > aucs
paste losses aucs > ${logfile}.losses_aucs
