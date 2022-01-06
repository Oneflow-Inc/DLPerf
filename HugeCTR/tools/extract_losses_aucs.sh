# Usage: $./extract_losses_aucs.sh logfile

# [HUGECTR][08:48:15][INFO][RANK0]: Iter: 470 Time(1 iters): 0.009258s Loss: 0.137985 lr:0.001000
# [HUGECTR][08:48:15][INFO][RANK0]: Evaluation, AUC: 0.744433
# [HUGECTR][08:48:15][INFO][RANK0]: Eval Time for 20 iters: 0.007553s

logfile=${1}
grep Iter: ${logfile} | cut -d " " -f 8 > losses.tmp
grep AUC: ${logfile} | cut -d " " -f 4 > aucs.tmp
paste losses.tmp aucs.tmp > ${logfile}.losses_aucs
echo "extract loss and AUC to ${logfile}.losses_aucs"
rm losses.tmp aucs.tmp
