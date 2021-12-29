for suffix in 1 2 3
do
        for bsz in 512 1024 2048 4096 4096 8192 16384
        do
                bash train_nn_graph.sh 1 1 $bsz 2322444 2 16 bsz_x2 $suffix
        done

        for bsz in 512 1024 2048 4096 4096 8192 16384
        do
                bash train_nn_graph.sh 1 8 $bsz 2322444 2 16 bsz_x2 $suffix
        done
done