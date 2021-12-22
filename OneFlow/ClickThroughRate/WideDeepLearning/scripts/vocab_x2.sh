for suffix in 1 2 3
do
        for vocab in 3200000 6400000 12800000 25600000 51200000
        do
                bash train_nn_graph.sh 1 1 16384 $vocab 7 16 vocab_x2 $suffix
        done

        for vocab in 3200000 6400000 12800000 25600000 51200000
        do
                bash train_nn_graph.sh 1 8 16384 $vocab 7 16 vocab_x2 $suffix
        done
done