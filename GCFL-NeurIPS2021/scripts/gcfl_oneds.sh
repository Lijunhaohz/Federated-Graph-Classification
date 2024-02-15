client_=10
seed_=10
seq_len_=10
repeat_=1

# make sure that you are in the GCFL-NeurIPS2021 directory

data_group='PROTEINS'
# make a new directory for the data_group inside the logs directory, if it does not exist
if [ ! -d "logs/${data_group}" ]; then
    mkdir -p logs/${data_group}
fi

nohup python -u GCFL_trial.py \
    --repeat ${repeat_} \
    --data_group ${data_group} \
    --num_clients ${client_} \
    --seed ${seed_}  \
    --epsilon1 0.03 \
    --epsilon2 0.06 \
    --seq_length ${seq_len_} \
    > logs/${data_group}/${data_group}_client_${client_}_seed_${seed_}_seq_${seq_len_}.log 2>&1 &
