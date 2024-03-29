client_=10
seed_=10
seq_len_=10

nohup python -u main_oneDS.py \
    --repeat 1 \
    --data_group 'PROTEINS' \
    --num_clients 10 \
    --seed 10  \
    --epsilon1 0.03 \
    --epsilon2 0.06 \
    --seq_length 10 \
    > logs/PROTEINS_client_${client_}_seed_${seed_}_seq_${seq_len_}_weighted.log 2>&1 &
