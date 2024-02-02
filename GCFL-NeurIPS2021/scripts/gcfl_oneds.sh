client_=10
seed_=10
seq_len_=10

cd ..

nohup python -u GCFL_trial.py \
    --repeat 1 \
    --data_group 'PROTEINS' \
    --num_clients 10 \
    --seed 10  \
    --epsilon1 0.03 \
    --epsilon2 0.06 \
    --seq_length 10 \
    > logs/trial_PROTEINS_client_${client_}_seed_${seed_}_seq_${seq_len_}.log 2>&1 &
