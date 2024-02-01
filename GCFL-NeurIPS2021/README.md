
No file chosen

Federated Graph Classification over Non-IID Graphs
This repository contains the implementation of the paper:

Federated Graph Classification over Non-IID Graphs

Requirements
To install requirements:

pip3 install -r requirements.txt
​
Run once for one certain setting
(1) OneDS: Distributing one dataset to a number of clients:

python main_oneDS.py --repeat {index of the repeat} --data_group {dataset} --num_clients {num of clients} --seed {random seed}  --epsilon1 {epsilon_1} --epsilon2 {epsilon_2} --seq_length {the length of gradient norm sequence}
​
(2) MultiDS: For multiple datasets, each client owns one dataset (datagroups are pre-defined in setupGC.py):

python main_multiDS.py --repeat {index of the repeat} --data_group {datagroup} --seed {random seed} --epsilon1 {epsilon_1} --epsilon2 {epsilon_2} --seq_length {the length of gradient norm sequence}
​
Run repetitions for all datasets
(1) To get all repetition results:

bash runnerfile
​
(2) To averagely aggregate all repetitions, and get the overall performance:

python aggregateResults.py --inpath {the path to repetitions} --outpath {the path to outputs} --data_partition {the data partition mechanism}
​
Or, to run one file for all:

bash runnerfile_aggregateResults
​
Outputs
The repetition results started with '{\d}_' will be stored in:

./outputs/seqLen{seq_length}/oneDS-nonOverlap/{dataset}-{numClients}clients/eps_{epsilon1}_{epsilon2}/repeats/, for the OneDS setting;

./outputs/seqLen{seq_length}/multiDS-nonOverlap/{datagroup}/eps_{epsilon1}_{epsilon2}/repeats/, for the MultiDS setting.

After aggregating, the two final files are:

avg_accuracy_allAlgos_GC.csv, which includes the average performance over clients for all algorithms;

stats_performanceGain_GC.csv, which shows the performance gain among all clients for all algorithms.

*Note: There are various arguments can be defined for different settings. If the arguments 'datapath' and 'outbase' are not specified, datasets will be downloaded in './data', and outputs will be stored in './outputs' by default.

Acknowledgement
Some codes adapted from Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints

If you find this work helpful, please cite
@inproceedings{xie2021federated,
      title={Federated Graph Classification over Non-IID Graphs}, 
      author={Han Xie and Jing Ma and Li Xiong and Carl Yang},
      booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
      year={2021}
}
​
