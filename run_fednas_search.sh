GPU=$1
MODEL=$2
# homo; hetero
DISTRIBUTION=$3
ROUND=$4
EPOCH=$5
BATCH_SIZE=$6

mpirun -np 16 -hostfile ./mpi_host_file python3 ./Studium/Master/Masterarbeit/FedNAS Implementierung/FedNAS/main.py \
  --gpu $GPU \
  --model $MODEL \
  --dataset SIDD \
  --partition $DISTRIBUTION  \
  --client_number 15 \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
