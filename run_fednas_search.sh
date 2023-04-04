GPU=$1
MODEL=$2
# homo; hetero
DISTRIBUTION=$3
ROUND=$4
EPOCH=$5
BATCH_SIZE=$6


OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun --allow-run-as-root --oversubscribe -np 16 -hostfile ./mpi_host_file python3 ./main.py \
  --gpu 1 \
  --model darts \
  --dataset sidd \
  --partition hetero  \
  --client_number 15 \
  --comm_round 5 \
  --epochs 1 \
  --batch_size 8 \
