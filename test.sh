set -e

function train() {
  cfg=$1
  thread=$2
  bz=$3
  args="batch_size=$3,is_test=1,is_predict=1,use_gpu=0"
  prefix=$4
  /opt/paddle/bin/paddle train \
    --job=test \
    --config=$cfg \
    --predict_output_dir=result \
    --init_model_path=runs/pass-00000 \
    --use_gpu=false \
    --num_passes=1 \
    --trainer_count=$thread \
    --log_period=10 \
    --test_period=100 \
    --config_args=$args \
    > logs/$prefix-${thread}gpu-$bz.predict.log 2>&1 
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

#========single-gpu=========#
train googlenet.py 6 128 googlenet 
