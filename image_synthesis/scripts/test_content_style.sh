set -ex

GPU_IDS=${1}
CLASS=${2}  # car | chair
DATASET=${3} # df | voxel
DISPLAY_ID=$((${4}*10+1))
DATE=`date +%Y-%m-%d`


SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
ROOT_DIR=${SCRIPTPATH}/..
CHECKPOINTS_DIR=${ROOT_DIR}/checkpoints/content_style/${CLASS}_${DATASET}/${DATE}/


# command
python3 test_content_style.py --gpu_ids ${GPU_IDS} \
  --display_id ${DISPLAY_ID} \
  --dataset_mode 'depth2real' \
  --model 'content_style' \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --class_3d ${CLASS} \
  --random_shift --color_jitter \
  --suffix {class_3d}_${DATASET} \
