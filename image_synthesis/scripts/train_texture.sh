set -ex

GPU_IDS=${1}
CLASS=${2}    # car | chair
DATASET=${3}  # voxel | df
DISPLAY_ID=$((${4}*10+1))
DATE=`date +%Y-%m-%d`
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
ROOT_DIR=${SCRIPTPATH}/..
# You need to update these directories for your own pretrained shape and texture models.
MODEL2D_DIR=${ROOT_DIR}/final_models/models_2D/${CLASS}_${DATASET}/latest
MODEL3D_DIR=${ROOT_DIR}/final_models/models_3D/${CLASS}_${DATASET}
CHECKPOINTS_DIR=${ROOT_DIR}/checkpoints/texture/${CLASS}_${DATASET}/${DATE}/

# command
python train.py --gpu_ids ${GPU_IDS} \
  --display_id ${DISPLAY_ID} \
  --dataset_mode image_and_${DATASET} \
  --model 'texture' \
  --class_3d ${CLASS} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --model2D_dir ${MODEL2D_DIR} \
  --model3D_dir ${MODEL3D_DIR} \
  --random_shift --color_jitter \
  --suffix {class_3d}_${DATASET}
