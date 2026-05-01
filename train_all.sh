mkdir -p logs

bash train_stage1_FakeAVCeleb.sh 2>&1 | tee logs/train_stage1_FakeAVCeleb.log
echo "train_stage1_FakeAVCeleb exit code: ${PIPESTATUS[0]}"

bash train_stage2_FakeAVCeleb.sh 2>&1 | tee logs/train_stage2_FakeAVCeleb.log
echo "train_stage2_FakeAVCeleb exit code: ${PIPESTATUS[0]}"

bash train_stage1_MAVOS-DD.sh 2>&1 | tee logs/train_stage1_MAVOS-DD.log
echo "train_stage1_MAVOS-DD exit code: ${PIPESTATUS[0]}"

bash train_stage2_MAVOS-DD.sh 2>&1 | tee logs/train_stage2_MAVOS-DD.log
echo "train_stage2_MAVOS-DD exit code: ${PIPESTATUS[0]}"