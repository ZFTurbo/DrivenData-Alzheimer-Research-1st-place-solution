# Change this variable to location of your python (Anaconda)
export PATH="/var/www/anaconda3-temp/bin/"
# Change this variable to location of your code
export PYTHONPATH="$PYTHONPATH:/var/www/test_alzheimer/"

# Main pipeline for inference
# pip install -r requirements.txt
python3 preproc_data/r01_extract_roi_parts.py
# Uncomment if you need to create new KFold split
# python3 preproc_data/r03_gen_kfold_split.py
python3 net_v13_3D_roi_regions_densenet121/r31_train_3D_model_dn121.py
python3 net_v14_d121_auc_large_valid/r31_train_3D_model_dn121.py
python3 net_v20_d121_only_tier1_finetune/r31_train_3D_model_dn121.py
python3 net_v20_d121_only_tier1_finetune/r42_process_test.py
