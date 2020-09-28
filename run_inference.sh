# Change this variable to location of your python (Anaconda)
export PATH="/var/www/anaconda3-temp/bin/"
# Change this variable to location of your code
export PYTHONPATH="$PYTHONPATH:/var/www/test_alzheimer/"

# Main pipeline for inference
# pip install -r requirements.txt
python3 preproc_data/r01_extract_roi_parts.py test
python3 net_v20_d121_only_tier1_finetune/r42_process_test.py
