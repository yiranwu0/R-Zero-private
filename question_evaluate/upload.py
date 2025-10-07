import json
import huggingface_hub
from datasets import Dataset, DatasetDict
from huggingface_hub import login
import argparse
import json
import os
STORAGE_PATH = os.getenv("STORAGE_PATH")
HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")
print(STORAGE_PATH)
# with open('tokens.json', 'r') as f:
#     token = json.load(f)['huggingface']
# login(token=token)
parser = argparse.ArgumentParser()
parser.add_argument("--repo_name", type=str, default="")
parser.add_argument("--max_score", type=float, default=0.7)
parser.add_argument("--min_score", type=float, default=0.3)
parser.add_argument("--experiment_name", type=str, default="Qwen_Qwen3-4B-Base_all")
args = parser.parse_args()

datas= []
for i in range(8):
    try:
        with open(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json', 'r') as f:
            data = json.load(f)
            datas.extend(data)
    except:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue


for i in range(8):
    try:
        os.remove(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json')
    except:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue

scores = [data['score'] for data in datas]
#  print the distribution of scores
import matplotlib.pyplot as plt
plt.hist(scores, bins=11)
plt.savefig('scores_distribution.png')
#count the number  of score between 0.2 and 0.8 
if not args.repo_name == "":
    filtered_datas = [{'problem':data['question'],'answer':data['answer'],'score':data['score']} for data in datas if data['score'] >= args.min_score and data['score'] <= args.max_score and data['answer'] != '' and data['answer']!= 'None']
    print(len(filtered_datas))
    
    # Create HuggingFace dataset and save locally as parquet for VERL training
    train_dataset = Dataset.from_list(filtered_datas)
    local_dataset_path = f'{STORAGE_PATH}/generated_question/{args.experiment_name}_filtered_dataset.parquet'
    train_dataset.to_parquet(local_dataset_path)
    print(f"Dataset saved locally as parquet to: {local_dataset_path}")
    
    # Also save as JSON for backup/inspection
    local_json_path = f'{STORAGE_PATH}/generated_question/{args.experiment_name}_filtered_dataset.json'
    with open(local_json_path, 'w') as f:
        json.dump(filtered_datas, f, indent=2)
    print(f"Dataset also saved as JSON to: {local_json_path}")
    
    print(f"Dataset ready with {len(filtered_datas)} samples. Skipping upload to hub.")







