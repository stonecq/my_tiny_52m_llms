import pickle
path = '/llama/datasets/tiny_llms/sft_train/sft_data.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)

print(1)