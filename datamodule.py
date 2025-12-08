import random
from datasets import load_dataset, Dataset

class EmotionDataset:
    def __init__(self, data_path, seed, split):
        ds = load_dataset(data_path, "split")

        self.dataset = ds[split]
        self.seed = seed

        self.label_name = self.dataset.features["label"].names

        random.seed(seed)
        idxs = list(range(len(self.dataset)))
        random.shuffle(idxs)
        self.dataset = self.dataset.select(idxs)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        return{
            "text": ex["text"],
            "label_id" : ex["label"],
            "label" : self.label_name[ex["label"]],
        }


    def __len__(self):  
        return len(self.dataset)
