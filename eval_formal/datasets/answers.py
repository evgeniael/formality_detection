from datasets.base_dataset import BaseDataset
import os
import pandas as pd

class Answers(BaseDataset):

    def __init__(self, num_datapoints: int):
        """
        Answers data pre-processed.
        """
        self.num_datapoints = num_datapoints
        super().__init__()

    def _load_data(self):
        path = os.path.join(self.data_dir, "answers")
        data = pd.read_csv(path, delimiter = "\t", header=None, names=["avg_formal", "votes_formal", "id", "text"])
        data = self._preprocess(data)
        
        data_subsample = data.sample(n=self.num_datapoints, random_state=42)
        return data_subsample

    def _preprocess(self, data):
        data = data.dropna()
        data["avg_formal"] = data["avg_formal"].astype(float)
        data["votes_formal"] = data["votes_formal"].str.split(',')
        data["votes_formal"] = data["votes_formal"].apply(get_int_list)
        return data

    def __getitem__(self, idx):
        return self.data.iloc[idx]["id"], self.data.iloc[idx]["text"], self.data.iloc[idx]["avg_formal"], self.data.iloc[idx]["votes_formal"]

def get_int_list(l):
    return [float(x) for x in l]

if __name__ == "__main__":
    answers = Answers()
    print(len(answers))
    print(answers[0])