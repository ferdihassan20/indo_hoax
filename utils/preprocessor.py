import json
import os

import pandas as pd

from transformers import AutoTokenizer

import lightning as L

from tqdm import tqdm
#Operasi matrix
import torch 

#Manage data
from  torch.utils.data import TensorDataset, DataLoader

class Preprocessor(L.LightningDataModule):

    def __init__(self, batch_size):
        super(Preprocessor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
        self.batch_size = batch_size

    def load_data(self):
        dataset = pd.read_csv("dataset/turnbackhoax_data.csv")
        dataset = dataset[["title", "label", "narasi", "counter"]]
        
        return dataset
    
    def preprocessor(self):
        dataset = self.load_data()

        if not os.path.exists("dataset/train_set.pt") \
            and not os.path.exists("dataset/val_set.pt") \
            and not os.path.exists("dataset/test_set.pt"):

            x_ids, x_att, y = [], [], []
            
            for _, data in tqdm(dataset.iterrows(), total = dataset.shape[0], desc = "Preprocessing hoax"):

                '''
                    1 = real
                    0 = fake
                '''

                title = data["title"]
                label = data["label"]
                narasi = data["narasi"]
                counter = data["counter"] #facts nya


                narasi = narasi.replace("['", "").replace("']", "")
                narasi = narasi.replace("', ", ". ")
                narasi = narasi.replace(".. ", ". ")


                counter = counter.replace("['", "").replace("']", "")
                counter = counter.replace("', ", ". ")
                counter = counter.replace(".. ", ". ")
        
                narasi_tok = self.tokenizer(
                    f"{title} {narasi}",
                    max_length = 200,
                    truncation = True,
                    padding = "max_length",
                    )
                
                x_ids.append(narasi_tok["input_ids"])
                x_att.append(narasi_tok["attention_mask"])
                y.append([1,0])
                
                counter_tok = self.tokenizer(
                    f"{title} {narasi}",
                    max_length = 200,
                    truncation = True,
                    padding = "max_length",
                    )
                
                x_ids.append(counter_tok["input_ids"])
                x_att.append(counter_tok["attention_mask"])
                y.append([0,1])

            x_ids = torch.tensor(x_ids)
            x_att = torch.tensor(x_att)
            # y = label (0 = fake, 1 = real)
            y = torch.tensor(y)


            #Seperti List tetapi bisa banyak

            #Misah 80/20
            #Ration training (90 train, 10 val)
            train_val_len = int(x_ids.shape[0] * 0.8)
            train_len = int(train_val_len  * 0.9)
            val_len = train_val_len - train_len
            test_len = x_ids.shape[0] - train_val_len
            
            print(f"All = {x_ids.shape[0]}")
            print(f"Train = {train_len}")
            print(f"Val = {val_len}")
            print(f"Tes = {test_len}")
            

            #Pisah data
            all_data = TensorDataset(x_ids, x_att, y)
            train_set, val_set, test_set = torch.utils.data.random_split(all_data, [train_len, val_len, test_len])

            torch.save(train_set, "dataset/train_set.pt")
            torch.save(val_set, "dataset/val_set.pt")
            torch.save(test_set, "dataset/test_set.pt")

            return train_set, val_set, test_set

        else:
            print("Load Data")
            train_set = torch.load("dataset/train_set.pt")
            val_set = torch.load("dataset/val_set.pt")
            test_set = torch.load("dataset/test_set.pt")
            
            return train_set, val_set, test_set

    def setup(self, stage = None):

        train_set, val_set, test_set = self.preprocessor()
        
        # 100 data
        # 80 = training
        # 20 = testing


        #all_data = self_preprocessor()


        # Training

        if stage == "fit":
            self.train_data = train_set
            self.val_data = val_set
        elif stage == "test":
            self.test_data = test_data

    def train_dataLoader(self):
        return DataLoader(
            self.train_data,
            #batch_size = 16
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 2
        )
    
    def val_dataLoader(self):
        return DataLoader(
            self.val_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 2
        )
    
    def test_dataLoader(self):
        return DataLoader(
            self.test_data,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 2
        )
    
    # Deep Learning
    # Data (Download, Collect) -> Preprocessing -> Training
    # Transformer / LM (BERT (Bahasanya))

if __name__ == "__main__":
    prepro = Preprocessor(batch_size=5)
    prepro.setup(stage = "fit")

    # for batch in prepro.train_dataLoader():
    #     print(batch)
    #     break