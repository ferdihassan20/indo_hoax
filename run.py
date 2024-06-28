import os
import lightning as L
import argparse
 
from model.hoax_detection_model import HoaxDetectionModel
from utils.preprocessor import Preprocessor

def input_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_gpu", action = "store_true")
    parser.add_argument("--max_epoch", type = int, default = 10)
    parser.add_argument("--batch_size", type = int, default = 10)
    parser.add_argument("--model_id", type = str, default = "indolem/indobert-base-uncased")
    parser.add_argument("--root_dir", type = str, default = 10)
    

    args = parser.parse_args()
    
    if args.use_gpu:
        device = "gpu"
    else:
        device = "cpu"
    
    config = {
        "use_gpu": device,
        "max_epoch" : args.max_epoch,
        "batch_size" : args.batch_size,
        "model_id" : args.model_id, 
        "root_dir" : args.root_dir
    }

if __name__ == "__main__":

    config = input_parser()

    # Menyiapkan data untuk diproses
    dm = Preprocessor (
        batch_size =  config["batch_size"]
    )

    model = HoaxDetectionModel(model_id = config("model_id"))

    trainer = L.Trainer(
        # di mana model training, gpu > cpu karena bisa kalkulasi matrix
        accelerator =  config("use_gpus"),
        # belajar sekian data dalam 1 kali
        max_epochs =  config("max_epoch"),
        # directory pentimpanan
        default_root_dir =  config("root_dir")
    )

    # Bagian training data
    trainer.fit(model, datamodule = dm)

    # Bagian Testing modul
    trainer.test(datamodule = dm, ckpt_path = 'best')