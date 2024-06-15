import os
import lightning as L
 
from model.hoax_detection_model import HoaxDetectionModel
from utils.preprocessor import Preprocessor

if __name__ == "__main__":

    # Menyiapkan data untuk diproses
    dm = Preprocessor (
        batch_size = 10
    )

    model = HoaxDetectionModel(model_id = "indolem/indobert-base-uncased")

    trainer = L.Trainer(
        # di mana model training, gpu > cpu karena bisa kalkulasi matrix
        accelerator = 'cpu',
        # belajar sekian data dalam 1 kali
        max_epochs = 1,
        # directory pentimpanan
        default_root_dir = 'logs/indobert'
    )

    # Bagian training data
    trainer.fit(model, datamodule = dm)

    # Bagian Testing modul
    trainer.test(datamodel = dm, ckpt_path = 'best')