import lightning as L
import torch.nn as nn
import torch
from transformers import AutoModel

# Pelajari istilahnya
from torchmetrics.classification import F1Score, Accuracy, Precision, Recall

class HoaxDetectionModel(L.LightningModule):
    #Constructor (Inisialisasi)
    def __init__(self,
                 model_id,
                 lr = 1e-5,
                 dropout = 0.3,
                 hidden_size = 768, 
                 model_dim = 512,
                 num_classes = 2
                 ):
        super(HoaxDetectionModel, self).__init__()

        #Otak/language model (Fungsi kemampuan berbahasa)
        #AutoModel = Jaringan Otak
        #from_pretrained = Memori
        self.lm = AutoModel.from_pretrained(model_id)

        #Tujuan dropout, untuk improve filtering
        self.dropout = nn.Dropout(dropout)

        # seberapa dalam rasio model di optimize
        self.lr = lr

        # Save ke memori khusus untuk hoax detection
        # dikecilkan dimensinya dari 768 -> 512
        self.pre_classifier = nn.Linear(hidden_size, model_dim)
        
        # model dim = 5, num_classes (benar / salah) = 2
        self.output_layer = nn.Linear(model_dim, num_classes)

        # Activation function / Normalisasi
        self.softmax = nn.Softmax()

        # Persiapan benchmarking
        self.prepare_metrics()

        self.loss_function = nn.BCEWithLogitsLoss()



    #Call / Forward (Running)
    def forward(self, x_ids, x_att):
        self.lm(
            input_ids = x_ids,
            attention_mask = x_att
        )
        #Full output model
        # 12 * 768
        # 12 = Layernya (Filter)
        # 768 = Probabilitas

        # dimensi pooler putput
        lm_out = lm_out.pooler_output
        out = self.dropout(lm_out)

        # Tersimpan di memori khusus untuk hoax detection
        out = self.pre_classifier(out)

        # 0.0232443389234 -> 0.023244 (normalisasi) -> 0 -> 1 (rangenya)
        out = self.output_layer(out)
        out = self.softmax(out)
        return out

    def prepare_metrics(self):
        # hoax atau ga nya true false
        task = "binary"

        self.acc_metrics = Accuracy(task = task, num_classes = 2)

        self.micro = F1Score(task = task, num_classes = 2, average = "micro")
        self.macro = F1Score(task = task, num_classes = 2, average = "macro")
        self.weighted = F1Score(task = task, num_classes = 2, average = "weighted")

        self.prec_metrics_micro = Precision(task = task, num_classes = 2, average = "micro")
        self.prec_metrics_macro = Precision(task = task, num_classes = 2, average = "macro")                            
        self.prec_metrics_weighted = Precision(task = task, num_classes = 2, average = "weighted")

        self.recall_metrics_micro = Recall(task = task, num_classes = 2, average = "micro")
        self.recall_metrics_macro = Recall(task = task, num_classes = 2, average = "macro")                            
        self.recall_metrics_weighted = Recall(task = task, num_classes = 2, average = "weighted")

        self.training_step_output = []
        self.validation_step_output = []
        self.test_step_output = []

    def benchmarking_step(self, pred, target):
    
        '''
        output pred / target = 
        [
            [0.001, 0.80],
            [0.8, 0.0001],
            [0.8, 0.0001]
        ]

        y_pred -> [1, 0, 0, 0]
        '''
        pred = torch.argmax(pred, dim = 1)
        target = torch.argmax(target, dim = 1)

        metrics = {}
        metrics["acccuracy"] = self.acc_metrics(pred, target)
        metrics["f1micro"] = self.f1_metrics_micro(pred, target)
        metrics["f1macro"] = self.f1_metrics_macro(pred, target)
        metrics["f1weighted"] = self.f1_metrics_weighted(pred, target)
        metrics["prec_micro"] = self.prec_metrics_micro(pred, target)
        metrics["prec_macro  "] = self.prec_metrics_macro(pred, target)
        metrics["prec_weighted "] = self.prec_metrics_weighted(pred, target)
        metrics["recall_micro"] = self.recall_metrics_micro(pred, target)
        metrics["recall_macro"] = self.recall_metrics_macro(pred, target)
        metrics["recall_weighted"] = self.recall_metrics_weighted(pred, target)
     
        return metrics

    #    self.acc_metrics(pred, target)
    #     self.f1_metrics_micro(pred, target)
    #     self.f1_metrics_macro(pred, target)
    #     self.f1_metrics_weighted(pred, target)
    #     self.prec_metrics_micro(pred, target)
    #     self.prec_metrics_macro(pred, target)
    #     self.prec_metrics_weighted(pred, target)
    #     self.recall_metrics_micro(pred, target)
    #     self.recall_metrics_macro(pred, target)
    #     self.recall_metrics_weighted(pred, target)
            

        # pred = torch.argmax(pred, dim = 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def training_step(self, batch, batch_idx):
        x_ids, x_att, y = batch

        y_pred = self(x_ids = x_ids, x_att = x_att)

        loss = self.loss_function(y_pred, target = y.float())

        metrics = self.benchmarking_step(pred = y_pred, target = y)
        
        metrics["loss"] = loss
        self.training_step_output.append(metrics)
        self.log_dict(metrics, prog_bar = True, on_epoch = True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x_ids, x_att, y = batch

        y_pred = self(x_ids = x_ids, x_att = x_att)

        loss = self.loss_function(y_pred, target = y.float())

        metrics = self.benchmarking_step(pred = y_pred, target = y)
        
        metrics["loss"] = loss
        self.validation_step_output.append(metrics)
        self.log_dict(metrics, prog_bar = True, on_epoch = True)

        return loss

    def test_step(self, batch, batch_idx):
        x_ids, x_att, y = batch

        y_pred = self(x_ids = x_ids, x_att = x_att)

        loss = self.loss_function(y_pred, target = y.float())

        metrics = self.benchmarking_step(pred = y_pred, target = y)
        
        metrics["loss"] = loss
        self.test_step_output.append(metrics)
        self.log_dict(metrics, prog_bar = True, on_epoch = True)