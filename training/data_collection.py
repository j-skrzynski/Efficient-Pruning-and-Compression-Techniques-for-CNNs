import datetime
from typing import Dict, NamedTuple, Union
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

EpochStepDesc = NamedTuple("EpochStepDesc", [("loss",float),
                                             ("accuracy",float),
                                             ("confusion_matrix",np.ndarray)])

EpochDesc = NamedTuple("EpochDesc",[("training",Union[None, EpochStepDesc]),
                                    ("validation",Union[None, EpochStepDesc]),
                                    ("pbm",Union[None, EpochStepDesc])])

EpochId = int

class ExperimentDataCollector():

    name: str
    class_names: str

    data: Dict[EpochId, EpochDesc]

    def __init__(self,name, class_names) -> None:
        self.name = name
        self.class_names = class_names
        self.file_name = name+"__"+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.writer = SummaryWriter('runs/'+self.file_name)
        self.data={}
        np.set_printoptions(threshold=np.inf)

    def start_epoch(self, epoch_id):
        self.epoch_id = epoch_id
        self.current_epoch = EpochDesc(None,None,None)

    def report_training(self, training_accuracy, training_loss,cm):
        step = EpochStepDesc(training_loss,training_accuracy,cm)
        self.training = step

    def report_validation(self, confucion_matrix, validation_loss):
        accuracy = np.trace(confucion_matrix) / np.sum(confucion_matrix)
        step = EpochStepDesc(validation_loss,accuracy,confucion_matrix)
        self.validation = step

    def report_after_pbm(self, confucion_matrix, validation_loss):
        accuracy = np.trace(confucion_matrix) / np.sum(confucion_matrix)
        step = EpochStepDesc(validation_loss,accuracy,confucion_matrix)
        self.pbm = step


    def close_epoch(self, net):
        

        ep  = EpochDesc(self.training,self.validation,self.pbm)

        self.data[self.epoch_id] = ep

        print(f"Epoch {self.epoch_id} ==> {str(ep)}")

        with open(self.file_name+".txt", 'a') as file:
            file.write(f"Epoch {self.epoch_id} ==> {str(ep)}" + '\n')

        torch.save(net.state_dict(), f"checkpoint_{self.epoch_id}__{self.file_name}.pth")

        self.writer.add_scalars('Training vs. Validation vs. PBM Loss',
                            { 'Training' : ep.training.loss, 'Validation' : ep.validation.loss, 'PBM': ep.pbm.loss},
                            self.epoch_id )
        self.writer.add_scalars('Training vs. Validation vs. PBM Accuracy',
                                { 'Training' : ep.training.accuracy, 'Validation' : ep.validation.accuracy, 'PBM':ep.pbm.accuracy },
                                self.epoch_id )
        
        self.writer.add_scalars('Classwise Accuracy (Training)',
                                self.classwise_acc_dict(ep.training.confusion_matrix),
                                self.epoch_id )
        
        self.writer.add_scalars('Classwise Accuracy (Validation)',
                                self.classwise_acc_dict(ep.validation.confusion_matrix),
                                self.epoch_id )
        
        self.writer.add_scalars('Classwise Accuracy (PBM)',
                                self.classwise_acc_dict(ep.pbm.confusion_matrix),
                                self.epoch_id )
        
        self.writer.flush()

        

    def classwise_acc(self, cm):
        return [cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0 for i in range(cm.shape[0])]
    
    def classwise_acc_dict(self, cm):
        classwise_acc = self.classwise_acc(cm)
        return {label:classwise_acc[i] for i, label in enumerate(self.class_names)}
         