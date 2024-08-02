
# %%
batch_size = 256
epochs = 150


# %%
from training.dataset import initialise_dataset

trainset,trainloader,testset,testloader,classes = initialise_dataset(batch_size, "imagenette")

# %%
from training.dataset import test_trial_batch_print


test_trial_batch_print(trainloader, batch_size,classes)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG16(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5, l2_reg=0.0001):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(4608, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))
        self.l2_reg = l2_reg
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def l2_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2)
        return self.l2_reg * l2_loss

# %%
from pbms.pbm_v1 import PBM_v1
from training.data_collection import ExperimentDataCollector
import torch.optim as optim

pr_dropout = 0.08
net = VGG16(num_classes=10, dropout_rate=0.2, l2_reg=0.0001)
pbm = PBM_v1()



edc = ExperimentDataCollector("pruning_experiment_1",classes)



# %%
# net = Net()
from training.training import evaluate, one_epoch_train


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# %%
for epoch_id in range(epochs):
    print("Epoch",epoch_id,"####################################")
    edc.start_epoch(epoch_id)
    print("@@ Training")
    # new_lr = lr_scheduler(epoch_id, learning_rate)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = new_lr
    t_loss, t_acc, t_cm = one_epoch_train(net,optimizer,criterion,trainloader,epoch_id)
    t_loss, t_acc, t_cm = evaluate(net, criterion, trainloader)
    edc.report_training(t_acc,t_loss,t_cm)

    print("@@ Basic evaluation")
    v_loss, v_acc, v_cm = evaluate(net, criterion, testloader)

    edc.report_validation(v_cm,v_loss)

    if(epoch_id == 30):
        set_learning_rate(optimizer,0.01)
    if(epoch_id >= 30):
        ### Pruning
        print("@@ We will prune --- --- --- --- --- --- ---")
        net.fc[0].p = pr_dropout
        net.fc1[0].p = pr_dropout



        pbm.run_pruning_by_merging(net.layer1[0], net.layer2[0],0.64,9,net.layer1[1]) #75
        pbm.run_pruning_by_merging(net.layer2[0], net.layer3[0],0.65,9,net.layer2[1])
        pbm.run_pruning_by_merging(net.layer3[0], net.layer4[0],0.70,15,net.layer3[1]) #80
        pbm.run_pruning_by_merging(net.layer4[0], net.layer5[0],0.70,15,net.layer4[1])
        pbm.run_pruning_by_merging(net.layer5[0], net.layer6[0],0.83,30,net.layer5[1]) #82,5
        pbm.run_pruning_by_merging(net.layer6[0], net.layer7[0],0.77,30,net.layer6[1])
        pbm.run_pruning_by_merging(net.layer7[0], net.layer8[0],0.75,30,net.layer7[1])
        pbm.run_pruning_by_merging(net.layer8[0], net.layer9[0],0.88,45,net.layer8[1])
        pbm.run_pruning_by_merging(net.layer9[0], net.layer10[0],0.88,45,net.layer9[1])
        pbm.run_pruning_by_merging(net.layer10[0], net.layer11[0],0.87,45,net.layer10[1])
        pbm.run_pruning_by_merging(net.layer11[0], net.layer12[0],0.84,45,net.layer11[1])
        pbm.run_pruning_by_merging(net.layer12[0], net.layer13[0],0.82,45,net.layer12[1])

        print("@@ Basic evaluation")
        pbm_loss, pbm_acc, pbm_cm = evaluate(net, criterion, testloader)
    else:
        pbm_loss, pbm_acc, pbm_cm = v_loss, v_acc, v_cm

    edc.report_after_pbm(pbm_cm,pbm_loss)

    edc.close_epoch(net)


