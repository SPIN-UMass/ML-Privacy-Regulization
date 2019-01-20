
# coding: utf-8

# In[23]:


from __future__ import print_function

import argparse
import os
import shutil
import time
import random


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np


# In[2]:


use_cuda = torch.cuda.is_available()


# In[3]:


use_cuda


# In[4]:



manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)

best_acc = 0  # best test accuracy


# In[25]:



class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model


# In[26]:


class InferenceAttack_HZ(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        super(InferenceAttack_HZ, self).__init__()
        self.features=nn.Sequential(
            nn.Linear(100,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            )
        self.labels=nn.Sequential(
           nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(64*2,256),
            
            nn.ReLU(),
            nn.Linear(256,128),
            
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        for key in self.state_dict():
            print (key)
            if key.split('.')[-1] == 'weight':    
                nn.init.normal(self.state_dict()[key], std=0.01)
                print (key)
                
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()
    def forward(self,x,l):
        
        out_x = self.features(x)
        out_l = self.labels(l)
        
        
        
            
        is_member =self.combine( torch.cat((out_x  ,out_l),1))
        
        
        return self.output(is_member)


# In[27]:


dataset='cifar100'
checkpoint_path='/mnt/nfs/work1/amir/milad/checkpoints_100cifar_alexnetdefense'
train_batch=100
test_batch=100
lr=0.05
epochs=500
state={}
state['lr']=lr


# In[52]:



def train_privatly(trainloader, model,inference_model, criterion, optimizer, epoch, use_cuda,num_batchs=10000,alpha=0.9):
    # switch to train mode
    model.train()
    inference_model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    

    first_id = -1
    for batch_idx, (inputs, targets) in (trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if first_id == -1:
            first_id = batch_idx
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        
        
        one_hot_tr = torch.from_numpy((np.zeros((outputs.size(0),100))-1)).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, targets.type(torch.cuda.LongTensor).view([-1,1]).data,1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        
        inference_output = inference_model ( outputs,infer_input_one_hot)
        #print (inference_output.mean())
        

        loss = criterion(outputs, targets) + (alpha)*(((inference_output-1.0).pow(2).mean()))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx%100==0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=500,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))
        if batch_idx-first_id >= num_batchs:
            break

    return (losses.avg, top1.avg)


# In[44]:



def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx%100==0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    ))

    return (losses.avg, top1.avg)


# In[45]:


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 100==0:
            
            print ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        ))

    return (losses.avg, top1.avg)


# In[46]:


def privacy_train(trainloader, model,inference_model, criterion, optimizer, epoch, use_cuda,num_batchs=1000):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()
    
    inference_model.train()
    model.eval()
    # switch to evaluate mode

    end = time.time()
    first_id = -1
    for batch_idx,((tr_input, tr_target) ,(te_input, te_target)) in trainloader:
        # measure data loading time
        if first_id == -1:
            first_id = batch_idx
        
        data_time.update(time.time() - end)
        tr_input = tr_input.cuda()
        te_input = te_input.cuda()
        tr_target = tr_target.cuda()
        te_target = te_target.cuda()
        
        
        v_tr_input = torch.autograd.Variable(tr_input)
        v_te_input = torch.autograd.Variable(te_input)
        v_tr_target = torch.autograd.Variable(tr_target)
        v_te_target = torch.autograd.Variable(te_target)
        
        # compute output
        model_input =torch.cat((v_tr_input,v_te_input))
        
        pred_outputs = model(model_input)
        
        infer_input= torch.cat((v_tr_target,v_te_target))
        
        mtop1, mtop5 =accuracy(pred_outputs.data, infer_input.data, topk=(1, 5))
        
        mtop1_a.update(mtop1[0], model_input.size(0))
        mtop5_a.update(mtop5[0], model_input.size(0))

        
        
        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0),100))-1)).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1,1]).data,1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        

        attack_model_input = pred_outputs#torch.cat((pred_outputs,infer_input_one_hot),1)
        member_output = inference_model(attack_model_input,infer_input_one_hot)
        
        
        
        is_member_labels = torch.from_numpy(np.reshape(np.concatenate((np.zeros(v_tr_input.size(0)),np.ones(v_te_input.size(0)))),[-1,1])).cuda()
        
        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)

        loss = criterion(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1=np.mean((member_output.data.cpu().numpy() >0.5)==v_is_member_labels.data.cpu().numpy())
        losses.update(loss.data[0], model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx-first_id > num_batchs:
            break

        # plot progress
        if batch_idx%10==0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx ,
                    size=500,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    ))

    return (losses.avg, top1.avg)


# In[47]:


def privacy_test(trainloader, model,inference_model, criterion, optimizer, epoch, use_cuda,num_batchs=1000):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()
    
    inference_model.eval()
    model.eval()
    # switch to evaluate mode

    end = time.time()
    for batch_idx,((tr_input, tr_target) ,(te_input, te_target)) in trainloader:
        # measure data loading time
        if first_id == -1:
            first_id = batch_idx

        data_time.update(time.time() - end)
        tr_input = tr_input.cuda()
        te_input = te_input.cuda()
        tr_target = tr_target.cuda()
        te_target = te_target.cuda()
        
        
        v_tr_input = torch.autograd.Variable(tr_input)
        v_te_input = torch.autograd.Variable(te_input)
        v_tr_target = torch.autograd.Variable(tr_target)
        v_te_target = torch.autograd.Variable(te_target)
        
        # compute output
        model_input =torch.cat((v_tr_input,v_te_input))
        
        pred_outputs = model(model_input)
        
        infer_input= torch.cat((v_tr_target,v_te_target))
        
        mtop1, mtop5 =accuracy(pred_outputs.data, infer_input.data, topk=(1, 5))
        
        mtop1_a.update(mtop1[0], model_input.size(0))
        mtop5_a.update(mtop5[0], model_input.size(0))

        
        
        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0),100))-1)).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1,1]).data,1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        

        attack_model_input = pred_outputs#torch.cat((pred_outputs,infer_input_one_hot),1)
        member_output = inference_model(attack_model_input,infer_input_one_hot)
        
        
        
        is_member_labels = torch.from_numpy(np.reshape(np.concatenate((np.zeros(v_tr_input.size(0)),np.ones(v_te_input.size(0)))),[-1,1])).cuda()
        
        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)

        loss = criterion(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1=np.mean((member_output.data.cpu().numpy() >0.5)==v_is_member_labels.data.cpu().numpy())
        losses.update(loss.data[0], model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx-first_id >= num_batchs:
            break

        # plot progress
#         if batch_idx%10==0:
#             print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
#                     batch=batch_idx ,
#                     size=len(trainloader),
#                     data=data_time.avg,
#                     bt=batch_time.avg,
#                     loss=losses.avg,
#                     top1=top1.avg,
#                     ))

    return (losses.avg, top1.avg)


# In[48]:


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [20,40]:
        state['lr'] *= 0.1 
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


# In[49]:


def save_checkpoint_adversary(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_adversary_best.pth.tar'))


# In[50]:


global best_acc
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if not os.path.isdir(checkpoint_path):
    mkdir_p(checkpoint_path)



# Data
print('==> Preparing dataset %s' % dataset)
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    
    
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

# prepare test data parts
transform_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])
    

if dataset == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10
else:
    dataloader = datasets.CIFAR100
    num_classes = 100



# Model
print("==> creating model ")


model = AlexNet(num_classes)


# In[51]:


model = model.cuda()


# In[37]:






#inferenece_model = torch.nn.DataParallel(inferenece_model).cuda()

cudnn.benchmark = True
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

criterion = nn.CrossEntropyLoss()
criterion_attack = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


# Resume
title = 'cifar-100'


# In[38]:


inferenece_model = InferenceAttack_HZ(100).cuda()
private_train_criterion = nn.MSELoss()


# In[39]:


optimizer_mem = optim.Adam(inferenece_model.parameters(), lr=0.00001)


# In[40]:


batch_privacy=100
trainset = dataloader(root='./data100', train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=batch_privacy, shuffle=True, num_workers=1)

trainset_private = dataloader(root='./data100', train=True, download=True, transform=transform_test)
trainloader_private = data.DataLoader(trainset, batch_size=batch_privacy, shuffle=True, num_workers=1)

testset = dataloader(root='./data100', train=False, download=False, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=batch_privacy, shuffle=True, num_workers=1)


# In[41]:


batch_privacy=100
trainset = dataloader(root='./data', train=True, download=True, transform=transform_test)
testset = dataloader(root='./data', train=False, download=False, transform=transform_test)

r = np.arange(50000)
np.random.shuffle(r)

private_trainset_intrain = []
private_trainset_intest = []

private_testset_intrain =[] 
private_testset_intest =[] 


for i in range(25000):
    private_trainset_intrain.append(trainset[r[i]])


    

for i in range(25000,50000):
    private_testset_intrain.append(trainset[r[i]])

    
r = np.arange(10000)
np.random.shuffle(r)
  
for i in range(5000):
    private_trainset_intest.append(testset[r[i]])


for i in range(5000,10000):
    private_testset_intest.append(testset[r[i]])







private_trainloader_intrain = data.DataLoader(private_trainset_intrain, batch_size=batch_privacy, shuffle=True, num_workers=1)
private_trainloader_intest = data.DataLoader(private_trainset_intest, batch_size=batch_privacy, shuffle=True, num_workers=1)


private_testloader_intrain = data.DataLoader(private_testset_intrain, batch_size=batch_privacy, shuffle=True, num_workers=1)
private_testloader_intest = data.DataLoader(private_testset_intest, batch_size=batch_privacy, shuffle=True, num_workers=1)





# In[42]:


is_best=False
best_acc=0.0
start_epoch=0
# Train and val
for epoch in range(start_epoch, 400):
    adjust_learning_rate(optimizer, epoch)

    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, epochs, state['lr']))

    train_enum = enumerate(trainloader)
    train_private_enum = enumerate(zip(trainloader_private,testloader))
    for i in range(500//2):
        
        
        if epoch>3:
            privacy_loss, privacy_acc = privacy_train(train_private_enum,model,inferenece_model,criterion_attack,optimizer_mem,epoch,use_cuda,1)
            train_loss, train_acc = train_privatly(train_enum, model,inferenece_model, criterion, optimizer, epoch, use_cuda,1,1)
            
            
            if i%10 ==0:
                print ('privacy res',privacy_acc,train_acc)
            if  (i+1)%50 ==0:
                train_private_enum = enumerate(zip(trainloader_private,testloader))
        else:
            train_loss, train_acc = train_privatly(train_enum, model,inferenece_model, criterion, optimizer, epoch, use_cuda,1000,0)
            break
        
        
    test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

    print ('test acc',test_acc)


    


    # save model
    is_best = test_acc>best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, False, checkpoint=checkpoint_path,filename='epoch%d'%epoch)


print('Best acc:')
print(best_acc)

