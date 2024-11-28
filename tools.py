import torch
import LoadData2
import numpy as np
import os
from PIL import Image

def imgtoimg(D,Model,D_optimizer,inputdataG2,img2,img,inputdata,BCE_loss,y_real_,y_fake_,L1_loss,wenbiao,k1,k2,k3,k4,ssimloss,lam_ps,optimizer):
    set_requires_grad(D, True)
    set_requires_grad(Model, False)
    D_optimizer.zero_grad()
    img_pre, wenbiao_pre = Model(inputdataG2, wenbiao,img.permute(0, 3, 1, 2))

    D_result_R, x1, x40, x6, x8 = D(img2.permute(0, 3, 1, 2), inputdata)
    D_result_F, x1_f, x40_f, x6_f, x8_f = D(img_pre, inputdata)
    Pimgloss = get_Ploss([x1, x40, x6, x8], [x1_f, x40_f, x6_f, x8_f], lam_ps) * 5
    D_PLOSS = torch.max((torch.tensor(2).float().to(device) - 1 * Pimgloss), torch.tensor(0).float().to(device))
    D_loss = 0.05*(BCE_loss(D_result_R, y_real_) + BCE_loss(D_result_F, y_fake_) + D_PLOSS)
    D_loss.backward(retain_graph=True)
    D_optimizer.step()

    set_requires_grad(Model, True)
    for i in range(k1):
        # Train G
        set_requires_grad(D, False)
        optimizer.zero_grad()
        img_pre, wenbiao_pre = Model(inputdataG2, wenbiao,img.permute(0, 3, 1, 2))
        D_result_R, x1, x40, x6, x8 = D(img2.permute(0, 3, 1, 2), inputdata)
        D_result_F, x1_f, x40_f, x6_f, x8_f = D(img_pre, inputdata)
        Pimgloss = get_Ploss([x1, x40, x6, x8], [x1_f, x40_f, x6_f, x8_f], lam_ps) * k2
        G_loss0 = BCE_loss(D_result_F, y_real_)
        temp1 = L1_loss(img, img_pre.permute(0, 2, 3, 1))
        themloss = L1_loss(img[:, :, int(img.shape[2] * 0.125):int(img.shape[2] * 0.875),int(img.shape[3] * 0.125):int(img.shape[3] * 0.875)],
                           img_pre.permute(0, 2, 3, 1)[:,:, int(img.shape[2] * 0.125):int(img.shape[2] * 0.875),int(img.shape[3] * 0.125):int(img.shape[3] * 0.875)])

        L1loss1 = L1_loss(wenbiao, wenbiao_pre) * 20
        pixloss = (25 * k3 + k4) * temp1
        ssim_loss = 1 - ssimloss(img.permute(0, 3, 1, 2), img_pre)
        G_loss = 0.05* (G_loss0 + Pimgloss + pixloss + L1loss1 + 5 * themloss)
        G_loss.backward(retain_graph=True)
        optimizer.step()

def show_res_(epoch, cnt, img,img_pre,data_dir,wenbiao,wenbiao_pre,maxwenbiaoT,minwenbiaoT,fname):
    newimgpath = data_dir + '/img/' + str(epoch) + '/'
    wenbiao = wenbiao.cpu().detach().numpy()*(maxwenbiaoT-minwenbiaoT) + minwenbiaoT
    wenbiao_pre = wenbiao_pre.cpu().detach().numpy() * (maxwenbiaoT - minwenbiaoT) + minwenbiaoT
    if not os.path.exists(newimgpath):
        LoadData2.makefiles([newimgpath])
    for i in range(len(img)):
        target_image0 = img[i, :, :, :3].cpu().detach().numpy() * 127.5 + 127.5
        target_image1 = img_pre[i, :, :, :3].cpu().detach().numpy() * 127.5 + 127.5
        temp = np.concatenate((target_image0, target_image1), axis=0)
        newimgpath0 = newimgpath + fname[i][:-4] + '_%d-%d_%d-%d_epoch_%d_%d_%d.jpg' % (wenbiao_pre[i,0],wenbiao_pre[i,1],
                                                                                        wenbiao[i,0],wenbiao[i,1],epoch, cnt, i)
        Image.fromarray(np.uint8(temp)).save(newimgpath0)


def show_testres(model,testloader,epoch,data_dir,maxwenbiaoT,minwenbiaoT,device = torch.device('cuda:0')):
    cnt = 0
    model.eval()
    # for inputdata, img, wenbiao, fname in testloader:
    for dataA, dataB, fname in testloader:
        [inputdata2, img2, wenbiao2], [inputdata, img, wenbiao] = dataA, dataB
        inputdata, img, wenbiao = inputdata.to(torch.float32).to(device), img.to(torch.float32).to(device), wenbiao.to(torch.float32).to(device)  # inputdata 4,6    wenbiao4,2
        inputdata2, img2, wenbiao2 = inputdata2.to(torch.float32).to(device), img2.to(torch.float32).to(device), wenbiao2.to(torch.float32).to(device)
        inputdataG = torch.cat([inputdata, inputdata2], dim=1)
        img_pre, wenbiao_pre = model(inputdataG,wenbiao2, img2.permute(0, 3, 1, 2))
        img_pre = img_pre.permute(0, 2, 3, 1)

        lossstr = "E:  %d  %d " % (epoch, cnt)
        mean1, mean2, max1, max2 = writewenbiao([epoch, cnt], wenbiao, wenbiao_pre, data_dir,maxwenbiaoT-minwenbiaoT,lossstr)
        show_res_(epoch, cnt, img,img_pre,data_dir,wenbiao,wenbiao_pre,maxwenbiaoT,minwenbiaoT,fname)
        cnt += 1
    model.train()

def changeLr(epoch,optimizer,k,saveepoch,D_optimizer):
    [k1, k2, k3, k4] = k
    if (epoch + 1) == 50:  # 11
        optimizer.param_groups[0]['lr'] /= 2
        D_optimizer.param_groups[0]['lr'] /= 2
        k1, k2, k3,k4 = 7,4, 1,0.1
        saveepoch = 49
        print("learning rate change!")
    if (epoch + 1) == 52:  # 11
        saveepoch = 200
        # print("learning rate change!")
    elif (epoch + 1) == 100:  # 11
        optimizer.param_groups[0]['lr'] /= 5
        D_optimizer.param_groups[0]['lr'] /= 2
        k1, k2, k3,k4 = 5, 2, 1,0
        k1 = 2
        print("learning rate change!")
    elif (epoch + 1) == 500:  # 11
        optimizer.param_groups[0]['lr'] /= 2
        D_optimizer.param_groups[0]['lr'] /= 1
        k1, k2, k3,k4 = 4, 1, 1,0
        k1 = 2
        print("learning rate change!")
    elif (epoch + 1) == 1000:  # 11
        optimizer.param_groups[0]['lr'] /= 5
        D_optimizer.param_groups[0]['lr'] /= 10
        k1, k2, k3,k4 = 4, 2, 1,0
        print("learning rate change!")

    return [optimizer,D_optimizer],[k1,k2,k3,k4], saveepoch

def removeCommonElements(a, b):
    for e in a[:]:
        if e in b:
            a.remove(e)
            # b.remove(e)
    return a

def gettraintest(a,b):
    a = removeCommonElements(a,b)
    index = 11*3-1+3
    b.insert(index, a[index])
    return a,b

def writewenbiao(epoches,real,pred,path,diet,lossstr):
    real = real.detach().cpu().numpy() * diet  #b,2
    pred = pred.detach().cpu().numpy() * diet
    diat1 = np.fabs(pred[:,0] - real[:,0])
    diat2 = np.fabs(pred[:, 1] - real[:, 1])
    mean1,mean2 = np.mean(diat1),np.mean(diat2)
    # min1, ,2 = np.min(diat1), np.min(diat2)
    max1, max2 = np.max(diat1), np.max(diat2)
    if not os.path.isdir(path):
        os.makedirs(path)
    loss_path = path + '/error.txt'
    file = open(loss_path, 'a')
    # file.write(
    #     str(epoches[0]) + '\t' + str(epoches[1]) + '\t' + str(mean1) + '\t' + str(mean2) + '\t' + str(max1)+ '\t' + str(max2) + '\n')
    file.write(lossstr + '\t' + str(mean1) + '\t' + str(mean2) + '\t' + str(max1) + '\t' + str(max2) + '\n')
    file.flush()
    file.close()
    return mean1, mean2, max1, max2
    pass

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    # for net in nets:
    #     if net is not None:
    for param in nets.parameters():
        param.requires_grad = requires_grad

def get_Perception(real_feature, fake_feature, lam_p):
    return lam_p * torch.mean(torch.abs(real_feature - fake_feature))

def get_Ploss(real_features, fake_features, lam_ps):
    p = torch.tensor(0).to(device).to(torch.float32)
    for i in range(len(real_features)):
        p += get_Perception(real_features[i],fake_features[i],lam_ps[i])
    return p

use_gpu = True
device = torch.device('cuda:0') if use_gpu else torch.device('cpu')