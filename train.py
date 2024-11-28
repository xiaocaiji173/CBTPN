from model import *
from tools import *
from calssim import *
# import test
# import models.pointnet_part_seg as pointnet_part_seg
# import models.pointnet2_sem_seg_msg as pointnet2_sem_seg_msg
# import models.pointnet2_part_seg_msg as pointnet2_part_seg_msg
# import models.pointnet2_sem_seg_msg as pointnet2_sem_seg_msg
# import perceptualloss
from  discriminator import discriminator
import re

BATCH_SIZE = 4
LEARNING_RATE = 0.0002
EPOCH = 2001
img_size = 256
saveepoch = 50
root = 'E:\\读博项目\\基准工况\\NN\\dataset\\'
file_root = os.path.join(root,'datalist_allused.txt')
img_root = os.path.join(root,'images-11')
canshu_root = ''
with open(file_root, "r") as fp:
    file_names = fp.read().split("\n")[:-1]

test_data = file_names[11*11:12*11]+file_names[13*11:14*11]+file_names[24*11:25*11]+file_names[27*11:28*11]
#31.13 50.21 34.69 68.21 26.6 39.4 30.4 94.6 0.0 5.9 0.0 0.073 0.0 0.035 0.0 0.04
maxwenbiaoT = 70 #68.21
minwenbiaoT = 30 #31.13
maxTair = 40#39.4
maxshidu = 100  #94.6
maxVair = 6  #5.9
maxRall = 0.08 #0.073
maxRs = 0.04 #0.035
maxRz = 0.04 #0.04
diat = maxwenbiaoT-minwenbiaoT
guiyihuacanshu = [minwenbiaoT,diat,maxTair,maxshidu,maxVair,maxRall,maxRs,maxRz]
train_data,test_data = gettraintest(file_names, test_data)
shuffle_ = True
use_gpu = True
device = torch.device('cuda:0') if use_gpu else torch.device('cpu')

trainresult_dir = 'result/' + 'train/'                  #训练结果保存路径
testresult_dir = 'result/' + 'test/'                  #训练结果保存路径

dataset = LoadData2.FacadesDataset(train_data, guiyihua = guiyihuacanshu,imagesize=img_size)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=shuffle_)
trainLoader = loader
dataset_test = LoadData2.FacadesDataset(test_data, guiyihua = guiyihuacanshu,imagesize=img_size,type = 1)
loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False)
testLoader = loader_test

Model = GAN_img_to_img()
D = discriminator()
if not os.path.exists('result/temp'): os.makedirs('result/temp')
files = os.listdir('result/temp')
epochstart = 0
if not files:
    Model.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    epochstart = 0
else:
    match = re.search(r'\d+', files[0])
    epochstart = int(match[0]) + 1
    # Model.weight_load(files[1])
    Model = torch.load('result/temp/' + files[1])
    D = torch.load('result/temp/'+ files[0])
    # D.weight_load(files[0])
Model.cuda()
D.cuda()
# PerceptualLoss = perceptualloss.PerceptualLoss(requires_grad=False).to(device)

CrossEntropyLoss = nn.BCEWithLogitsLoss().to(device)
L1_loss = nn.L1Loss().to(device)
MSELOSS = nn.MSELoss().to(device)
BCE_loss = nn.BCELoss().to(device)
Smoothloss = nn.SmoothL1Loss().to(device)
ssimloss = SSIM().to(device)

optimizer = torch.optim.Adam(Model.parameters(), lr=LEARNING_RATE,betas=(0.9, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE*1, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# Train the model
k1 = 7
k2 = 2
k3 = 0.2
k4 = 0.8
lam_ps = torch.tensor([15,2.5,2.5,1]).to(device)
for epoch in range(epochstart , epochstart  + EPOCH):
# for epoch in range(EPOCH):
    [optimizer, D_optimizer], [k1, k2, k3, k4], saveepoch = changeLr(epoch, optimizer, [k1, k2, k3, k4], saveepoch,
                                                                     D_optimizer)
    avg_loss = 0
    cnt = 0
    Yindex = 0

    for dataA,dataB,fname in trainLoader:
        [inputdata2, img2, wenbiao2], [inputdata, img, wenbiao] = dataA,dataB
        inputdata, img, wenbiao = inputdata.to(torch.float32).to(device), img.to(torch.float32).to(device), wenbiao.to(torch.float32).to(device)  #inputdata 4,6    wenbiao4,2\
        inputdata2, img2, wenbiao2 = inputdata2.to(torch.float32).to(device), img2.to(torch.float32).to(device), wenbiao2.to( torch.float32).to(device)
        inputdataG = torch.cat([inputdata, inputdata2], dim=1)
        inputdataD = [inputdata,wenbiao]

        y_real_ = torch.ones(len(img)).unsqueeze(1)
        y_fake_ = torch.zeros(len(img)).unsqueeze(1)
        y_real_, y_fake_ = y_real_.cuda().clone().detach(), y_fake_.cuda().clone().detach()

        set_requires_grad(D, True)
        set_requires_grad(Model, False)
        D_optimizer.zero_grad()
        img_pre, wenbiao_pre = Model(inputdataG,wenbiao2, img2.permute(0, 3, 1, 2))
        D_result_R, x1,x40,x6,x8 = D(img.permute(0,3,1,2), inputdataD)
        D_result_F, x1_f,x40_f,x6_f,x8_f = D(img_pre, inputdataD)
        Pimgloss = get_Ploss([x1, x40, x6, x8], [x1_f, x40_f, x6_f, x8_f], lam_ps) * 5
        D_PLOSS = torch.max((torch.tensor(2).float().to(device) - 1 * Pimgloss), torch.tensor(0).float().to(device))
        tempd = BCE_loss(D_result_R, y_real_) + BCE_loss(D_result_F, y_fake_)
        D_loss = tempd + D_PLOSS
        D_loss.backward(retain_graph=True)
        D_optimizer.step()

        set_requires_grad(Model, True)
        for i in range(k1):
        #Train G
            set_requires_grad(D, False)
            optimizer.zero_grad()
            img_pre, wenbiao_pre = Model(inputdataG, wenbiao2,img2.permute(0, 3, 1, 2))
            D_result_R, x1, x40, x6, x8 = D(img.permute(0, 3, 1, 2), inputdataD)
            D_result_F, x1_f, x40_f, x6_f, x8_f = D(img_pre, inputdataD)
            Pimgloss = get_Ploss([x1, x40, x6, x8], [x1_f, x40_f, x6_f, x8_f], lam_ps) * k2
            G_loss0 = BCE_loss(D_result_F, y_real_)
            L1loss1 = L1_loss(wenbiao, wenbiao_pre) * 20
            temp1 = L1_loss(img, img_pre.permute(0, 2, 3, 1))
            themloss = L1_loss(img[:, :, int(img.shape[2] * 0.125):int(img.shape[2] * 0.875),int(img.shape[3] * 0.125):int(img.shape[3] * 0.875)],
                        img_pre.permute(0, 2, 3, 1)[:,:, int(img.shape[2] * 0.125):int(img.shape[2] * 0.875),int(img.shape[3] * 0.125):int(img.shape[3] * 0.875)])

            pixloss = (25 * k3 + k4) * temp1
            ssim_loss = 1 - ssimloss(img.permute(0, 3, 1, 2), img_pre)
            G_loss = G_loss0 + Pimgloss + pixloss + L1loss1 + 0 * ssim_loss + 5 * themloss
            G_loss.backward(retain_graph=True)
            optimizer.step()

        inputdataG2 = torch.cat([inputdata2, inputdata], dim=1)
        inputdataD = [inputdata2,wenbiao2]
        imgtoimg(D, Model, D_optimizer, inputdataG2, img2, img, inputdataD, BCE_loss, y_real_, y_fake_, L1_loss, wenbiao,
                 k1, k2, k3, k4, ssimloss, lam_ps, optimizer)
        img_pre = img_pre.permute(0, 2, 3, 1)

        cnt += 1
        lossstr = "E: %d %d Dloss: %f,Gloss: %f, pixloss: %f" % (epoch, cnt,tempd.data, G_loss0.data, temp1.data)
        mean1, mean2, max1, max2 = writewenbiao([epoch,cnt], wenbiao, wenbiao_pre, trainresult_dir,maxwenbiaoT-minwenbiaoT,lossstr)

        print("[E: %d_%d] Dloss: %f,Gloss: %f, pixloss: %f,Ploss: %f,L1loss: %f,,ssim: %f 平均温差: %f, %f, 最大温差: %f, %f"
              % (epoch, cnt,D_loss.data, G_loss0.data, G_loss0.data,Pimgloss.data, L1loss1.data,ssim_loss.data, mean1, mean2, max1, max2))

        if ((epoch+0) % saveepoch) == 0:
            show_res_(epoch, cnt, img,img_pre,trainresult_dir,wenbiao,wenbiao_pre,maxwenbiaoT,minwenbiaoT,fname)
            # show_testres(Model, testLoader, epoch, testresult_dir, maxwenbiaoT, minwenbiaoT)

        cnt += 0

    [os.remove(os.path.join('result\\temp\\', f)) for f in os.listdir('result\\temp\\') if os.path.isfile(os.path.join('result\\temp\\', f))]
    torch.save(Model, 'result\\temp\\G_%d.pkl' % (epoch))
    torch.save(D, 'result\\temp\\D_%d.pkl' % (epoch))
    # import time
    # time.sleep(120)

    if ((epoch+0) % saveepoch) == 0:
        show_testres(Model, testLoader, epoch, testresult_dir, maxwenbiaoT, minwenbiaoT)
        torch.save(Model, 'result\\CNN_%d.pkl' % (epoch + 1))
        # pass


