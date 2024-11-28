import numpy as np
import re
import os
import math
import torch
# import tecplot
# from tecplot.constant import *
# tecplot.session.connect(port=7600)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx
# import seaborn as sns
# import open3d as o3d

def writeloss(path,epoches,loss, avg_loss,Closs,Lloss=0,loss3=0):
    loss_path=path+'\\loss.txt'
    file = open(loss_path,'a')
    # file.write(str(loss.data.cpu().numpy())+'\t'+str(loss2.data.cpu().numpy())+'\n')
    file.write(str(loss) + '\t' + str(avg_loss)+ '\t' + str(Closs)+ '\t' + str(Lloss) + '\n')
    file.flush()
    file.close
    # data_path = path + '\\data.txt'
    # file2 = open(data_path, 'a')
    # # file2.write(str(epoches)+'\t'+str(loss.data.cpu().numpy())+'\t'+str(loss2.data.cpu().numpy())+'\n')
    # file2.write(str(epoches) + '\t' + str(loss) + '\t' + str(avg_loss)+ '\t' + str(Closs)+ '\t' + str(Lloss)+ '\t' + str(loss3) + '\n')
    # file2.flush()
    # file2.close

def writeerror(path,epoches,T_tesult,T_real,diat):
    if not os.path.isdir(path):
        os.makedirs(path)
    # T_tesult = np.transpose(T_tesult.cpu().detach().numpy(), (0,  1))
    # T_real = np.transpose(T_real.cpu().detach().numpy(), (0,  1))
    T_tesult = T_tesult.cpu().detach().numpy()
    T_real = T_real.cpu().detach().numpy()
    error = T_tesult-T_real
    # error = error0*42/1.0
    error2 = np.fabs(error)
    # maxe1 = np.max(error)
    maxe = np.max(error2)
    # mine = np.min(error2)
    # avee1 = np.mean(error)
    avee = np.mean(error2)
    # mede1 = np.median(error)
    # mede2 = np.median(error2)
    # stde1 = np.std(error)
    stdee = np.std(error2)

    loss_path = path + '/error.txt'
    file = open(loss_path,'a')
    file.write(str(epoches[0])+'\t'+ str(epoches[1])+'\t'+ str(maxe)+'\t'+ str(avee)+'\t'+str(stdee)+'\n')
    file.flush()
    file.close
    return maxe,avee

def writeerror_2(path,epoches,T_result,T_real,diat,lenv):
    if not os.path.isdir(path):
        os.makedirs(path)
    # T_result = [T_result[i, :lenv[i]].cpu().detach().numpy() for i in range(len(lenv))]
    # T_real = [T_real[i, :lenv[i]].unsqueeze(0).cpu().detach().numpy() for i in range(len(lenv))]
    # T_result0, T_real0 =T_result[0], T_real[0]
    # for i in range(len(lenv)-1):
    #     T_result0 = np.append(T_result0,T_result[i+1])
    #     T_real0 = np.append(T_real0,T_real[i+1])

    T_result = T_result.cpu().detach().numpy()
    T_real = T_real.cpu().detach().numpy()
    error = T_result[0, :lenv[0]] - T_real[0, :lenv[0]]
    for i in range(len(lenv) - 1):
        error = np.append(error,T_result[i+1, :lenv[i+1]] - T_real[i+1, :lenv[i+1]])
    # T_result0, T_real0 = T_result[0], T_real[0]
    # T_tesult = T_tesult.cpu().detach().numpy()
    # T_real = T_real.cpu().detach().numpy()
    # error = T_result0-T_real0
    error2 = np.fabs(error)
    maxe = np.max(error2)
    avee = np.mean(error2)

    stdee = np.std(error2)

    loss_path = path + '/error.txt'
    file = open(loss_path,'a')
    file.write(str(epoches[0])+'\t'+ str(epoches[1])+'\t'+ str(maxe)+'\t'+ str(avee)+'\t'+str(stdee)+'\n')
    file.flush()
    file.close
    return maxe,avee

def writedata(T,xyzline,conectline,titlelint,Nodes,Elements,filename):
    lenx = len(xyzline)
    # allT = T.squeeze(0)
    allT = T.flatten()
    # lenT = len(allT) // lenx
    # if not os.path.exists(filename):
    #     os.makedirs(filename)
    f = open(filename, 'w') #清空文件内容再写
    # f.write('TITLE     = "fluent18.0.0  build-id: 10373"'+'\n')
    # f.write('VARIABLES = "X"'+'\n'+'"Y"'+'\n'+'"Z"'+'\n'+'"Temperature"'+'\n')
    # f.write('DATASETAUXDATA Common.DensityVar="19"'+'\n')
    # f.write('DATASETAUXDATA Common.PressureVar="4"'+'\n')
    # f.write('DATASETAUXDATA Common.TemperatureVar="14"'+'\n')
    # f.write('DATASETAUXDATA Common.UVar="5"'+'\n')
    # f.write('DATASETAUXDATA Common.VectorVarsAreVelocity="TRUE"'+'\n')
    # f.write('DATASETAUXDATA Common.ViscosityVar="21"'+'\n')
    # f.write('DATASETAUXDATA Common.VVar="7"'+'\n')
    # f.write('DATASETAUXDATA Common.WVar="9"'+'\n')
    # strname = ['ZONE T = "qian"','ZONE T = "hou"','ZONE T = "zuo"','ZONE T = "you"','ZONE T = "shang"','ZONE T = "xia"']
    # str0 = 'Nodes = 1681, Elements = 1600, ZONETYPE = FEQuadrilateral'+ '\n'+'DATAPACKING = BLOCK'
    # str1 = 'AUXDATA Common.BoundaryCondition = "Wall"'+ '\n'+ 'AUXDATA Common.IsBoundaryZone = "TRUE"' + '\n'+   'DT = (SINGLE SINGLE SINGLE SINGLE)'
    NoT = 0   #计数
    for i in range(lenx):
        # print(i,Elements[i])
    # for i in range(1):
        lenT = int(Elements[i])
        if (i!=0):
            NoT = NoT + int(Elements[i-1])
        for j in range(len(titlelint[i])):
            f.write(titlelint[i][j])
        for j in range(len(xyzline[i])):
            f.write(xyzline[i][j])
        for j in range(lenT//5):
            # print(j)
            if (i == 1):
                aa = 0
            str2 = str(allT[NoT+5*j+0])+' '+str(allT[NoT+5*j+1])+' '+str(allT[NoT+5*j+2])+' '+str(allT[NoT+5*j+3])+' '+str(allT[NoT+5*j+4])+'\n'
            f.write(str2)
        for j in range(lenT%5):
            f.write(str(allT[NoT+(lenT//5)*5+j])+ '\n')
        if (lenT%5 != 0):
            f.write('\n')
        for j in range(len(conectline[i])):
            f.write(conectline[i][j])
    f.close()

def writedata2(T,xyzline,conectline,filename):
    lenx = len(xyzline)
    # allT = T.squeeze(0)
    allT = T.flatten()
    lenT = len(allT) // lenx
    # if not os.path.exists(filename):
    #     os.makedirs(filename)
    f = open(filename, 'w') #清空文件内容再写
    f.write('TITLE     = "fluent18.0.0  build-id: 10373"'+'\n')
    f.write('VARIABLES = "X"'+'\n'+'"Y"'+'\n'+'"Z"'+'\n'+'"Temperature"'+'\n')
    f.write('DATASETAUXDATA Common.DensityVar="19"'+'\n')
    f.write('DATASETAUXDATA Common.PressureVar="4"'+'\n')
    f.write('DATASETAUXDATA Common.TemperatureVar="14"'+'\n')
    f.write('DATASETAUXDATA Common.UVar="5"'+'\n')
    f.write('DATASETAUXDATA Common.VectorVarsAreVelocity="TRUE"'+'\n')
    f.write('DATASETAUXDATA Common.ViscosityVar="21"'+'\n')
    f.write('DATASETAUXDATA Common.VVar="7"'+'\n')
    f.write('DATASETAUXDATA Common.WVar="9"'+'\n')
    strname = ['ZONE T = "qian"','ZONE T = "hou"','ZONE T = "zuo"','ZONE T = "you"','ZONE T = "shang"','ZONE T = "xia"']
    str0 = 'Nodes = 1681, Elements = 1600, ZONETYPE = FEQuadrilateral'+ '\n'+'DATAPACKING = BLOCK'
    str1 = 'AUXDATA Common.BoundaryCondition = "Wall"'+ '\n'+ 'AUXDATA Common.IsBoundaryZone = "TRUE"' + '\n'+   'DT = (SINGLE SINGLE SINGLE SINGLE)'
    for i in range(lenx):
    # for i in range(1):
        f.write(strname[i]+'\n')
        f.write('STRANDID = '+ str(i+3) +', SOLUTIONTIME = 0'+'\n')
        f.write(str0+'\n')
        # if i == 0:
        f.write('VARLOCATION = ([4]=CELLCENTERED)'+'\n')
        f.write(str1+'\n')

        for j in range(len(xyzline[i])):
            f.write(xyzline[i][j])
        for j in range(lenT//5):
            str2 = str(allT[i*lenT+5*j+0])+' '+str(allT[i*lenT+5*j+1])+' '+str(allT[i*lenT+5*j+2])+' '+str(allT[i*lenT+5*j+3])+' '+str(allT[i*lenT+5*j+4])+'\n'
            f.write(str2)
        for j in range(lenT%5):
            f.write(str(allT[i*lenT+(lenT//5)*5+j]))
        if (lenT%5 != 0):
            f.write('\n')
        for j in range(len(conectline[i])):
            f.write(conectline[i][j])
    f.close()

def write_traindata(result_dir,epoch,output,files,Yindex,xyzline, conectline,titlelint,Nodes2,Elements2):
    # if not os.path.isdir(result_dir):
    #     os.makedirs(result_dir)
    T = np.transpose(np.array(output.cpu().data), (0, 1))
    T = (T - 0.0) / 1.0
    T = T * 42 + 350
    for i in range(len(output)):
        str_list = list(files[Yindex])
        b = '_' + str(i % 4)
        str_list.insert(8, b)
        name = ''.join(str_list)
        filename = result_dir + '\\' + str(epoch) + '\\' + name
        if not os.path.exists(result_dir + '\\' + str(epoch)):
            os.makedirs(result_dir + '\\' + str(epoch))
        writedata(T[i], xyzline, conectline,titlelint,Nodes2,Elements2, filename)

def write_testdata(result_dir,epoch,output,files,Yindex,xyzline, conectline,i):
    # if not os.path.isdir(result_dir):
    #     os.makedirs(result_dir)
    T = np.transpose(np.array(output.cpu().data), (0, 2, 3, 1))
    T = (T - 0.0) / 1.0
    T = T * 42 + 350

    str_list = list(files[Yindex])
    b = '_' + str(i % 4)
    str_list.insert(8, b)
    name = ''.join(str_list)
    filename = result_dir + '\\' + str(epoch) + '\\' + name
    if not os.path.exists(result_dir + '\\' + str(epoch)):
        os.makedirs(result_dir + '\\' + str(epoch))
    writedata(T[i], xyzline, conectline, filename)

def write_result(G,input,real,num_epoch=0,path='',data_dir2='',xyzline=0,conectline=0):
    G.eval()
    # data_dir2 = 'datasets\\facades\\test\\b'
    result_dir = 'result/' + path
    files = os.listdir(data_dir2)
    # imgG.append(x_imgs0[0])
    # imgG.append(x_imgs0[1])
    for i in range(len(input)):
        Yindex = i//4
        output = G(input[i])
        # gt_img = np.transpose(real[i], (0, 3, 1, 2)).clone().detach().float().cuda()
        writeerror(result_dir, num_epoch, output, real[i])

        output = np.transpose(np.array(output.cpu().data), (0, 2, 3, 1))
        # gt_img = np.array(real[i].cpu().data
        # gt_img = np.transpose(np.array(output.cpu().data), (0, 2, 3, 1))
        T = (T - 0.0) / 1.0
        T = T * 42 + 350
        # gt_img = gt_img * 42 + 350

        str_list = list(files[Yindex])
        b = '_' + str(i%4)
        str_list.insert(8, b)
        name = ''.join(str_list)
        filename = result_dir + '\\'+ str(num_epoch)+ '\\'+ name
        if not os.path.exists(result_dir + '\\'+ str(num_epoch)):
            os.makedirs(result_dir + '\\'+ str(num_epoch))
        writedata(T,xyzline,conectline,filename)

    G.train()


def show_testresult(netG,epoch,test_inputs,path = 'test',data_dir2='datasets\\facades\\test\\b',xyzline = 0, conectline = 0,device =torch.device('cuda:0')):
    # test.show_results2(self.netG,self.device,epoch,self.real_A,imgtest_path,imgsize)
    if not os.path.isdir(data_dir2):
        os.makedirs(data_dir2)
    testa,testb,testb2 = [],[],[]
    for i in range(len(test_inputs)):
        # AtoB = self.opt['direction'] == 'AtoB'
        test_input = test_inputs[i]
        testa.append(np.transpose(test_input['A'], (0, 3, 1, 2)).clone().detach().float().to(device)/ 255)
        # testb.append(np.transpose(test_input['B' if AtoB el se 'A'], (0, 3, 1, 2)).clone().detach().float().to(self.device))
        testb2.append(np.transpose(test_input['B'], (0, 3, 1, 2)).clone().detach().float().to(device))  # 6,40,40
        # testb2.append(test_input['B'])   #40 40 6
    write_result(netG, testa, testb2, epoch,path,data_dir2,xyzline,conectline)

def setview(plot,width=10,alpha=120,theta = -120,psi=60,position = (57,27,30)):
    plot.view.width = width
    plot.view.alpha = alpha
    plot.view.theta = theta
    plot.view.psi = psi
    plot.view.position = position

def drawtecpic_xyzT(xyznp,T,faceNo,savepath,Tmin=274, Tmax=344,width=1.9, alpha=70, theta=-130, psi=100, position=(6.6, 5.6, -1.4),
               variableindex = 3,showcontour = True,legendshow = False,axisshow = False):
    # Nodes, Elements, T, xyz, faceNo = readtecplot.readTemperature_node(infile)
    # xyznp = (np.array(xyz)-center1)/scale1
    # xyznp = xyznp[:, bbb] * aaa
    # faceNo = np.array(faceNo)
    tecplot.new_layout()
    frame = tecplot.active_frame()
    if not frame.has_dataset:
        dataset = frame.create_dataset('Dataset', ['x', 'y', 'z', 'T'])
    # a.add_fe_zone(ZoneType.FETriangle, 'Interpolated', len(new_verts0), len(new_face_idx1))
    newzone = dataset.add_fe_zone(ZoneType.FETriangle, 'Interpolated', len(xyznp), len(faceNo))
    newzone.values(0)[:] = xyznp[:, 0].ravel()
    newzone.values(1)[:] = xyznp[:, 1].ravel()
    newzone.values(2)[:] = xyznp[:, 2].ravel()
    newzone.values(3)[:] = np.array(T).ravel()
    # newzone.nodemap[:] = list(np.array(new_face_idx1))
    newzone.nodemap[:] = faceNo
    # ds = tecplot.data.load_tecplot(infile, read_data_option=ReadDataOption.Replace)
    plot = tecplot.active_frame().plot(PlotType.Cartesian3D)
    plot.activate()

    tecplot.active_frame().plot().contour(0).colormap_name = 'Small Rainbow'

    tecplot.active_frame().plot().show_contour = showcontour
    tecplot.active_frame().plot().contour(0).variable_index = variableindex
    tecplot.active_frame().plot().contour(0).levels.reset_levels(np.linspace(Tmin, Tmax, 100))

    legend = tecplot.active_frame().plot().contour(0).legend
    legend.show = legendshow
    plot = tecplot.active_frame().plot(PlotType.Cartesian3D)
    setview(plot, width, alpha, theta, psi, position)
    plot.activate()
    plot.axes.orientation_axis.show = axisshow
    tecplot.export.save_png(savepath, 800, supersample=3)

def drawimg_plt2(xyz,T,T2,T_min,T_max,savepath):
    # temperatrue_fake = np.loadtxt('test_results_visions\\3\\fake-2000.txt')
    # temperatrue_real = np.loadtxt('test_results_visions\\3\\real-2000.txt')
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # x = temperatrue_fake[:, 0]
    # y = temperatrue_fake[:, 1]
    # z = temperatrue_fake[:, 2]
    # t_fake = temperatrue_fake[:, 3]
    # t_real = temperatrue_real[:, 3]

    #colorslist = ['blue','green','red']
    colorslist = sns.color_palette("rainbow", 32)
    cmaps = colors.LinearSegmentedColormap.from_list('mylist', colorslist, N = 800)

    # fig = plt.figure()
    fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax = plt.subplot(1, 2, 1, projection='3d')
    # ax.view_init(30, 70)                        #分别是上下旋转和左右旋转
    ax.view_init(140, -60)  # 分别是上下旋转和左右旋转
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.7, 3.2, 1]))      #前面三个参数控制拉伸
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.6, 0.35, 1.6, 1]))  # 前面三个参数控制拉伸
    cNorm = colors.Normalize(vmin=T_min, vmax=T_max)
    ax.scatter(x, y, z, c = T, cmap=cmaps,norm=cNorm)

    plt.axis('off')
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmaps)
    scalarMap.set_array(T)
    fig.colorbar(scalarMap)

    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax2 = plt.subplot(1, 2, 2, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')    #分别是上下旋转和左右旋转
    ax2.view_init(140, -60)  # 分别是上下旋转和左右旋转
    # # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.7, 3.2, 1]))      #前面三个参数控制拉伸
    ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax2), np.diag([0.6, 0.35, 1.6, 1]))  # 前面三个参数控制拉伸
    cNorm = colors.Normalize(vmin=T_min, vmax=T_max)
    ax2.scatter(x, y, z, c=T2, cmap=cmaps, norm=cNorm)

    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmaps)
    scalarMap.set_array(T)
    fig.colorbar(scalarMap)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    plt.axis('off')
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # plt.show()
    fig.savefig(savepath)

def drawimg_plt1(xyz,T,T_min,T_max,savepath):
    # temperatrue_fake = np.loadtxt('test_results_visions\\3\\fake-2000.txt')
    # temperatrue_real = np.loadtxt('test_results_visions\\3\\real-2000.txt')
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # x = temperatrue_fake[:, 0]
    # y = temperatrue_fake[:, 1]
    # z = temperatrue_fake[:, 2]
    # t_fake = temperatrue_fake[:, 3]
    # t_real = temperatrue_real[:, 3]

    #colorslist = ['blue','green','red']
    colorslist = sns.color_palette("rainbow", 32)
    cmaps = colors.LinearSegmentedColormap.from_list('mylist', colorslist, N = 800)

    # fig = plt.figure()
    fig = plt.figure(figsize=(12, 6))
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.view_init(30, 70)                        #分别是上下旋转和左右旋转
    ax.view_init(140, -60)  # 分别是上下旋转和左右旋转
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.7, 3.2, 1]))      #前面三个参数控制拉伸
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.6, 0.35, 1.6, 1]))  # 前面三个参数控制拉伸
    cNorm = colors.Normalize(vmin=T_min, vmax=T_max)
    ax.scatter(x, y, z, c = T, cmap=cmaps,norm=cNorm)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmaps)
    scalarMap.set_array(T)
    fig.colorbar(scalarMap)

    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # plt.show()
    fig.savefig(savepath)
    plt.close("all")
    plt.clf()
