from bilateral_solver import bilateral_solver_output
from features_extract import deep_features
from torch_geometric.data import Data
from extractor import ViTExtractor
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import util
import os
from PIL import Image
from google.colab.patches import cv2_imshow
import cv2
import matplotlib.pyplot as plt
from numpy import asarray
import tifffile as tiff

def GNN_seg( modelName,mode, cut, alpha, epoch, K, pretrained_weights, images, masks, out_dir, save, cc, bs, log_bin, res, facet, layer,
            stride, device, iou,cts, activ):
    """
    Segment entire dataset; Get bounding box (k==2 only) or segmentation maps
    bounding boxes will be in the following format: class, confidence, left, top , right, bottom
    (class and confidence not in use for now, set as '1')
    @param cut: chosen clustering functional: NCut==1, CC==0
    @param epoch: Number of epochs for every step in image
    @param K: Number of segments to search in each image
    @param pretrained_weights: Weights of pretrained images
    @param dir: Directory for chosen dataset
    @param out_dir: Output directory to save results
    @param cc: If k==2 chose the biggest component, and discard the rest (only available for k==2)
    @param b_box: If true will output bounding box (for k==2 only), else segmentation map
    @param log_bin: Apply log binning to the descriptors (correspond to smother image)
    @param device: Device to use ('cuda'/'cpu')
    """
    ##########################################################################################
    # Dino model init
    ##########################################################################################
    model_type = 'dino_vits8'
    #print('edited4')
    extractor = ViTExtractor(model_type, stride, model_dir=pretrained_weights, device=device)
    # VIT small feature dimension, with or without log bin
    imgcount =1
    if not log_bin:
        feats_dim = 384
    else:
        feats_dim = 6528

    if "8" in model_type:
      patch = 8
    else:
      patch = 16

    # if two stage make first stage foreground detection with k == 2
    if mode == 1 or mode == 2:
        foreground_k = K
        K = 2

    ##########################################################################################
    # GNN model init
    ##########################################################################################
    # import cutting gnn model if cut == 0 NCut else CC
    if cut == 0: 
        from gnn_pool import GNNpool 
    else: 
        from gnn_pool_cc import GNNpool

    # model = GNNpool(feats_dim, 64, 32, K, device, activ).to(device)
    # model = GNNpool(feats_dim, 96, 32, K, device, activ).to(device)
    model = GNNpool(feats_dim, 128, 32, K, device, activ).to(device)
    torch.save(model.state_dict(), 'model.pt')
    model.train()
    if mode == 1 or mode == 2:
        model2 = GNNpool(feats_dim, 64, 32, foreground_k, device).to(device)
        torch.save(model2.state_dict(), 'model2.pt')
        model2.train()
    if mode == 2:
        model3 = GNNpool(feats_dim, 64, 32, 2, device).to(device)
        torch.save(model3.state_dict(), 'model3.pt')
        model3.train()


    total_iou = 0
    ##########################################################################################
    # Iterate over files in input directory and apply GNN segmentation
    ##########################################################################################
    dir_name = f'model_GAT_03_EDGEW_epochs_{epoch[0]}_gat2_stride_{stride}_patch_{patch}_{K}_{activ}_silu_cts_mode{mode}_cut_{cut}_alpha_{alpha}_bs_{bs}'.replace('.','_')
    dirpath = os.path.join(out_dir,dir_name)
    if not os.path.isdir(dirpath):
      os.makedirs(dirpath)

    if bs:
      dir_name1 = dir_name+'without_BS'
      dirpath1 = os.path.join(out_dir,dir_name1)
      if not os.path.isdir(dirpath1):
        os.makedirs(dirpath1)

    kkk = 0    
    for image, true_mask in tqdm(zip(images, masks)):
        kkk += 1
        filename = image.split('/')[-1].split('.')[0]
        filetype = image.split('/')[-1].split('.')[-1]
        
        if 'tif' in filetype:
          #from libtiff import TIFF
          # print(image)
          image = asarray(tiff.imread(image))
          #image = asarray(TIFF.open(image).read_image())
          # true_mask = asarray(TIFF.open(true_mask))
          # true_mask = asarray(Image.fromarray(true_mask).convert('L'))
        else:
          image = asarray(Image.open(image))
        true_mask = asarray(Image.open(true_mask).convert('L'))
          
        image, true_mask = cts(image, true_mask)          

        # print(true_mask.shape)
        # true_mask = tiff.imread(true_mask)
        # true_mask = (true_mask == 255).astype(np.uint8)
        # cv2_imshow(image)
        # cv2_imshow(true_mask)
        # image = tiff.imread(image)

        
        # image, true_mask = cts(image, true_mask)
        true_mask = np.where(true_mask ==255, 1, 0).astype(np.uint8)

        if 'tif' in filetype:
          image_tensor, image = util.load_data_img1(image, res)
        else:
          image_tensor, image = util.load_data_img(image, res)
        #print('image_tensor',image_tensor.shape)
        # print(f'image tensor shape={image_tensor.shape},image shape={image.shape}')
        kkk += 1
        # Extract deep features, from the transformer and create an adj matrix
        F = deep_features(image_tensor, extractor, layer, facet, bin=log_bin, device=device)
        # print(f'feature shape is {F.shape}')
        W = util.create_adj(F, cut, alpha)
        # print(f'W_min={W.min()}, W_max={W.max()}')
        # print(f'W_shape={W.shape}')
        # print(f'W nonzeros={np.count_nonzero(W)}')
        # ## Create KNN matting ###################################
        # filters = torch.einsum('ab,cd->abcd',torch.eye(3),torch.ones(patch,patch))
        # # filters = torch.ones(3, 3, patch, patch)
        # # inputs = torch.randn(1, 4, 5, 5)
        # # print(image_tensor.shape)
        # resized_img = torch.nn.functional.conv2d(image_tensor, filters, stride=stride)
        # resized_img = torch.permute(resized_img, (0, 2, 3, 1))
        # # print(resized_img.shape)
        # resized_img = np.squeeze(resized_img)
        # W_knn = util.knn_affinity(resized_img).toarray()
        # W_knn = np.where(W_knn > 0, 1, 0).astype(np.float32)
        # # print(f'Wknn_shape={W_knn.shape}')
        # # print(f'Wknn_min={W_knn.min()}, Wknn_max={W_knn.max()}')
        # # print(f'Wknn nonzeros={np.count_nonzero(W_knn)}')
        # W = W + 0.4*W_knn
        ###########################################################
        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = util.load_data(W, F)
        data = Data(node_feats, edge_index, edge_weight).to(device)

        # re-init weights and optimizer for every image
        model.load_state_dict(torch.load(modelName, map_location=torch.device(device)))
        opt = optim.AdamW(model.parameters(), lr=0.0001)

        ##########################################################################################
        # GNN pass
        ##########################################################################################
        for _ in range(epoch[0]):
            opt.zero_grad()
            A, S = model(data, torch.from_numpy(W).to(device))
            loss = model.loss(A, S)
            loss.backward()
            opt.step()
            #print(loss.detach())

        # polled matrix (after softmax, before argmax)
        S = S.detach().cpu()
        S = torch.argmax(S, dim=-1)
        # print('hi22',S.unique())
        ##########################################################################################
        # Post-processing Connected Component/bilateral solver
        ##########################################################################################
        #print('S',S.shape)
        mask0, S = util.graph_to_mask(S, cc, stride, image_tensor, image)
        #print('mask0',mask0.shape)
        # print(mask0.shape)
        # print(image_tensor.shape)
        # apply bilateral solver
        
        if bs:
            util.save_or_show([image, np.where(mask0==True, 255,0).astype(np.uint8), util.apply_seg_map(image, mask0, 0.5),true_mask*255], filename, dirpath1 ,save)
            mask0 = bilateral_solver_output(image, mask0,sigma_spatial=11, sigma_luma=5, sigma_chroma=5)[1]
        mask0 = np.where(mask0==True, 1,0).astype(np.uint8)
        # mask0_resized = cv2.resize(mask0, (true_mask.shape[1], true_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        # print("Hi")
        # print(mask0)

        try:
          cur_iou = iou(mask0, true_mask)
          cur_iou1 = iou(1-mask0, true_mask)
          if cur_iou < cur_iou1:
            cur_iou = cur_iou1
            mask0 = 1 - mask0
          total_iou += cur_iou 

        # print(mask0.shape, true_mask.shape)
        # plt.imshow(mask0)
        # plt.show()
        # plt.imshow(true_mask)
        # plt.show()
            # print(len(mask0.ravel()), len(true_mask.ravel()))
            
#             for x in mask0.ravel():
#                 if not x == 0:
#                     print(x, end=" ")
            
            # break
          print(total_iou/imgcount, imgcount,cur_iou)
          imgcount  = imgcount +1
        except:
        #   print("hi")
          pass
        util.save_or_show([image, mask0*255, util.apply_seg_map(image, mask0, 0.5),true_mask*255], filename, dirpath ,save)
        
    return (total_iou / imgcount)*100

if __name__ == '__main__':
    ################################################################################
    # Mode
    ################################################################################
    # mode == 0 Single stage segmentation
    # mode == 1 Two stage segmentation for foreground
    # mode == 2 Two stage segmentation on background and foreground
    mode = 0
    ################################################################################
    # Clustering function
    ################################################################################
    # NCut == 0
    # CC == 1
    # alpha = k-sensetivity paremeter
    cut = 0
    alpha = 3
    ################################################################################
    # GNN parameters
    ################################################################################
    # Numbers of epochs per stage [mode0,mode1,mode2]
    epochs = [10, 100, 10]
    # Number of steps per image
    step = 1
    # Number of clusters
    K = 2
    ################################################################################
    # Processing parameters
    ################################################################################
    # Show only largest component in segmentation map (for k == 2)
    cc = False
    # apply bilateral solver
    bs = False
    # Apply log binning to extracted descriptors (correspond to smoother segmentation maps)
    log_bin = False
    ################################################################################
    # Descriptors extraction parameters
    ################################################################################
    # Directory to pretrained Dino
    pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
    # Resolution for dino input, higher res != better performance as Dino was trained on (224,224) size images
    res = (280, 280)
    # stride for descriptor extraction
    stride = 8
    # facet fo descriptor extraction (key/query/value)
    facet = 'key'
    # layer to extract descriptors from
    layer = 11
    ################################################################################
    # Data parameters
    ################################################################################
    # Directory of image to segment
    in_dir = './images/single/'
    out_dir = './results/'
    save = False
    ################################################################################
    # Check for mistakes in given arguments
    assert not(K != 2 and cc), 'largest connected component only available for k == 2'

    # if CC set maximum number of clusters
    if cut == 1:
        K = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If Directory doesn't exist than download
    if not os.path.exists(pretrained_weights):
        url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
        util.download_url(url, pretrained_weights)

    GNN_seg(mode, cut, alpha, epochs, K, pretrained_weights, in_dir, out_dir, save, cc, bs, log_bin, res, facet, layer, stride,
            device)
