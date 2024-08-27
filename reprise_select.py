import sys
sys.path.append('core')
import cv2
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from igev_stereo import IGEVStereo
import os
import argparse
from utils.utils import InputPadder


from matplotlib import pyplot as plt
import pyvista as pv

torch.backends.cudnn.benchmark = True
half_precision = True


DEVICE = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Iterative Geometry Encoding Volume for Stereo Matching and Multi-View Stereo (IGEV-Stereo)')

#parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/kitti/kitti15.pth')

parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./models/Selective-IGEV/middlebury/middlebury_finetune.pth')

parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

# parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/data/KITTI_raw/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/*.png")
# parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/data/KITTI_raw/2011_09_26/2011_09_26_drive_0005_sync/image_03/data/*.png")

# parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./test_video/left/*.JPG")
# parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./test_video/right/*.JPG")


parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./test_pipe/left/*.JPG")
parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./test_pipe/right/*.JPG")

# parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./test_video_light/left/*.JPG")
# parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./test_video_light/right/*.JPG")

parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

args = parser.parse_args()
model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
model.load_state_dict(torch.load(args.restore_ckpt))
model = model.module
model.to(DEVICE)
model.eval()

left_images = sorted(glob.glob(args.left_imgs, recursive=True))
right_images = sorted(glob.glob(args.right_imgs, recursive=True))
print(f"Found {len(left_images)} images.")


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def apply_colormap(depth, colormap=cv2.COLORMAP_JET):
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(np.uint8(depth_normalized), colormap)
    return depth_colored


def depthmap_to_grayscale(depth):
    # Normaliser la carte de profondeur pour qu'elle soit entre 0 et 255
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    # Convertir en niveau de gris
    depth_grayscale = np.uint8(depth_normalized)
    return depth_grayscale

def display_image(image, title='Image'):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def add_colorbar(image, depth, colormap=cv2.COLORMAP_JET):
    h, w = image.shape[:2]
    colorbar_width = 50
    colorbar = np.zeros((h, colorbar_width, 3), dtype=np.uint8)
    
    for i in range(h):
        value = int((h - i - 1) * 255 / h)
        colorbar[i, :] = cv2.applyColorMap(np.uint8([[value]]), colormap)[0, 0]

    image_with_colorbar = np.hstack((image, colorbar))
    
    min_val = np.min(depth)
    max_val = np.max(depth)
    
    for i, val in enumerate(np.linspace(min_val, max_val, num=10)):
        y = int(h - (i * h / 9))
        cv2.putText(image_with_colorbar, f'{val:.2f}', (w + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image_with_colorbar

def add_text(image, text, position, font_scale=1, thickness=2):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def create_depth_map_image(depth):
    depth_colored = apply_colormap(depth / 1000.0)
    depth_with_colorbar = add_colorbar(depth_colored, depth / 1000.0)
    
    #add_text(depth_with_colorbar, 'Depth Map', (10, 30), font_scale=1, thickness=2)
    # add_text(depth_with_colorbar, 'X', (depth_with_colorbar.shape[1]//2, depth_with_colorbar.shape[0] - 10), font_scale=1, thickness=2)
    # add_text(depth_with_colorbar, 'Y', (10, depth_with_colorbar.shape[0]//2), font_scale=1, thickness=2)
    
    return depth_with_colorbar




def create_point_cloud_image(point_cloud, filename='point_cloud.png', points=[]):
    plotter = pv.Plotter(off_screen=True)
    plotter.add_points(point_cloud, scalars='depth', point_size=5, render_points_as_spheres=True, cmap='jet')

# Calculer le barycentre du nuage de points
    barycenter = np.mean(points, axis=0)

# Définir les valeurs de position de la caméra
    center_x, center_y, center_z = barycenter  # Le point vers lequel la caméra est orientée (centre de la scène)
    print(np.min(points[:,2]))
    x, y, z = center_x, center_y + np.min(points[:,2])*5, center_z  # Les coordonnées du barycentre sont utilisées comme position de la caméra
    up_x, up_y, up_z = 0, 0, -1  # Vecteur "haut" de la caméra (par exemple, l'axe Z positif)
    plotter.camera_position = [(x, y, z), (center_x, center_y, center_z), (up_x, up_y, up_z)]
    plotter.show(screenshot=filename)
    plotter.close()




if __name__ == '__main__':

    fps_list = np.array([])
    #videoWrite = cv2.VideoWriter('./IGEV_Stereo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1242, 750))
    #videoWrite = cv2.VideoWriter('./IGEV_Stereo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (514, 752))
    videoWrite = cv2.VideoWriter('./IGEV_Stereo_2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (514*3, 376))
    #videoWrite = cv2.VideoWriter('./IGEV_Stereo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (514, 376))
    for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
        # image1 = load_image(imfile1)
        # image2 = load_image(imfile2)

        # load image

        print(imfile1)
        print(imfile2)

        image1 = np.array(Image.open(imfile1)).astype(np.uint8)
        image2 = np.array(Image.open(imfile2)).astype(np.uint8)

        # Resize images to desired dimensions
        ht0, wd0 = 376, 514
        image1 = cv2.resize(image1, (wd0, ht0))
        image2 = cv2.resize(image2, (wd0, ht0))
        # rectify image due to distortion

        K_l = np.array([322.580, 0.0, 259.260, 0.0, 322.580, 184.882, 0.0, 0.0, 1.0]).reshape(3,3)
        d_l = np.array([-0.070162237, 0.07551153, 0.0012286149,  0.00099302817, -0.018171599])
        R_l = np.array([
    0.9999956354796169, -0.002172438871054654, 0.002002381349442793,
     0.002175041160237588, 0.9999967917532834, -0.00129833704855268,
     -0.001999554367437393, 0.001302686643787701, 0.9999971523908654
        ]).reshape(3,3)
        P_l = np.array([
    322.6092376708984, 0, 257.7363166809082, 0,
     0, 322.6092376708984, 186.6225147247314, 0,
     0, 0, 1, 0
            ]).reshape(3,4)
        map_l = cv2.initUndistortRectifyMap(K_l, d_l, R_l, P_l[:3,:3], (514, 376), cv2.CV_32F)
       
        #print("------------ Right pre rectification ------------------")
        K_r = np.array([
    322.638671875, 0, 255.9466552734375,
     0, 322.638671875, 187.4475402832031,
     0, 0, 1
            ]).reshape(3,3)
        d_r = np.array([
            -0.070313379,
     0.071827024,
     0.0004486586,
     0.00070285366,
     -0.015095583
            ]).reshape(5)
        R_r = np.array([
    0.9999984896881986, -0.001713768657967563, -0.0002891683050380818,
     0.001714143276046202, 0.9999976855080072, 0.001300265918105914,
     0.0002869392807828858, -0.001300759630204676, 0.9999991128447232
        ]).reshape(3,3)
        
        P_r = np.array([
    322.6092376708984, 0, 257.7363166809082, 48.37263543147446,
     0, 322.6092376708984, 186.6225147247314, 0,
     0, 0, 1, 0
            ]).reshape(3,4)
        map_r = cv2.initUndistortRectifyMap(K_r, d_r, R_r, P_r[:3,:3], (514, 376), cv2.CV_32F)

        intrinsics_vec = [322.6092376708984, 322.6092376708984, 257.7363166809082, 186.6225147247314]
        ht0, wd0 = [376, 514]

        # read all png images in folder
        #print("------- image paths ------")
        image1 = cv2.remap(image1, map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)
        image1_rectified = image1
        image2 = cv2.remap(image2, map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)

        # conversion torch
        print("--- conversion torch")
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image1 = image1[None].to(DEVICE)

        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        image2 = image2[None].to(DEVICE)

        padder = InputPadder(image1.shape, divis_by=32)
        image1_pad, image2_pad = padder.pad(image1, image2)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=half_precision):
                disp = model(image1_pad, image2_pad, iters=16, test_mode=True)
                disp = padder.unpad(disp)
        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        fps = 1000/runtime
        fps_list = np.append(fps_list, fps)
        if len(fps_list) > 5:
            fps_list = fps_list[-5:]
        avg_fps = np.mean(fps_list)
        print('Stereo runtime: {:.3f}'.format(1000/avg_fps))

        disp_np = (2*disp).data.cpu().numpy().squeeze().astype(np.uint8)
        disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_PLASMA)


        fx, fy, cx1, cy = 322.580, 322.580, 259.260, 184.882
        cx2 = 255.9466552734375
        baseline=149.604

        disp_depth = disp.cpu().numpy().squeeze()
        depth = (fx * baseline) / (-disp_depth + (cx2 - cx1))

        H, W = depth.shape
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        points_grid = np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0) * depth

        mask = np.ones((H, W), dtype=bool)

        # Remove flying points
        mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
        mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False

        points = points_grid.transpose(1,2,0)[mask]

        NUM_POINTS_TO_DRAW = min(100000, points.shape[0])

        subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW,), replace=False)
        points_subset = points[subset]
        
        x, y, z = points_subset.T

        points = np.column_stack((x, -z, -y))  # Flipping y and z as in the original code

        point_cloud = pv.PolyData(points)

        # Ajouter les valeurs de profondeur comme scalaires pour la coloration
        #point_cloud['depth'] = np.round(z/1000)
        point_cloud['depth'] = z/1000

        point_cloud_image_path = 'point_cloud.png'
        create_point_cloud_image(point_cloud, point_cloud_image_path, points)
        point_cloud_image = cv2.imread(point_cloud_image_path)

        # plt.figure(figsize=(10, 8))
        # #plt.imshow(np.round(depth/1000), cmap='jet')
        # plt.imshow(depth/1000, cmap='jet')
        # plt.colorbar(label='Depth')
        # plt.title('Depth Map')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()

# Créer l'image de la carte de profondeur
        depth_map_image = create_depth_map_image(depth)
        # cv2.imshow("test",depth_map_image)
        # cv2.waitKey(0)


        image_np = np.array(Image.open(imfile1)).astype(np.uint8)       
        #out_img = np.concatenate((image_np, disp_np), 0)

        print("image_np.shape : ",image_np.shape)
        print("disp_depth.shape : ",disp_depth.shape)
        print("depth.shape : ",depth.shape)

        # #depth_img = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_PLASMA)
        # #depth_img = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)
        # depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        # depth_img = cv2.applyColorMap(np.uint8(depth_normalized), cv2.COLORMAP_JET)
        # # cv2.imshow("tmp",depth_img)
        # # cv2.waitKey(0)
        # #
        # # print("depth_img.shape : ",depth_img.shape)
        #
        # #out_img = np.concatenate((image_np, depth_img), 0)

        # Redimensionner depth_map_image pour qu'elle ait la même largeur que image_np
        print("depth_map_image.shape : ",depth_map_image.shape)
        # new_height = int(depth_map_image.shape[0] * (image_np.shape[1] / depth_map_image.shape[1]))
        # depth_map_image_resized = cv2.resize(depth_map_image, (image_np.shape[1], new_height))
        depth_map_image_resized = cv2.resize(depth_map_image, (514, 376))
        print("depth_map_image_resized.shape : ",depth_map_image_resized.shape)

        # Récupérer le nom du fichier avec l'extension
        filename_with_extension = os.path.basename(imfile1)
        # Récupérer uniquement le nom du fichier sans l'extension
        filename_without_extension = os.path.splitext(filename_with_extension)[0]

        file_path = "/home/ivm/Selective-Stereo/Selective-IGEV/test_pipe/depth_imgs/"+filename_without_extension+".png"
        print("======= file_path : ",file_path)

        depth_grayscale = depthmap_to_grayscale(depth / 1000.0)

        #display_image(depth_grayscale, title='DepthMap grayscale')
        
        cv2.imwrite(file_path, depth_grayscale)

        depth_data = cv2.resize(depth, (514, 376))

        depth_path = "/home/ivm/Selective-Stereo/Selective-IGEV/test_pipe/depth/"+filename_without_extension+".npy" 
        np.save(depth_path, depth_data)



        point_cloud_image_resized = cv2.resize(point_cloud_image, (514,376))
        print("point_cloud_image_resized.shape : ",point_cloud_image_resized.shape)

        out_img = np.concatenate((image_np, depth_map_image_resized, point_cloud_image_resized), 1)
        # cv2.putText(
        #     out_img,
        #     "%.1f fps" % (avg_fps),
        #     (10, image_np.shape[0]+30),
        #     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out_img,
            "%.1f fps" % (avg_fps),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('img', out_img)
        cv2.waitKey(1)
        videoWrite.write(out_img)
        # videoWrite.write(depth_map_image_resized)
    videoWrite.release()
