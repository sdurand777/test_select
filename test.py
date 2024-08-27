import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import torch.nn.functional as F
import pyvista as pv
import cv2

DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)



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



def demo(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            # image1 = load_image(imfile1)
            # image2 = load_image(imfile2)
            #
            # padder = InputPadder(image1.shape, divis_by=32)
            # image1, image2 = padder.pad(image1, image2)
            #
            # disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            # disp = disp.cpu().numpy()
            # disp = padder.unpad(disp)
            # file_stem = imfile1.split('/')[-2]
            # filename = os.path.join(output_directory, f'{file_stem}.png')
            # plt.imsave(filename, disp.squeeze(), cmap='jet')


            image1 = np.array(Image.open(imfile1)).astype(np.uint8)
            image2 = np.array(Image.open(imfile2)).astype(np.uint8)

            plt.imshow(image1)
            plt.show()

            # Resize images to desired dimensions
            ht0, wd0 = 376, 514
            image1 = cv2.resize(image1, (wd0, ht0))
            image2 = cv2.resize(image2, (wd0, ht0))
            # rectify image due to distortion

            plt.imshow(image1)
            plt.show()

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

            plt.imshow(image1)
            plt.show()

            print("image1 shape : ", image1.shape)
            print("image2 shape : ", image2.shape)

            # conversion torch
            print("--- conversion torch")
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
            image1 = image1[None].to(DEVICE)

            image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
            image2 = image2[None].to(DEVICE)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            #_, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

            file_stem = imfile1.split('/')[-2]
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')

            # save disp
            disp = flow_up.cpu().numpy().squeeze()

            # use depth
            # # Calibration for MIddleBury
            # fx, fy, cx1, cy = 3997.684, 3997.684, 1176.728, 1011.728
            # cx2 = 1307.839
            # baseline=193.001 # in millimeters

            # Calibration IVM
            # fx, fy, cx1, cy = 2567.184959203512, 2567.184959203512, 2039.7738057347667, 1445.7039464541153
            # # cx de la deuxieme image
            # cx2 = 2090.0688455990453
            # baseline=149.604

            # Calibration IVM SLAM
            fx, fy, cx1, cy = 322.580, 322.580, 259.260, 184.882
            cx2 = 255.9466552734375
            baseline=149.604

            # inverse-project
            depth = (fx * baseline) / (-disp + (cx2 - cx1))
            H, W = depth.shape
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            points_grid = np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0) * depth

            mask = np.ones((H, W), dtype=bool)

            # Remove flying points
            mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
            mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False

            points = points_grid.transpose(1,2,0)[mask]

            # testF_folder = Path("datasets/Middlebury/MiddEval3/testF/Bicycle2")
            # image = imread(testF_folder / "im0.png")
            # # test image IVM pleine taille non rectifie
            # image = imread("test_ivm/20230823-13h05m17s_08424LRL.JPG")
            image = image1_rectified
            colors = image[mask].astype(np.float64) / 255

            #NUM_POINTS_TO_DRAW = 100000
            NUM_POINTS_TO_DRAW = min(100000, points.shape[0])

            subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW,), replace=False)
            points_subset = points[subset]
            colors_subset = colors[subset]

            print("""
            Controls:
            ---------
            Zoom:      Scroll Wheel
            Translate: Right-Click + Drag
            Rotate:    Left-Click + Drag
            """)

            x, y, z = points_subset.T

            # fig = go.Figure(
            #     data=[
            #         go.Scatter3d(
            #             x=x, y=-z, z=-y, # flipped to make visualization nicer
            #             mode='markers',
            #             marker=dict(size=1, color=colors_subset)
            #         )
            #     ],
            #     layout=dict(
            #         scene=dict(
            #             xaxis=dict(visible=True),
            #             yaxis=dict(visible=True),
            #             zaxis=dict(visible=True),
            #         )
            #     )
            # )
            # fig.show()

            points = np.column_stack((x, -z, -y))  # Flipping y and z as in the original code

# Créer un objet PolyData avec les points
            point_cloud = pv.PolyData(points)

# Ajouter les valeurs de profondeur comme scalaires pour la coloration
            #point_cloud['depth'] = np.round(z/1000)
            point_cloud['depth'] = z/1000
            
            print("z :")
            print(np.min(z))
            print(np.max(z))
            print("y :")
            print(np.min(y))
            print(np.max(y))



# Créer un plotter
            plotter = pv.Plotter()

# Ajouter le nuage de points au plotter avec la taille des points augmentée
            plotter.add_points(point_cloud, scalars='depth', point_size=5, render_points_as_spheres=True, cmap='jet')

            point_cloud_image_path = 'point_cloud.png'
            create_point_cloud_image(point_cloud, point_cloud_image_path, points)
            point_cloud_image = cv2.imread(point_cloud_image_path)
            cv2.imshow("test depth",point_cloud_image)
            cv2.waitKey(0)


# Afficher le nuage de points
            plotter.show()

# # Supposons que x, y, z et colors_subset sont déjà définis
# # Créer un nuage de points avec PyVista
#             points = np.column_stack((x, -z, -y))  # Flipping y and z as in the original code
#
# # Créer un objet PolyData avec les points
#             point_cloud = pv.PolyData(points)
#
# # Ajouter des couleurs si nécessaire
#             point_cloud['colors'] = colors_subset
#
# # Créer un plotter
#             plotter = pv.Plotter()
#
# # Ajouter le nuage de points au plotter
#             plotter.add_points(point_cloud, scalars='colors', point_size=1, render_points_as_spheres=True)
#
# # Afficher le nuage de points
#             plotter.show()


            # Display depth map with color bar
            plt.figure(figsize=(10, 8))
            #plt.imshow(np.round(depth/1000), cmap='jet')
            plt.imshow(depth/1000, cmap='jet')
            plt.colorbar(label='Depth')
            plt.title('Depth Map')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default=None)
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default=None)

    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./test_ivm/img0.jpg")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./test_ivm/img1.jpg")

    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./test_300/img0.jpg")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./test_300/img1.jpg")



    parser.add_argument('--output_directory', help="directory to save output", default=None)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
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

    demo(args)
