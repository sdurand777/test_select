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
import time

DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

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

        #left_images = sorted(glob.glob("/home/ivm/test_select/demo-imgs/Motorcycle/left/*.png", recursive=True))
        #left_images = sorted(glob.glob("/home/ivm/test_select/test/left/*.png", recursive=True))

        #left_images = [args.left_imgs]
        print("left_images : ", left_images)

        right_images = sorted(glob.glob(args.right_imgs, recursive=True))

        #right_images = sorted(glob.glob("/home/ivm/test_select/demo-imgs/Motorcycle/right/*.png", recursive=True))
        #right_images = sorted(glob.glob("/home/ivm/test_select/test/right/*.png", recursive=True))

        #right_images = [args.right_imgs]

        print("right_images : ", right_images)

        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")




        total_inference_time = 0
        num_iterations = 100

        # Warm-up iterations (to avoid any cold-start delays)
        for _ in range(num_iterations):
            for (imfile1, imfile2) in zip(left_images, right_images):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)
                padder = InputPadder(image1.shape, divis_by=32)
                image1, image2 = padder.pad(image1, image2)
                start_time = time.time()
                _ = model(image1, image2, iters=16, test_mode=True)
                end_time = time.time()
                inference_time = end_time - start_time
                total_inference_time += inference_time

        average_inference_time = total_inference_time / num_iterations
        print(f"Average Inference Time over {num_iterations} iterations: {average_inference_time:.4f} seconds")


        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = disp.cpu().numpy()
            disp = padder.unpad(disp)
            #file_stem = imfile1.split('/')[-2]
            file_stem = "result"
            filename = os.path.join(output_directory, f'{file_stem}.png')
            plt.imsave(filename, disp.squeeze(), cmap='jet')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo_imgs/Motorcycle/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo_imgs/Motorcycle/im1.png")

    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo_imgs/Motorcycle/resized/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo_imgs/Motorcycle/resized/im1.png")



    parser.add_argument('--output_directory', help="directory to save output")
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
