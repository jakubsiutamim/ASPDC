import argparse
import functools

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch.utils.data import DataLoader
from PIL import Image

from networks.sub_networks import DeblurringNet
from utils.data_processing import *
import torch.nn as nn
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchvision.transforms import ToPILImage
from utils.data_processing import get_normalize, toTensor


def process_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frames_num)
    try:
        while True:
            ret, frame = cap.read()
        
            if not ret:
                break
    
            yield frame
         
    finally:
        cap.release()

def preprocess_to_tensor(img):
    normalize = get_normalize()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, _ = normalize(img, img)
    return toTensor(img).unsqueeze(0) 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--testset_dir', type=str, default='/app/repo/GOPRO')
    parser.add_argument('--dst', type=str, default='/app/output/Capture0010/motion06350/')
    parser.add_argument('--src', type=str, default='/app/data/Capture0010/motion_06350.mov')

    return parser.parse_args()

def numpy_to_cv2(filepath, img):
        
    img = (img* 255).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    print(filepath) 
    cv2.imwrite(filepath, img)

def video_inference(cfg):
    net = DeblurringNet(norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)).to(
        cfg.device)

    pretrained_dict = torch.load('final_model/DeblurringNet_FT.pth')
    net.load_state_dict(pretrained_dict['deblurring_state_dict'])
    
    print('co')
    with torch.no_grad():
        for idx, frame in enumerate(process_video_frames(cfg.src)):
            img_tensor = preprocess_to_tensor(np.copy(frame))
            img_tensor = img_tensor.to(cfg.device)

            result, _ = net(img_tensor)

            result = torch.clamp(result, -1, 1)
            result = (result + 1) /2

            result = result[0, ...].detach().permute(1, 2, 0).cpu().numpy()

            #numpy_to_cv2(f'frame_original_{idx}.jpg', frame)
            #numpy_to_cv2(f'frame_deblurred_{idx}.jpg', result)
            cv2.imwrite(f'{cfg.dst}frame_original{idx}.jpg', frame)
            numpy_to_cv2(f'{cfg.dst}frame_deblurred_{idx}.jpg', result)
            if idx > 1:
                return

def inference(test_loader, cfg):
    net = DeblurringNet(norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)).to(
        cfg.device)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    pretrained_dict = torch.load('final_model/DeblurringNet_FT.pth')
    net.load_state_dict(pretrained_dict['deblurring_state_dict'])
    
    print(f'test_loader: {list(test_loader)}')
    psnr_list = []
    ssim_list = []
    # torch.cuda.empty_cache()
    with torch.no_grad():
        for idx_iter, (img_hr, img_blur) in enumerate(test_loader):
            

            img_hr = img_hr.to(cfg.device)
            img_blur = img_blur.to(cfg.device)
            deblurred_rgb, _ = net(img_blur)

            deblurred_rgb = torch.clamp(deblurred_rgb, -1, 1)

            deblurred_rgb = (deblurred_rgb + 1) / 2

            img_hr = (img_hr + 1) / 2
            img_blur = (img_blur + 1) / 2

            deblurring_output = deblurred_rgb[0, ...].detach().permute(1, 2, 0).cpu().numpy()

            hr_numpy = img_hr[0, ...].detach().permute(1, 2, 0).cpu().numpy()
            blur_numpy = img_blur[0, ...].detach().permute(1, 2, 0).cpu().numpy()

            psnr = peak_signal_noise_ratio(deblurring_output, hr_numpy)
            psnr_list.append(psnr)
            print(psnr)
            
            numpy_to_cv2('out_images/end_hr.jpg', hr_numpy)
            numpy_to_cv2('out_images/end_blur.jpg', blur_numpy)
            numpy_to_cv2('out_images/end_deblurred.jpg', deblurring_output)
            # ssim = structural_similarity(deblurring_output, hr_numpy, multichannel=True)
            # ssim_list.append(ssim)
            # print(ssim)
            torch.cuda.empty_cache()
    print()
    print(np.mean(np.array(psnr_list)))
    # print(np.mean(np.array(ssim_list)))


def main(cfg):
    # test_set = TestSetLoader(dataset_dir=cfg.testset_dir)
    # test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    # inference(test_loader, cfg)
    video_inference(cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
