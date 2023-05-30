import os
import sys
import argparse
from src.utils.preprocess import CropAndExtract


current_code_path = sys.argv[0]
current_root_path = os.path.split(current_code_path)[0]

os.environ['TORCH_HOME'] = os.path.join(current_root_path, './checkpoints')

path_of_lm_croper = os.path.join(current_root_path, './checkpoints', 'shape_predictor_68_face_landmarks.dat')
path_of_net_recon_model = os.path.join(current_root_path, './checkpoints', 'epoch_20.pth')
dir_of_BFM_fitting = os.path.join(current_root_path, './checkpoints', 'BFM_Fitting')

preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, 'cuda')


def precompute(ref_pose, save_dir='./results'):
    """ precompute to extract information from video for MATLAB style .mat data

    :param ref_pose:
    :param save_dir:
    :return:
    """
    ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
    ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
    os.makedirs(ref_pose_frame_dir, exist_ok=True)
    print('3DMM Extraction for the reference video providing pose')
    ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir)


def traverse_and_precompute(video_dir, result_dir):
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.flv', '.mov')):  # add or remove video formats as needed
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, video_dir)
                save_dir = os.path.join(result_dir, relative_path)
                os.makedirs(save_dir, exist_ok=True)
                precompute(video_path, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Precompute video data.')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the video directory.')
    parser.add_argument('--result_dir', type=str, required=True, help='Path to the result directory.')
    args = parser.parse_args()

    traverse_and_precompute(args.video_dir, args.result_dir)
