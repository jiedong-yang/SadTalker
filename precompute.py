import os
import cv2
import sys
import json
import shutil
import argparse
import logging

from tqdm import tqdm
from src.utils.preprocess import CropAndExtract

# Set up logging
logging.basicConfig(filename='precompute.log', level=logging.INFO)

current_code_path = sys.argv[0]
current_root_path = os.path.split(current_code_path)[0]

os.environ['TORCH_HOME'] = os.path.join(current_root_path, './checkpoints')

path_of_lm_croper = os.path.join(current_root_path, './checkpoints', 'shape_predictor_68_face_landmarks.dat')
path_of_net_recon_model = os.path.join(current_root_path, './checkpoints', 'epoch_20.pth')
dir_of_BFM_fitting = os.path.join(current_root_path, './checkpoints', 'BFM_Fitting')

preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, 'cuda')

error_videos = []


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
    try:
        ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir)
    except TypeError:
        error_videos.append(ref_pose)
        os.rmdir(ref_pose_frame_dir)


def traverse_and_precompute(video_dir, result_dir):
    for root, dirs, files in tqdm(os.walk(video_dir)):
        for file in tqdm(files):
            if file.endswith(('.mp4', '.avi', '.flv', '.mov')):  # add or remove video formats as needed
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, video_dir)
                save_dir = os.path.join(result_dir, relative_path)
                os.makedirs(save_dir, exist_ok=True)
                precompute(video_path, save_dir)


def traverse_and_get_videos(video_dir):
    video_paths = []
    for root, dirs, files in tqdm(os.walk(video_dir)):
        for file in tqdm(files):
            if file.endswith(('.mp4', '.avi', '.flv', '.mov')):  # add or remove video formats as needed
                video_path = os.path.join(root, file)
                video_paths.append(video_path)
    print(len(video_paths))
    return video_paths


def exclude_videos_from_json(video_paths, json_file):
    with open(json_file, 'r') as f:
        excluded_videos = json.load(f)
    return sorted([video for video in video_paths if video not in excluded_videos])


def create_video_mapping(video_paths, json_path):
    video_mapping = {}
    for i, video_path in tqdm(enumerate(video_paths)):
        # retrieve video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps

        # precomputed .mat path
        data_path = video_path[:]
        data_path = os.path.splitext(data_path.replace('DataFinal', 'datafinal-results'))[0]
        base_name = os.path.basename(data_path)
        data_path = os.path.join(data_path, base_name + '.mat')

        video_mapping[i] = {
            "path": video_path,
            "duration": duration,
            "nframes": frame_count,
            "fps": fps, "data_path": data_path
        }

        with open(json_path, 'w') as f:
            json.dump(video_mapping, f)
    return video_mapping


def build_precompute_dataset(video_dir, result_dir, json_name='data.json'):
    # 1. precompute all video clips and get error videos
    traverse_and_precompute(args.video_dir, args.result_dir)
    logging.info(f"Precompute completed for directory: {video_dir}")

    # 2. save error videos
    with open(os.path.join(result_dir, 'error_videos.json'), 'w') as f:
        json.dump(error_videos, f, indent=4)
    logging.info(f"Saved error videos to: {os.path.join(result_dir, 'error_videos.json')}")

    # 3. get all video paths
    video_paths = traverse_and_get_videos(video_dir)

    # 4. exclude error videos
    valid_video_paths = exclude_videos_from_json(video_paths, os.path.join(result_dir, 'error_videos.json'))

    # 5. create video mapping
    video_mapping = create_video_mapping(valid_video_paths, os.path.join(result_dir, json_name))
    logging.info(f"Created video mapping for {len(video_mapping)} videos.")

    # 6. log some stats
    logging.info(f"Number of videos that failed to precompute: {len(error_videos)}")
    logging.info(f"Total number of frames in all videos: {len(video_paths)}")
    logging.info(f"Average frames per second across all videos: {len(valid_video_paths)/len(video_paths):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Precompute video data.')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the video directory.')
    parser.add_argument('--result_dir', type=str, required=True, help='Path to the result directory.')
    parser.add_argument('--json_name', type=str, default='data.json', help='Json file name for data')
    args = parser.parse_args()

    build_precompute_dataset(args.video_dir, args.result_dir, args.json_name)

    # with open(os.path.join(args.result_dir, 'error_videos.json'), 'w') as f:
    #     json.dump(error_videos, f, indent=4)
