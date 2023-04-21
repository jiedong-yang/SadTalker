# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, File

import torch
from time import strftime
import os
import sys
import time
from typing import Optional, List, Union

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data

import tempfile


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        # current_code_path = sys.argv[0]
        # current_root_path = os.path.split(current_code_path)[0]
        #
        # os.environ['TORCH_HOME'] = os.path.join(current_root_path, './checkpoints')

        path_of_lm_croper = os.path.join('./checkpoints',
                                         'shape_predictor_68_face_landmarks.dat')
        path_of_net_recon_model = os.path.join('./checkpoints', 'epoch_20.pth')
        dir_of_BFM_fitting = os.path.join('./checkpoints', 'BFM_Fitting')
        wav2lip_checkpoint = os.path.join('./checkpoints', 'wav2lip.pth')

        audio2pose_checkpoint = os.path.join('./checkpoints', 'auido2pose_00140-model.pth')
        audio2pose_yaml_path = os.path.join('src', 'config', 'auido2pose.yaml')

        audio2exp_checkpoint = os.path.join('./checkpoints', 'auido2exp_00300-model.pth')
        audio2exp_yaml_path = os.path.join('src', 'config', 'auido2exp.yaml')

        self.free_view_checkpoint = os.path.join('./checkpoints', 'facevid2vid_00189-model.pth.tar')

        # init model
        print(path_of_net_recon_model)
        self.preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, "cuda")

        print(audio2pose_checkpoint)
        print(audio2exp_checkpoint)
        self.audio_to_coeff = Audio2Coeff(
            audio2pose_checkpoint, audio2pose_yaml_path,
            audio2exp_checkpoint, audio2exp_yaml_path,
            wav2lip_checkpoint, "cuda"
        )

        print(self.free_view_checkpoint)

        # mapping_checkpoint = os.path.join('./checkpoints', 'mapping_00229-model.pth.tar')
        # facerender_yaml_path = os.path.join('src', 'config', 'facerender.yaml')
        #
        # print(mapping_checkpoint)
        # self.animate_from_coeff = AnimateFromCoeff(
        #     self.free_view_checkpoint, mapping_checkpoint, facerender_yaml_path, "cuda"
        # )

    def predict(
        self,
        image: Path = Input(description="Avatar image input", default=Path('./examples/source_image/happy.png')),
        audio: Path = Input(
            description="Driving audio input, mono only", default=Path('./examples/driven_audio/junk_audio.mp3')
        ),
        # ref_pose: Path = Input(description="pose reference video"),
        # ref_eyeblink: Path = Input(description="eye blink reference video"),
        preprocess: str = Input(description="preprocess mode", choices=['crop', 'resize'], default='crop'),
        still: str = Input(
            description="still mode", choices=['True', 'False'], default='False'
        ),
        pose_style: int = Input(description="style of poses, from 0 to 45", ge=0, le=45, default=0),
        batch_size: int = Input(
            description="the batch size of facerender, defaulted by 16", ge=1, le=32, default=16
        ),
        expression_scale: float = Input(
            description="expression scale of output", ge=.01, le=5., default=1.
        ),
        enhancer: str = Input(
            description="Face enhancer, [gfpgan, RestoreFormer]", choices=['None', 'gfpgan', 'RestoreFormer'], default='None'
        ),
        background_enhancer: str = Input(
            description="background enhancer, realesrgan", choices=['None', 'realesrgan'], default='None'
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        image, audio, ref_pose = str(image), str(audio), None

        ref_eyeblink = None

        input_yaw_list, input_pitch_list, input_roll_list = None, None, None

        still = True if still == "True" else False
        enhancer = None if enhancer == 'None' else enhancer
        background_enhancer = None if background_enhancer == 'None' else background_enhancer

        save_dir = os.path.join(os.getcwd(), 'tmp')
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print('3DMM Extraction for source image')

        if preprocess == 'full':
            mapping_checkpoint = os.path.join('./checkpoints', 'mapping_00109-model.pth.tar')
            facerender_yaml_path = os.path.join('src', 'config', 'facerender_still.yaml')
        else:
            mapping_checkpoint = os.path.join('./checkpoints', 'mapping_00229-model.pth.tar')
            facerender_yaml_path = os.path.join('src', 'config', 'facerender.yaml')

        print(mapping_checkpoint)
        self.animate_from_coeff = AnimateFromCoeff(
            self.free_view_checkpoint, mapping_checkpoint, facerender_yaml_path, "cuda"
        )

        with torch.inference_mode():
            first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
                image, first_frame_dir, preprocess, source_image_flag=True
            )

            if first_coeff_path is None:
                # print("Can't get the coeffs of the input")
                return "Can't get the coeffs of the input"

            if ref_eyeblink is not None:
                ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
                ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
                os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing eye blinking')
                ref_eyeblink_coeff_path, _, _ = self.preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir)
            else:
                ref_eyeblink_coeff_path = None

            if ref_pose is not None:
                if ref_pose == ref_eyeblink:
                    ref_pose_coeff_path = ref_eyeblink_coeff_path
                else:
                    ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                    ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                    os.makedirs(ref_pose_frame_dir, exist_ok=True)
                    print('3DMM Extraction for the reference video providing pose')
                    ref_pose_coeff_path, _, _ = self.preprocess_model.generate(ref_pose, ref_pose_frame_dir)
            else:
                ref_pose_coeff_path = None

            # audio2ceoff
            batch = get_data(first_coeff_path, audio, "cuda", ref_eyeblink_coeff_path, still=still)
            coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

            # 3dface render
            # if face3dvis:
            #     from src.face3d.visualize import gen_composed_video
            #     gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path,
            #                        os.path.join(save_dir, '3dface.mp4'))

            # coeff2video
            data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio,
                                       batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                       expression_scale=expression_scale, still_mode=still,
                                       preprocess=preprocess)

            outputs = self.animate_from_coeff.generate(
                data, save_dir, image, crop_info,
                enhancer=enhancer,
                background_enhancer=background_enhancer,
                preprocess=preprocess
            )

        if preprocess == 'full':
            # print(outputs)
            results = []
            for output in outputs:
                # print(output)
                results.append(Path(output))
            return results
        return [Path(outputs)]
