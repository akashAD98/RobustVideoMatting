#! /usr/bin/python3
# coding=utf-8

import torch
from model import MattingNetwork
from inference import convert_video


if __name__ == '__main__':

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# option1: load mobilenetv3 model
	model = MattingNetwork('mobilenetv3').eval().to(device)
	model.load_state_dict(torch.load('work/checkpoint/rvm_mobilenetv3.pth'))

	# output video
	convert_video(
		model,                                       # （cpu or cuda）
		input_source='work/video/input.avi',         # 
		# num_workers=1,                             # 
		# input_resize=(1080, 720),                  # 
		output_type='video',                         # "video" or "png_sequence"（PNG）
		output_background='image',                   #  "green", "white", "image"
		output_composition='work/video/output.avi',  # 
		output_alpha="work/video/pha.avi",           # 
		output_foreground="work/video/fgr.avi",      #
		output_video_mbps=4,                         # 
		downsample_ratio=None,                       # 
		require_audio=True,                          # 
        seq_chunk=1,                                 # 
		progress=True,                               # 
	)

	# output png_sequence
	convert_video(
		model,                                        # （cpu or cuda）
		input_source='work/video/input.avi',          # 
		# num_workers=1,                              # 
		# input_resize=(1080, 720),                   # 
		output_type='png_sequence',                   # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
		output_background='image',                    # put "default","green", "white", "image" ""assign your own image"
		output_composition='work/video/output',       # 
		output_alpha="work/video/pha",                # 
		output_foreground="work/video/fgr",           # 
		downsample_ratio=None,                        # 512px
		seq_chunk=1,                                  # 
		progress=True                                 # 
	)
