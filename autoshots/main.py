#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
这个Python脚本使用funclip库来执行视频剪辑的两个阶段：
1. 识别阶段：从视频文件中提取识别结果和SRT文件。
2. 剪辑阶段：根据指定的文本和时间范围剪辑视频。

作者：[WangWei]
创建日期：[2024-06-12]
版本：1.0
"""

import os
from funclip import VideoClipper, get_parser, runner

# 定义常量或全局变量
VIDEO_FILE = 'examples/2022云栖大会_片段.mp4'
OUTPUT_DIR = './output'

# 主程序入口
if __name__ == "__main__":
    # 创建解析器并获取命令行参数
    parser = get_parser()
    args = parser.parse_args([
        '--stage', '1',
        '--file', VIDEO_FILE,
        '--output_dir', OUTPUT_DIR
    ])

    # 执行识别阶段
    runner(
        stage=args.stage,
        file=args.file,
        sd_switch=args.sd_switch,
        output_dir=args.output_dir,
        dest_text=None,
        dest_spk=None,
        start_ost=0,
        end_ost=0,
        output_file=None
    )

    # 执行剪辑阶段
    args = parser.parse_args([
        '--stage', '2',
        '--file', VIDEO_FILE,
        '--output_dir', OUTPUT_DIR,
        '--dest_text', '我们把它跟乡村振兴去结合起来，利用我们的设计的能力',
        '--start_ost', '0',
        '--end_ost', '100',
        '--output_file', './output/res.mp4'
    ])
    runner(
        stage=args.stage,
        file=args.file,
        sd_switch=args.sd_switch,
        output_dir=args.output_dir,
        dest_text=args.dest_text,
        dest_spk=None,
        start_ost=args.start_ost,
        end_ost=args.end_ost,
        output_file=args.output_file
    )