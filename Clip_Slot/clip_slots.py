"""
from moviepy.editor import *
video1 = "./../slot2024_05_27/1ren/2.mp4"
video2 = "./../slot2024_05_27/1ren/3.mp4"

clip1 = VideoFileClip(video1)
clip2 = VideoFileClip(video2)
final_clip = concatenate_videoclips([clip1, clip2])
final_clip.write_videofile("output.mp4", fps=24, remove_temp=False)
"""


"""
from moviepy.editor import VideoFileClip, concatenate_videoclips

video_1 = VideoFileClip("./2.mp4")
video_2 = VideoFileClip("./3.mp4")

video_2_resized = video_2.resize(width=video_1.w, height=video_1.h) 

final_video= concatenate_videoclips([video_1, video_2_resized])

final_video.write_videofile("final_video.mp4", fps=24)
"""

"""
from moviepy.editor import VideoFileClip, concatenate_videoclips

clip1 = VideoFileClip("./2.mp4")
clip2 = VideoFileClip("./3.mp4")

# 确保两个视频的帧率一致
if clip1.fps != clip2.fps:
    clip2 = clip2.set_duration(clip1.duration)

# 确保两个视频的大小一致
if clip1.size != clip2.size:
    clip2 = clip2.resize(clip1.size)

final_clip = concatenate_videoclips([clip1, clip2])

# 使用与源视频相同的编码器
final_clip.write_videofile("my_concatenation.mp4", codec='libx264')
"""


from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import resize

def adjust_fps_and_size(clip1, clip2):
    max_fps = max(clip1.fps, clip2.fps)

    # Adjust fps
    if clip1.fps != max_fps:
        clip1 = clip1.set_duration(clip1.duration*clip1.fps/max_fps)
        clip1.fps = max_fps
    if clip2.fps != max_fps:
        clip2 = clip2.set_duration(clip2.duration*clip2.fps/max_fps)
        clip2.fps = max_fps

    # Adjust size
    max_size = max(clip1.size, clip2.size)
    clip1 = resize(clip1, max_size)
    clip2 = resize(clip2, max_size)

    return clip1, clip2

# Load your videos
clip1 = VideoFileClip("./2.mp4")
clip2 = VideoFileClip("./3.mp4")

# Adjust fps and size
clip1, clip2 = adjust_fps_and_size(clip1, clip2)

# Now you can concatenate them
final_clip = concatenate_videoclips([clip1,clip2])

# Write the result to a file
final_clip.write_videofile("final.mp4", codec='libx264')



"""

# 主要是需要moviepy这个库
from moviepy.editor import *
import os
 
# 定义一个数组
L = []
 
# 访问 video 文件夹 (假设视频都放在这里面)
for root, dirs, files in os.walk("./../slot2024_05_27/1ren"):
    # 按文件名排序
    files.sort()
    # 遍历所有文件
    for file in files:
        # 如果后缀名为 .mp4
        if os.path.splitext(file)[1] == '.mp4':
            # 拼接成完整路径
            filePath = os.path.join(root, file)
            # 载入视频
            video = VideoFileClip(filePath)
            # 添加到数组
            L.append(video)
 
# 拼接视频
final_clip = concatenate_videoclips(L)
 
# 生成目标视频文件
final_clip.to_videofile("./target.mp4", fps=24, remove_temp=False)

"""

"""

final = CompositeVideoClip([video1, video2])  #video2会覆盖在video1之上
final = CompositeVideoClip([video1, video2.set_start(5).crossfadein(1)])  #先播放video1，在第五秒开始video2播放，并以“渐入”的特效显示

指定位置，将两个视频在同一画面播放。假设我们已经选用了合适的尺寸的两个视频,计算尺寸使视频不会覆盖。
video = CompositeVideoClip([
                            video1.set_pos((0,150),
                            video2.set_pos((100,150))
                            ])

"""