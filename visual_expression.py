import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math

#グローバル変数
pop_flag = False

##ズーム
#pilow -> cv2
def pil2cv(imgPIL) -> np.ndarray:
  imgCV_RGB = np.array(imgPIL, dtype = np.uint8)
  imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
  return imgCV_BGR
#cv2 -> pillow
def cv2pil(imgCV):
  imgCV_RGB = imgCV[:, :, ::-1]
  imgPIL = Image.fromarray(imgCV_RGB)
  return imgPIL

def zoom_frames(frames:list, xmin:float, xmax:float, ymin:float, ymax:float) -> list:
  height, width, _ = frames[0].shape
  box_width = xmax - xmin #ズームする箇所の横幅
  box_height = ymax - ymin #ズームする箇所の縦幅
  zoom_x = int(xmax + xmin) / 2 #ズームの中心x
  zoom_y = int(ymax + ymin) / 2 #ズームの中心y
  zoom_range = box_width*4 / width #ズームの比率
  #切り取る箇所の要素
  trim_width = int(width * zoom_range)
  trim_height = int(height * zoom_range)
  trim_xmin = int(zoom_x - trim_width / 2)
  trim_xmax = int(trim_xmin + trim_width)
  trim_ymin = int(zoom_y - trim_height / 2)
  trim_ymax = int(trim_ymin + trim_height)
  trim_frames = []
  for i in range(len(frames)):
    frame = frames[i]
    frame = cv2pil(frame)
    trim_frame = frame.crop((int(trim_xmin), int(trim_ymin), int(trim_xmax), int(trim_ymax)))
    trim_frame = trim_frame.resize((width, height))
    trim_frame = pil2cv(trim_frame)
    trim_frame = trim_frame.astype(np.uint8)
    trim_frames.append(trim_frame)
  return trim_frames

##方向線（radial_blur）：メイン　radial_frames
#画像にラディアルブラーをかける
def radial_blur(src, pos, ratio, iterations, margin): #(フレーム画像、中心座標、最大縮小率、重ね枚数、余白) -> フレーム画像
  h, w = src.shape[0:2]
  n = iterations
  m = margin
  # 背景作成(255:白　0:黒)
  bg = np.ones(src.shape, dtype=np.uint8) * 255
  bg = cv2.resize(bg, (int(m * w), int(m * h)))
  # 背景の中心に元画像を配置
  bg[int((m - 1) * h / 2):int((m - 1) * h / 2) + h, int((m - 1) * w / 2):int((m - 1) * w / 2) + w] = src
  image_list = []
  h *= m
  w *= m
  c_x = pos[0] * m
  c_y = pos[1] * m
  # 縮小画像の作成
  for i in range(n):
    r = ratio + (1 - ratio) * (i + 1) / n
    shrunk = cv2.resize(src, (int(r * w), int(r * h)))
    left = int((1 - r) * c_x)
    right = left + shrunk.shape[1]
    top = int((1 - r) * c_y)
    bottom = top + shrunk.shape[0]
    bg[top:bottom, left:right] = shrunk
    image_list.append(bg.astype(np.int32))
  # 最終的な出力画像の作成
  dst = sum(image_list) / n
  dst = dst.astype(np.uint8)
  image_list.clear()
  r = (1 + ratio) / 2
  dst = dst[int((1 - r) * c_y):int(((1 - r) * c_y + h) * r), int((1 - r) * c_x):int(((1 - r) * c_x + w) * r)]
  dst = cv2.resize(dst, (int(w / m), int(h / m)))
  return dst

#中心->外に向けて白->黒のマスクを作成
def make_gradation(width:int, height:int) -> np.array:
  grad = np.empty((width, height, 3), dtype=np.uint8)
  basis = height
  if height > width:
    basis = width
  for h in range(height):
    for w in range(width):
      p = np.array([w, h])
      q = np.array([width/2, height/2])
      dist = np.linalg.norm(p - q)
      gray_value = int(255 * (math.cos(0.7*math.pi*dist/(basis*0.8))))
      if gray_value < 0:
        gray_value = 0
      if gray_value > 255:
        gray_value = 255
      grad[w, h] = [gray_value, gray_value, gray_value]
  return grad

#マスクを元にフレームの透明度を変える
def mask_frame(frame, mask):
  height, width = frame.shape[0:2]
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
  for h in range(height):
    row = frame[h]
    for w in range(width):
      col = row[w]
      col[3] = mask[w, h]
      row[w] = col
    frame[h] = row
  return frame

#画像を重ねる(ラディアルブラーの中心を鮮明にするため)
def alpha_blend(frame: np.array, alpha_frame: np.array, position):
    x1, y1 = max(position[0], 0), max(position[1], 0)
    x2 = min(position[0] + alpha_frame.shape[1], frame.shape[1])
    y2 = min(position[1] + alpha_frame.shape[0], frame.shape[0])
    ax1, ay1 = x1 - position[0], y1 - position[1]
    ax2, ay2 = ax1 + x2 - x1, ay1 + y2 - y1
    frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - alpha_frame[ay1:ay2, ax1:ax2, 3:] / 255) + \
                          alpha_frame[ay1:ay2, ax1:ax2, :3] * (alpha_frame[ay1:ay2, ax1:ax2, 3:] / 255)
    return frame

#フレームのlistにラディアルブラーをかける
def radial_frames(frames:list, mask:np.array) -> list:
  height, width = frames[0].shape[0:2]
  center_x, center_y = width/2, height/2
  radial_frames = []
  for i in range(len(frames)):
    frame = frames[i]
    frame = frame.astype(np.float32)
    radial_frame = radial_blur(frame, (center_x, center_y), 0.92, 6, 1.0)
    masked_frame = mask_frame(frame, mask)
    radial_frame = alpha_blend(radial_frame, masked_frame, (0, 0))
    radial_frame = radial_frame.astype(np.uint8)
    radial_frames.append(radial_frame)
    print(str(i/len(frames)*100) + "%")
  return radial_frames


##画面分割(3分割)
def split_frames(frames, df_performer_movement) -> list:
  height, width, _ = frames[0].shape
  split_frames = []
  split_num = 3 #分割する数
  for frame in frames:
    split_img_list = []
    frame_width = width/split_num #分割後の画像の横幅
    performer_count = 0
    for row in df_performer_movement.itertuples():
      trim_center_y = (row.ymin + row.ymax) / 2
      trim_height = row.ymax - row.ymin
      trim_width = trim_height * (frame_width/height)
      trim_frame = frame[int(trim_center_y - (trim_height/2) ): int(trim_center_y + (trim_height/2)), int(row.xmin): int(row.xmax)]
      trim_frame = cv2.resize(trim_frame, (int(frame_width), int(height)))
      split_img_list.append(trim_frame)
      performer_count += 1
      if performer_count == split_num:
        break
    split_frame = cv2.hconcat(split_img_list)
    split_frame = cv2.resize(split_frame, (width, height))
    split_frames.append(split_frame)
  return split_frames