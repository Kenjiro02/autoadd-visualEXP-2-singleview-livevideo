#映像に映る演者を検知
##毎フレーム必要な要素（１つ前のフレーム、１つ前の演者認識結果）
##以下を最初に定義してください
###現在のそれぞれの演者の動作量
####df_performer_movement = pd.DataFrame([], columns=["xmin", "ymin", "xmax", "ymax", "movement", "visual_express"])
###前のフレーム情報
####frame_before = None #1つ前のフレーム
####df_objects_performer_before = pd.DataFrame([], [])#1つ前の上手く人物認識できたフレーム情報
####df_performer_movement_before = pd.DataFrame([], columns=["xmin", "ymin", "xmax", "ymax", "movement", "visual_express"])#１つ前のフレームでのそれぞれの演者の動作量

import cv2
import numpy as np
import pandas as pd
import random
from ultralytics import YOLO
model = YOLO("yolov10n.pt")
names = model.names

def yolo_detection(frame, df_objects_performer_before): #（フレーム画像、１フレーム前の演者データフレーム）-> 演者データフレーム
  objects_person = []
  #yoloでフレームから人物検出しデータフレームに格納
  results = model(frame)
  for result in results:
    boxes = result.boxes
    for box in boxes:
      x1, y1, x2, y2 = box.xyxy[0].tolist()
      conf = box.conf[0]
      cls = box.cls[0].item()
      label = names[int(cls)]
      if label == "person":
        part = "empty"
        miss_count = 0
        objects_person.append([x1, y1, x2, y2, conf, part, miss_count])
  df_objects_person = pd.DataFrame(objects_person, columns=["xmin", "ymin", "xmax", "ymax", "conf", "part", "miss_count"])
  #基準線を定めそれより上の人物を演者とし、ソート
  df_objects_performer = df_objects_person[df_objects_person['ymin'] <= height/2+50]#yonige
  df_objects_performer = df_objects_performer.sort_values('xmin')
  df_objects_performer = df_objects_performer.reset_index(drop=True)
  #重なっている部分を削除
  drop_list = []
  for i in range(len(df_objects_performer)):
    if i != 0:
      if not  df_objects_performer.xmax[i-1] < df_objects_performer.xmin[i] or df_objects_performer.xmax[i] < df_objects_performer.xmin[i-1]:
        drop_list.append(i)
  df_objects_performer = df_objects_performer.drop(df_objects_performer.index[drop_list])
  df_objects_performer = df_objects_performer.reset_index(drop=True)

  #上手く検知できなかった演者を補完（前のフレーム情報から）
  if df_objects_performer_before.empty:#前のフレームの演者データフレームが空（最初のフレーム）ならパートidを割り当てる
    for i in range(len(df_objects_performer)):
      df_objects_performer.loc[i, "part"] = "player" + str(random.randint(0, 9)) + str(random.randint(0, 9)) + str(random.randint(0, 9)) + str(random.randint(0, 9)) + str(random.randint(0, 9))
  else:
    df_objects_performer_before_copy = df_objects_performer_before.copy()
    #前のフレーム情報に上手く検知できなかった演者だけ残す
    for i in range(len(df_objects_performer)):
      df_objects_performer_before_copy = df_objects_performer_before_copy.reset_index(drop=True)
      drop_list = []
      for j in range(len(df_objects_performer_before_copy)):
        if df_objects_performer.xmax[i] < df_objects_performer_before_copy.xmin[j] or df_objects_performer_before_copy.xmax[j] < df_objects_performer.xmin[i]:
          continue
        if df_objects_performer.ymax[i] < df_objects_performer_before_copy.ymin[j] or df_objects_performer_before_copy.ymax[j] < df_objects_performer.ymin[i]:
          continue
        df_objects_performer.loc[i, "part"] = df_objects_performer_before_copy.part[j]
        drop_list.append(j)
      df_objects_performer_before_copy = df_objects_performer_before_copy.drop(df_objects_performer_before_copy.index[drop_list])
    df_objects_performer_before_copy = df_objects_performer_before_copy.reset_index(drop=True)
    #連続して検出されなかった人物（誤検出）を削除
    for i in range(len(df_objects_performer_before_copy)):
      df_objects_performer_before_copy.loc[i, "miss_count"] = df_objects_performer_before_copy.miss_count[i] + 1
      if df_objects_performer_before_copy.miss_count[i] > 30:
        df_objects_performer_before_copy = df_objects_performer_before_copy.drop(df_objects_performer_before_copy.index[i])
    df_objects_performer = pd.concat([df_objects_performer, df_objects_performer_before_copy])
    df_objects_performer = df_objects_performer.reset_index(drop=True)
    #追加で検出されたものにパートidを割り当てる
    for i in range(len(df_objects_performer)):
      if df_objects_performer.part[i] == "empty":
        df_objects_performer.loc[i, "part"] = "player" + str(random.randint(0, 9)) + str(random.randint(0, 9)) + str(random.randint(0, 9)) + str(random.randint(0, 9)) + str(random.randint(0, 9))
    df_objects_performer = df_objects_performer.sort_values('xmin')
    df_objects_performer = df_objects_performer.reset_index(drop=True)

  return df_objects_performer

#検出した演者を囲む
def box_performer(frame, df_objects_perfromer): #（フレーム画像、演者のデータフレーム）-> フレーム画像
  height, widht, _ = frame.shape
  for i in range(len(df_objects_perfromer)):
    x1 = df_objects_perfromer.xmin[i]
    y1 = df_objects_perfromer.ymin[i]
    x2 = df_objects_perfromer.xmax[i]
    y2 = df_objects_perfromer.ymax[i]
    left_pt = (int(x1), int(y1))
    right_pt = (int(x2), int(y2))
    cv2.rectangle(frame, left_pt, right_pt, (0,255,0), 2)
    part = df_objects_perfromer.part[i]
    cv2.putText(frame, text=part, org=(int(x1), int(y1)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)
  return frame


#特徴点検出
def feature_extraction(frame, frame_before, df_objects_performer, df_objects_performer_before,  df_performer_movement, draw_flag=False):
  height, widht, _ = frame.shape

  if frame_before is None:
    return frame

  #今のフレームと前のフレームで重複する演者だけにする
  df_objects_performer_sort = df_objects_performer[df_objects_performer["part"].isin(df_objects_performer_before["part"])]
  df_objects_performer_sort = df_objects_performer_sort.reset_index(drop=True)
  df_objects_performer_before_sort = df_objects_performer_before[df_objects_performer_before["part"].isin(df_objects_performer["part"])]
  df_objects_performer_before_sort = df_objects_performer_before_sort.reset_index(drop=True)
  for i in range(len(df_objects_performer_sort)):
    if not df_objects_performer_sort.part[i] in df_performer_movement.index:#追加で検出された演者を登録する
      df_append = pd.DataFrame({'xmin': [df_objects_performer_sort.xmin[i]], 'ymin': [df_objects_performer_sort.ymin[i]],\
                                'xmax': [df_objects_performer_sort.xmax[i]], 'ymax': [df_objects_performer_sort.ymax[i]],\
                                'movement' : [0], 'visual_express' : [-1]}, index=[df_objects_performer_sort.part[i]])
      df_performer_movement = pd.concat([df_performer_movement, df_append])
    else :#df_performer_movementの登録内容を更新
      movement_xmin = df_performer_movement.loc[df_objects_performer_sort.part[i], "xmin"]
      movement_ymin = df_performer_movement.loc[df_objects_performer_sort.part[i], "ymin"]
      movement_xmax = df_performer_movement.loc[df_objects_performer_sort.part[i], "xmax"]
      movement_ymax = df_performer_movement.loc[df_objects_performer_sort.part[i], "ymax"]
      sort_xmin = df_objects_performer_sort.loc[i, "xmin"]
      sort_ymin = df_objects_performer_sort.loc[i, "ymin"]
      sort_xmax = df_objects_performer_sort.loc[i, "xmax"]
      sort_ymax = df_objects_performer_sort.loc[i, "ymax"]
      df_performer_movement.loc[df_objects_performer_sort.part[i], "xmin"] = min(movement_xmin, sort_xmin)
      df_performer_movement.loc[df_objects_performer_sort.part[i], "ymin"] = min(movement_ymin, sort_ymin)
      df_performer_movement.loc[df_objects_performer_sort.part[i], "xmax"] = max(movement_xmax, sort_xmax)
      df_performer_movement.loc[df_objects_performer_sort.part[i], "ymax"] = max(movement_ymax, sort_ymax)

  #検出された演者の周りで、フレーム間で共通する特徴点を検出し移動量から動作量を推定
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  frame_gray_before = cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY)
  for i in range(len(df_objects_performer_sort)):
    trim_xmin = min(df_objects_performer_sort.xmin[i], df_objects_performer_before_sort.xmin[i])
    trim_xmax = max(df_objects_performer_sort.xmax[i], df_objects_performer_before_sort.xmax[i])
    trim_ymin = min(df_objects_performer_sort.ymin[i], df_objects_performer_before_sort.ymin[i])
    trim_ymax = max(df_objects_performer_sort.ymax[i], df_objects_performer_before_sort.ymax[i])
    gray1 = frame_gray_before[int(trim_ymin) : int(trim_ymax), int(trim_xmin): int(trim_xmax)]
    gray2 = frame_gray[int(trim_ymin) : int(trim_ymax), int(trim_xmin): int(trim_xmax)]
    ST_param = dict( maxCorners = 50, qualityLevel = 0.1, minDistance = 4, blockSize = 7)
    ft1 = cv2.goodFeaturesToTrack(gray1, mask = None, **ST_param )
    LK_param = dict( winSize  = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    ft2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, ft1, None, **LK_param )
    if status is not None:#オプティカルフローが計算できた場合
      #オプティカルフローを検出した特徴点を取得
      g1 = ft1[status == 1]#1フレーム目
      g2 = ft2[status == 1]#2フレーム目
      #特徴点とオプティカルフローを描画する
      for j, (pt2, pt1) in enumerate(zip(g2, g1)):
        #1フレーム目の特徴点座標
        x1, y1 = pt1.ravel()
        #2フレーム目の特徴点座標
        x2, y2 = pt2.ravel()
        #比率を調整
        x1 = int(trim_xmin) + x1
        y1 = int(trim_ymin) + y1
        x2 = int(trim_xmin) + x2
        y2 = int(trim_ymin) + y2
        #現フレームに特徴点を描画
        if draw_flag:
          frame = cv2.circle(frame, (int(x2), int(y2)), 3,  [0, 0, 200], -1)
        #動作量をデータフレームに追加する
        p = np.array([x1, y1])
        q = np.array([x2, y2])
        dist = np.linalg.norm(p - q)
        df_performer_movement.loc[df_objects_performer_sort.part[i], "movement"] = df_performer_movement.movement[df_objects_performer_sort.part[i]] + dist
  if draw_flag:
    return frame, df_performer_movement
  else:
    return df_performer_movement