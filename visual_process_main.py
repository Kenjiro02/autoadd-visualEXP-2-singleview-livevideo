import cv2
import numpy as np
import pandas as pd
import time
import sys
import ffmpeg
from calc_onset import onset_consider_volume
from performer_detection import yolo_detection, feature_extraction, box_performer
from visual_expression import zoom_frames, radial_frames, split_frames, make_gradation
"""
メイン処理
Args:
  input_video(str) : 入力ビデオのパス
  output_video(str) : 出力ビデオのパス
Output:
  .mp4 : 出力ビデオ
Note:
  以下のファイルを作成してください
  video audio stock output
使用例:
  python visual_process_main.py video/yumeno_test.mp4 output/yumeno_test_output.mp4
"""

def main(input_video:str, output_video:str):
  print("args : " + input_video + ", " + output_video)

  # エラー処理
  if not input_video.endswith('.mp4'):
    raise ValueError("ファイルの拡張子が.mp4ではありません")
  if not output_video.endswith('.mp4'):
    raise ValueError("ファイルの拡張子が.mp4ではありません")
  
  # ----default_setting----
  # 入力動画を読み取る
  cap = cv2.VideoCapture(input_video)# 動画
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))# 横幅
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))# 縦幅
  size = (width, height)# (横幅, 縦幅)
  fps = float(cap.get(cv2.CAP_PROP_FPS))# fps
  totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))# フレーム総数

  # 出力フォーマット
  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  out_video = cv2.VideoWriter("stock/out_video.mp4", fourcc, fps, (width, height))# 出力動画

  # 入力動画から音声を抽出
  stream = ffmpeg.input(input_video) # 入力
  audio_file = "audio/" + input_video.split("/")[-1].replace(".mp4", "") + ".wav"
  stream = ffmpeg.output(stream, audio_file) # 出力
  ffmpeg.run(stream, quiet=True, overwrite_output=True)# 実行

  # オンセット計算
  switch_visual_timing = onset_consider_volume(audio_file)
  print("onset calc finished!!")

  # 変数定義
  frame_count = 0
  start_time = time.time()
  frames = [] # フレーム格納配列
  times = [] # 時間格納配列
  write_time_before = 0 # フレームブロックの書き出しタイミング
  # 現在のそれぞれの演者の動作量
  df_performer_movement = pd.DataFrame([], columns=["xmin", "ymin", "xmax", "ymax", "movement", "visual_express"])
  # 前のフレーム情報
  frame_before = None # 1つ前のフレーム
  df_objects_performer_before = pd.DataFrame([], [])# 1つ前の上手く人物認識できたフレーム情報
  df_performer_movement_before = pd.DataFrame([], columns=["xmin", "ymin", "xmax", "ymax", "movement", "visual_express"])# １つ前のフレームでのそれぞれの演者の動作量

  # ラディアルブラーでのマスク作成
  mask = make_gradation(width, height)
  mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  
  # メイン処理
  while True:
    # 画像を取得
    ret, frame = cap.read()
    frame_original = frame
    # 再生が終了したらループを抜ける
    if ret == False:
      write_frames(frames, times, out_video) # フレームの書き込み
      break
    #-------処理--------
    #演者認識
    df_objects_performer = yolo_detection(frame, df_objects_performer_before)
    frame = box_performer(frame, df_objects_performer)
    #動体検知
    #df_performer_movement = feature_extraction(frame, frame_before, df_objects_performer, df_objects_performer_before, df_performer_movement)
    #動体検知による映像表現、logのリセット
    if len(switch_visual_timing) > 0:
      # オンセットタイミングを超えたら映像表現を付与
      if switch_visual_timing[0] < frame_count/fps:
        if df_performer_movement_before.empty:#最初のフレームではフレーム情報をコピー
          first_block = True
          #df_performer_movement_before = df_performer_movement.copy()
        else:
          """
          #映像表現
          for row in df_performer_movement.itertuples():
            if row.Index in df_performer_movement_before.index:
              #演者それぞれの1フレームごとの平均動作量
              movement = row.movement
              movement_before = df_performer_movement_before.movement[row.Index]
              movement_per_frame = movement / len(frames)
              movement_before_per_frame = movement_before / len(frames)
              movement_ratio_per_frame = movement_per_frame / movement_before_per_frame
              if movement_ratio_per_frame > 2.0:
                df_performer_movement.loc[row.Index, "visual_express"] = int(2)
              elif movement_ratio_per_frame > 1.2:
                df_performer_movement.loc[row.Index, "visual_express"] = int(1)
              else:
                df_performer_movement.loc[row.Index, "visual_express"] = int(0)
            else:
              df_performer_movement.loc[row.Index, "visual_express"] = int(0)
          if not df_performer_movement[df_performer_movement['visual_express'] == 2].empty:
            df_performer_movement_2 = df_performer_movement[df_performer_movement['visual_express'] == 2]
            df_performer_movement_2 = df_performer_movement_2.sort_values('movement')
            df_performer_movement_2 = df_performer_movement_2.reset_index()
            if len(df_performer_movement_2) == len(df_performer_movement):#演者全員の動作量が(2)
              #画面分割
              frames = split_frames(frames, df_performer_movement)
              print("split")
            else:#演者一人でも動作量が(2)
              line_xmin = df_performer_movement_2.loc[0, "xmin"]
              line_xmax = df_performer_movement_2.loc[0, "xmax"]
              line_ymin = df_performer_movement_2.loc[0, "ymin"]
              line_ymax = df_performer_movement_2.loc[0, "ymax"]
              #方向線
              frames = zoom_frames(frames, line_xmin, line_xmax, line_ymin, line_ymax)
              frames = radial_frames(frames, mask)
              print("line")
          elif not df_performer_movement[df_performer_movement['visual_express'] == 1].empty:#演者一人でも動作量が(1)
            df_performer_movement_1 = df_performer_movement[df_performer_movement['visual_express'] == 1]
            df_performer_movement_1 = df_performer_movement_1.sort_values('movement')
            df_performer_movement_1 = df_performer_movement_1.reset_index()
            zoom_xmin = df_performer_movement_1.loc[0, "xmin"]
            zoom_xmax = df_performer_movement_1.loc[0, "xmax"]
            zoom_ymin = df_performer_movement_1.loc[0, "ymin"]
            zoom_ymax = df_performer_movement_1.loc[0, "ymax"]
            #ズーム
            frames = zoom_frames(frames, zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax)
            print("zoom")
          elif len(df_performer_movement[df_performer_movement['visual_express'] == 0]) == len(df_performer_movement):#演者全員の動作量が(0)
            #映像表現無し
            print("None")
          df_performer_movement_before = df_performer_movement.copy()
          df_performer_movement = pd.DataFrame([], columns=["xmin", "ymin", "xmax", "ymax", "movement"])
          """
        switch_visual_timing.pop(0)#オンセットタイミングの先頭を削除
        # 前回の書き出しから間がある場合のエラー処理
        if times[0] - write_time_before > 1:
          raise ValueError("frame export ERROR")
        write_frames(frames, times, out_video) # フレームの書き込み
        write_time_before = times[-1]
        frames.clear()
        times.clear()

    
    # フレーム情報の保存
    frames.append(frame)
    times.append(frame_count/fps)
    df_objects_performer_before = df_objects_performer
    #--------------------

    print(frame_count/totalFrames*100)
    frame_before = frame_original
    frame_count+=1

  cap.release()
  out_video.release()
  video_set_audio("stock/out_video.mp4", audio_file, output_video)
  print("success!")
  
# フレームの書き込み
def write_frames(frames: list, times: list, video_writer: cv2.VideoWriter) -> None:
  for i in range(len(frames)):
    frame = frames[i]
    time = times[i]
    video_writer.write(frame)
  print("write frame finished!!")

# 動画と音声を結合
def video_set_audio(input_video: str, input_audio: str, output_video: str) -> None:
  stream_v = ffmpeg.input(input_video) # 入力動画
  stream_a = ffmpeg.input(input_audio) # 入力音声
  stream_out = ffmpeg.output(stream_v, stream_a, output_video, vcodec="copy", acodec="aac") # 出力動画
  ffmpeg.run(stream_out, quiet=True, overwrite_output=True) # 実行

if __name__ == "__main__":
  input_video = sys.argv[1]
  output_video = sys.argv[2]
  main(input_video, output_video)