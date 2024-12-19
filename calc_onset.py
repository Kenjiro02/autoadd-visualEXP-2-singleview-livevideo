import numpy as np
import librosa
import matplotlib.pyplot as plt

#音量を考慮したオンセット検知のタイミング返す
def onset_consider_volume(audio: str, strength: float = 1) -> list: #（オーディオファイル名、オンセット強度）-> オンセット配列
  #音声の要素を抽出
  y, sr = librosa.load(audio, sr=16000, mono=True) #音声読み込み
  rms = librosa.feature.rms(y=y) #音量の計算
  times = librosa.times_like(rms, sr=sr) #時間軸の生成
  times_y = np.arange(0,len(y)) / sr #y用時間軸の生成
  dBref = 2e-5 #1Pa=94dBとなるための定数
  db = 20 * np.log10(rms/dBref) #音量→dB変換
  #音量を考慮したオンセット検知
  res_n = np.polyfit(times, rms[0],8) #n次式にする
  rms_n = np.poly1d(res_n)(times) #n次式の結果の配列
  rms_n_mean = np.mean(rms_n)
  rms_n = (rms_n - rms_n_mean)
  ad_ratio = 1 - rms_n  / np.max(np.abs(rms_n))
  rms[0] = np.multiply(rms[0], ad_ratio)#音量✖️比率
  onset_envelope = librosa.onset.onset_strength(S=rms, sr=sr)**(2*strength) #オンセット強度を計算
  onset_times = librosa.times_like(onset_envelope, sr=sr) #オンセットの時間軸
  onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr) #オンセット検知
  onset_timing = []
  ad_time = 0
  #近すぎるオンセットをまとめ、配列に入れる
  for i in range(len(onset_times[onset_frames])):
    if i == 0:
      ad_time = onset_times[onset_frames][i]
      onset_timing.append(ad_time)
      continue
    if onset_times[onset_frames][i] - ad_time > 1:
      ad_time = onset_times[onset_frames][i]
      onset_timing.append(ad_time)
  return onset_timing

def add_another_timing(onset_timing:list, another_list:list) -> list:
  #タイミングを加える
  onset_timing = sorted([*onset_timing, *another_list])
  #近すぎるオンセットをまとめる
  visual_timing = []
  ad_time = 0
  for i in range(len(onset_timing)):
    if i == 0:
      ad_time = onset_timing[i]
      visual_timing.append(ad_time)
      continue
    if onset_timing[i] - ad_time > 1:
      ad_time = onset_timing[i]
      visual_timing.append(ad_time)
  return visual_timing

def draw_onset_graph(audio:str, strength: float = 1, another_list: list = []) -> None:
  #音声の要素を抽出
  y, sr = librosa.load(audio, sr=16000, mono=True) #音声読み込み
  rms = librosa.feature.rms(y=y) #音量の計算
  times = librosa.times_like(rms, sr=sr) #時間軸の生成
  times_y = np.arange(0,len(y)) / sr #y用時間軸の生成
  dBref = 2e-5 #1Pa=94dBとなるための定数
  db = 20 * np.log10(rms/dBref) #音量→dB変換
  #音量を考慮したオンセット検知
  res_n=np.polyfit(times, rms[0],8) #n次式にする
  rms_n = np.poly1d(res_n)(times) #n次式の結果の配列
  ad_ratio = 2 - (rms_n / max(rms_n))#音量を考慮した比率配列
  rms[0] = np.multiply(rms[0], ad_ratio)#音量✖️比率
  onset_envelope = librosa.onset.onset_strength(S=rms, sr=sr)**(2*strength) #オンセット強度を計算
  onset_times = librosa.times_like(onset_envelope, sr=sr) #オンセットの時間軸
  onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr) #オンセット検知
  onset_timing = []
  ad_time = 0
  #近すぎるオンセットをまとめ、配列に入れる
  for i in range(len(onset_times[onset_frames])):
    if i == 0:
      ad_time = onset_times[onset_frames][i]
      onset_timing.append(ad_time)
      continue
    if onset_times[onset_frames][i] - ad_time > 1:
      ad_time = onset_times[onset_frames][i]
      onset_timing.append(ad_time)
  #グラフ描画
  plt.figure(figsize=(12, 8))
  if len(another_list) != 0:
    plt.title('onset_timing + alpha')
  elif len(another_list) == 0:
    plt.title('onset_timing')
  plt.subplot(2, 1, 2)
  plt.plot(onset_times, onset_envelope, label='Onset strength')
  plt.vlines(onset_timing, 0, onset_envelope.max(), color='r', alpha=0.9,
            linestyle='--', label='Onsets')
  if len(another_list) != 0:
    plt.vlines(another_list, 0, onset_envelope.max(), color='g', alpha=0.9, linestyle='--', label='alpha')
  plt.xlabel('Time')
  plt.ylabel('Onset strength')
  plt.legend(frameon=True, framealpha=0.75)
  plt.tight_layout()
  plt.show()