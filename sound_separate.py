#!echo n | conda install ffmpeg
from genericpath import isfile
import io
from pathlib import Path
import os
import select
from shutil import rmtree
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.utils import db_to_float, ratio_to_db

#音源分離 - run_meducs(input:str, output:str)
class Meducs:
    # Customize the following options!
    model = "mdx_extra_q"
    extensions = ["mp3", "wav", "ogg", "flac"]  # we will look for all those file types.
    #two_stems = None   # only separate one stems from the rest, for instance
    two_stems = "vocals"

    # Options for the output audio.
    mp3 = False
    mp3_rate = 320
    float32 = False  # output as float 32 wavs, unsused if 'mp3' is True.
    int24 = False    # output as int24 wavs, unused if 'mp3' is True.
    # You cannot set both `float32 = True` and `int24 = True` !!

    in_path = './demucs/'
    out_path = './demucs_separated/'

    def __init__(self, *
                , model=None
                , extentions=None
                , two_stems=None
                , mp3=None, mp3_rate=None
                , float32=None, int24=None
                , in_path=None, out_path=None
                ):
        self.model = model or self.model
        self.extensions = extentions or self.extensions
        self.two_stems = two_stems or self.two_stems
        self.mp3 = mp3 or self.mp3
        self.mp3_rate = mp3_rate or self.mp3_rate
        self.float32 = float32 or self.float32
        self.in_path = in_path or self.in_path
        self.out_path = out_path or self.out_path

    def find_files(self, in_path):
        out = []
        for file in Path(in_path).iterdir():
            if file.suffix.lower().lstrip(".") in self.extensions:
                out.append(file)
        return out

    @staticmethod
    def copy_process_streams(process: sp.Popen):
        def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
            assert stream is not None
            if isinstance(stream, io.BufferedIOBase):
                stream = stream.raw
            return stream

        p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
        stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
            p_stdout.fileno(): (p_stdout, sys.stdout),
            p_stderr.fileno(): (p_stderr, sys.stderr),
        }
        fds = list(stream_by_fd.keys())

        while fds:
            # `select` syscall will wait until one of the file descriptors has content.
            ready, _, _ = select.select(fds, [], [])
            for fd in ready:
                p_stream, std = stream_by_fd[fd]
                raw_buf = p_stream.read(2 ** 16)
                if not raw_buf:
                    fds.remove(fd)
                    continue
                buf = raw_buf.decode()
                std.write(buf)
                std.flush()

    def separate(self, inp=None, outp=None, *
                , model=None
                , two_stems=None
                , mp3=None, mp3_rate=None
                , float32=None, int24=None):
        inp = inp or self.in_path
        outp = outp or self.out_path
        model = model or self.model
        two_stems = two_stems or self.two_stems
        mp3 = mp3 or self.mp3
        mp3_rate = mp3_rate or self.mp3_rate
        float32 = float32 or self.float32
        int24 = int24 or self.int24

        cmd = ["python3", "-m", "demucs.separate", "-o", str(outp), "-n", model]
        if mp3:
            cmd += ["--mp3", f"--mp3-bitrate={mp3_rate}"]
        if float32:
            cmd += ["--float32"]
        if int24:
            cmd += ["--int24"]
        if two_stems is not None:
            cmd += [f"--two-stems={two_stems}"]
        p_inp = Path(inp)
        if p_inp.is_file() and p_inp.exists() and p_inp.suffix.lstrip(".").lower() in self.extensions:
            files = [inp]
        else:
            print(f"No valid audio files in {inp}")
            return
        print("Going to separate the files:")
        print('\n'.join(files))
        print("With command: ", " ".join(cmd))

        p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
        self.copy_process_streams(p)
        p.wait()
        if p.returncode != 0:
            print("Command failed, something went wrong.")

        return str(Path(outp).joinpath(model, p_inp.stem))

def run_meducs(input:str, output:str): -> None
  meducs = Meducs()
  meducs.separate(input, output)
  print("meducs success!")

#ボーカル音源の無音部分をカットし分割で保存
def vocal_split(input:str, output:str): -> None
  ##音源読み込み、音量の最大値/5以下のマスクを作成
  y, sr = librosa.load(input, sr=None)#音源読み込み
  rms = librosa.feature.rms(y=y) #音量の計算
  times = librosa.times_like(rms, sr=sr) #時間軸の生成
  times_y = np.arange(0,len(y)) / sr #y用時間軸の生成
  dBref = 2e-5 #1Pa=94dBとなるための定数
  db = 20 * np.log10(rms/dBref) #音量→dB変換
  array_db = db[0].tolist()#デシベルをリスト型に変換
  mask = [db > max(array_db)/5 for db in array_db]#array_dbのうち、最大値/5以下のマスクを作成

  ##無音部分の波形をゼロにする
  sound = AudioSegment.from_file(input, format="wav")#音源読み込み
  stock_wav = [] #切り取った音声ファイルと音声の有無 [[file, stock_flag]]
  stock_s = 0 #切り取る音声の始点
  stock_flag = False #音声の有無
  file_count = 0 #音声のあるファイル数
  for i in range(len(array_db)):
    #音声の有無が切り替わったら
    if not mask[i] == stock_flag:
      stock_wav.append([sound[stock_s*1000 : times[i]*1000], stock_flag])#sock_wavに音声ファイルとstock_flagを追加
      #音声有りだったら
      if stock_flag == True:
        wav_vocal = sound[stock_s*1000 : times[i]*1000] #音声有りの部分を切り取る
        time = wav_vocal.duration_seconds #切り取ったファイルの長さ

        #音量が小さすぎたら無視する
        if wav_vocal.rms < 10:
          #次のループのために更新
          stock_s = times[i]
          stock_flag = mask[i]
          continue

        #1秒以下のファイルは無視する
        if time <= 1:
          #次のループのために更新
          stock_s = times[i]
          stock_flag = mask[i]
          continue

        #音声有りファイルをフォルダに保存
        wav_vocal_filename = "wav_vocal_" + str(file_count) + ".wav"
        wav_vocal.export(output + "/" + wav_vocal_filename, format="wav")
        file_count = file_count + 1
        print(wav_vocal_filename + "[ start : " + str(stock_s) + "s" + " | | | " + "end : " + str(times[i]) + "s ]")
      #次のループのために更新
      stock_s = times[i]
      stock_flag = mask[i]
    #最後のファイルを切り取る
    if i == len(array_db)-1:
      stock_wav.append([sound[stock_s*1000 : times[i]*1000], stock_flag])
      if stock_flag == True:
        wav_vocal = sound[stock_s*1000 : times[i]*1000] #音声有りの部分を切り取る
        time = wav_vocal.duration_seconds #切り取ったファイルの長さ

        #音量が小さすぎたら無視する
        if wav_vocal.rms < 10:
          #次のループのために更新
          stock_s = times[i]
          stock_flag = mask[i]
          continue

        #1秒以下のファイルは無視する
        if time <= 1:
          #次のループのために更新
          stock_s = times[i]
          stock_flag = mask[i]
          continue

        #音声有りファイルをフォルダに保存
        wav_vocal_filename = "wav_vocal_" + str(file_count) + ".wav"
        wav_vocal.export(output + "/" + wav_vocal_filename, format="wav")
        file_count = file_count + 1
        print(wav_vocal_filename + "[ start : " + str(stock_s) + "s" + " | | | " + "end : " + str(times[i]) + "s ]")