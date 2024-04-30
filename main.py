#!/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import librosa
import numpy as np
import scipy
import argparse

import warnings
warnings.filterwarnings('ignore')


def load_audio(filename):
    return librosa.load(filename, sr=44100)

def find_offset(audio1, audio2):
    # 相関を計算
    # correlation = np.correlate(audio1, audio2, mode='full')  # numpyは遅すぎた
    correlation = scipy.signal.correlate(audio1, audio2, mode='full', method="fft")
    # 最大相関を持つインデックスを見つける
    lag = np.argmax(correlation) - len(audio2) + 1
    return lag

# 長さを揃えるためにパディング挿入
def insert_padding(audio1, audio2):
  if len(audio1) > len(audio2):
      audio2 = np.pad(audio2, (0, len(audio1) - len(audio2)), 'constant')
  else:
      audio1 = np.pad(audio1, (0, len(audio2) - len(audio1)), 'constant')
  return audio1, audio2

def format_time(frame, sample_rate) -> str:
    seconds = abs(frame) / sample_rate
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}"

# ffprobeのshow_streamsを取得
def get_stream_info(video_path):
    """指定されたビデオファイルの解像度とピクセルフォーマットを取得する"""
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_streams',
        '-of', 'json', 
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception("ffprobe encountered an error: " + result.stderr.decode())
    
    video_info = json.loads(result.stdout)
    if 'streams' in video_info and len(video_info['streams']) > 0:
        videos = [s for s in video_info['streams'] if s["codec_type"] == "video"]
        video = videos[0] if len(videos) !=0 else None
        audios = [s for s in video_info['streams'] if s["codec_type"] == "audio"]
        audio = audios[0] if len(audios) !=0 else None
        return video_info['streams'], video, audio
    else:
        raise Exception("No video stream found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="video file filepath")
    parser.add_argument("audio", help="audio file filepath")
    parser.add_argument("--check", help="audio file filepath", action="store_true")
    args = parser.parse_args()

    dest_dir = os.path.join(os.path.dirname(args.video), "cut")
    streams, video, audio = get_stream_info(args.video)
    
    if not args.check:
        os.makedirs(dest_dir, exist_ok=True)
        assert video["codec_name"] == "h264", "only support h264 for video codes"
        assert audio["codec_name"] == "aac", "only support aac for audio codes"

    # 音声をロード
    audio1, sr = load_audio(args.video)
    audio2, sr = load_audio(args.audio)
    raw_len = len(audio2)

    audio1, audio2 = insert_padding(audio1, audio2)

    # FFTして相関を調べる
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html#scipy.signal.correlate
    correlation = scipy.signal.correlate(audio1, audio2, mode='full', method="fft")
    # 最大相関を持つインデックスを見つける
    lag = np.argmax(correlation) - len(audio2) + 1

    print("Offset:", lag, sr)

    ss = format_time(lag, sr)
    t = format_time(raw_len, sr)
    print(ss)
    print(t)

    if args.check:
        sys.exit(0)

    seconds = abs(lag) / sr
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    video_name, ext = os.path.splitext(os.path.basename(args.video))
    output_name = f"{video_name}_{int(h):02}-{int(m):02}-{s:06.3f}{ext}"
    print(output_name)

    if lag < 0:
        # ラグがマイナス値の場合は、黒い映像を生成して先頭にpaddingとして挿入する
        print("backward offset mode")
        empty_video_path = os.path.join(dest_dir, f"{video_name}_empty{ext}")
        out_video_path = os.path.join(dest_dir, f"{video_name}_out{ext}")

        ffmpeg_args = [
            "ffmpeg",
            "-i", args.video,
            "-f", "lavfi",
            "-i", "aevalsrc=0|0:c=stereo",
            "-map", "0:0",
            "-map", "1:0",
            "-filter_complex", "tpad=start_duration=10000",
            "-t", ss,
            "-c:v", str(video["codec_name"]),
            "-profile:v", str(video["profile"]),
            "-level:v", str(video["level"]),
            "-pix_fmt", str(video["pix_fmt"]),
            empty_video_path,
            "-y",
        ]
        print(" ".join(ffmpeg_args))
        subprocess.run(ffmpeg_args)

        plist = [
            os.path.abspath(empty_video_path),
            os.path.abspath(args.video),
        ]
        concat_plist = "\n".join([f"file '{x}'" for x in plist])
        concat_plist += "\n"

        with tempfile.NamedTemporaryFile(mode="w+",suffix=".txt",encoding='utf8') as temp_file:
            temp_file.write(concat_plist)
            temp_file.flush()

            ffmpeg_args = [
                "ffmpeg",
                "-safe", "0",
                "-f", "concat",
                "-i", temp_file.name,
                "-t", t,
                "-c", "copy",
                "-movflags", "faststart",
                os.path.join(dest_dir, output_name),
                "-y",
            ]
            print(" ".join(ffmpeg_args))
            subprocess.run(ffmpeg_args)
            
        os.remove(empty_video_path)

    else:
        ffmpeg_args = [
            "ffmpeg",
            "-ss", ss,
            "-i", args.video,
            "-t", t,
            "-c", "copy",
            "-movflags", "faststart",
            os.path.join(dest_dir, output_name),
            "-y",
        ]
        print(" ".join(ffmpeg_args))
        subprocess.run(ffmpeg_args)

