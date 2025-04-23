import wave
import sys

if len(sys.argv) > 1:
    wav_path = sys.argv[1]
    print("first arg:", wav_path)
else:
    print("usage: python audio_test.py [file_path].")
    sys.exit()


wav_file = wave.open(wav_path, 'r')
params = wav_file.getparams()
nchannels, sampwidth, framerate, nframes, comptype, compname = params
frames = wav_file.readframes(nframes)
print(frames)
wav_file.close()
