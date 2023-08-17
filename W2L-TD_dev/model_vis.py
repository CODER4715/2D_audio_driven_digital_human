import netron
import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
import pylab
import onnx

def nt():
    model_file = 'WTLDT.onnx'
    # onnx_model = onnx.load("WTLDT_fp16.onnx")
    # onnx.checker.check_model(onnx_model)
    netron.start(model_file)

def mel_spec():
    y, sr = librosa.load('media/cn1.mp3')
    whale_song, _ = librosa.effects.trim(y)
    S = librosa.feature.melspectrogram(whale_song, sr=sr, n_fft=800, hop_length=200, n_mels=80)
    S_DB = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(S_DB, sr=sr, hop_length=200, x_axis='time', y_axis='mel')

    plt.colorbar(format='%+2.0f dB');


def mel2():

    # plt.use('Agg')  # No pictures displayed

    sig, fs = librosa.load('media/cn1.mp3')
    # make pictures name
    save_path = 'test.jpg'

    pylab.axis('off')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()

if __name__=='__main__':
    nt()
    # mel_spec()
    # mel2()