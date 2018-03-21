from pynput import keyboard
import time
import pyaudio
import wave
from sched import scheduler
import sys
import scipy.io.wavfile as wav
from deepspeech.model import Model as dsmodel
import os

CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'
base_dir = os.path.abspath("dspeech/model_files/")
p = pyaudio.PyAudio()

def reset_audio():
    global p
    p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
    frames.append(in_data)
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
    print("Recording", int(abs(time_info['input_buffer_adc_time'] * 100)) * ".")
    return (in_data, pyaudio.paContinue)

class MyListener(keyboard.Listener):
    def __init__(self):
        super(MyListener, self).__init__(self.on_press, self.on_release)
        self.key_pressed = None
        self.AUDIO_FILE = 'output/output_' + str(int(time.time())) + '.wav'
        self.wf = wave.open(self.AUDIO_FILE, 'wb')
        self.wf.setnchannels(CHANNELS)
        self.wf.setsampwidth(p.get_sample_size(FORMAT))
        self.wf.setframerate(RATE)

    def on_press(self, key):
        if key.char == 'r':
            self.key_pressed = True
        return True

    def on_release(self, key):
        if key.char == 'r':
            self.key_pressed = False
        return True

def recorder():
    global started, p, stream, frames

    if listener.key_pressed and not started:
        # Start the recording
        try:
            stream = p.open(format=FORMAT,
                             channels=CHANNELS,
                             rate=RATE,
                             input=True,
                             frames_per_buffer=CHUNK,
                             stream_callback = callback)
            print("Stream active:", stream.is_active())
            started = True
        except:
            raise

    elif not listener.key_pressed and started:
        print("Stop recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        listener.wf.writeframes(b''.join(frames))
        listener.wf.close()

        print ("Speech recording complete..")
        return
    # Reschedule the recorder function in 100 ms.
    task.enter(0.1, 1, recorder, ())

class STT:
    def __init__(self, buffer_file=base_dir+'/output_graph.pb', nfeatures=26, ncontext=9, abc_file=base_dir+'/alphabet.txt',
    beam_width=500):
        self.buffer_file = buffer_file
        self.nfeatures = nfeatures
        self.ncontext = ncontext
        self.abc_file = abc_file
        self.beam_width = beam_width

    def get_text(self, wav_file):
        fs, audio = wav.read(wav_file)
        dsm = dsmodel(self.buffer_file, self.nfeatures, self.ncontext, self.abc_file, self.beam_width)
        text = dsm.stt(audio, fs)
        return text

if __name__ == '__main__':
    stt = STT()
    while True:
        reset_audio()
        listener = MyListener()
        listener.start()
        frames = []
        started = False
        stream = None
        print ("Press and hold the 'r' key to begin recording")
        print ("Release the 'r' key to end recording")
        task = scheduler(time.time, time.sleep)
        task.enter(0.1, 1, recorder, ())
        task.run()
        print(stt.get_text(listener.AUDIO_FILE))
