import whisper
import speech_recognition as sr
from queue import Queue
import os
import numpy as np
import torch
import ffmpeg
import deep_translator
from deep_translator import GoogleTranslator
import io
from io import BytesIO
from tempfile import NamedTemporaryFile

"""
load the AI model
Set up my speech recognizer
for korean to english just use the translate function
for english to korean might use bard

continuously record the mic
add audio data to a list when audio is being picked up
transcribe in 30 second windows
"""

data_queue = Queue()

def load_AI():
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("large", device=DEVICE)

def load_mic():
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    source = sr.Microphone(sample_rate=16000)
    with source:
        recorder.adjust_for_ambient_noise(source)
    return recorder, source

def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_wav_data()
        data_queue.put(io.BytesIO(data))

def main():
    # bg thread to constantly record audio
    model = load_AI()
    recorder, source = load_mic()
    recorder.listen_in_background(source, record_callback, phrase_time_limit=2)
    print("Model loaded.\n")
    while(True):
        # we have a bg thread that is constantly putting audio files into the data queue in the form of bits
        # they put it in 30 second chunks
        # look at the queue and transcribe it one by one
        if not data_queue.empty():
            wav_data = data_queue.get()
            tmp_file = NamedTemporaryFile().name
            with open(tmp_file, 'w+b') as f:
                    f.write(wav_data.read())
            transcription = model.transcribe(tmp_file, fp16=False)
            text = transcription["text"]
            spoken_lang = transcription["language"]
            if spoken_lang == 'ko':
                result = GoogleTranslator(source='korean', target='english').translate(text=text)
                print(result)
            else:
                result = GoogleTranslator(source='english', target='korean').translate(text=text)
                print(result)
            
if __name__ == "__main__":
    main()