from gtts import gTTS
import os
from pygame import mixer
from tempfile import TemporaryFile
from io import BytesIO
import time

tts = gTTS(text='Hello world.', lang='en')
#tts.save("hello.mp3")
#os.system("mpg321 hello.mp3")

print("init mixer")
mixer.init()
#mp3_fp = BytesIO()
mp3_fp = TemporaryFile()
print("Write to bytes like")
tts.write_to_fp(mp3_fp)
mp3_fp.seek(0)
print("load file in mixer")
mixer.music.load(mp3_fp)
print("play sound")
mixer.music.play()

while mixer.music.get_busy():
  time.sleep(0.1)

print("Done")