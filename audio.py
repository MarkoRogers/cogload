from faster_whisper import WhisperModel
import time
stime = time.time()
model_size = "large-v3"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("cleaned.wav", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
print(time.time()-stime)