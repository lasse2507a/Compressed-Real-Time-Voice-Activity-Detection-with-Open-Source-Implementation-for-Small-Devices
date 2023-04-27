import threading
import time
from real_time_implementation.audio_recorder import AudioRecorder
from real_time_implementation.real_time_preprocessing import RealTimeMFSCPreprocessor

F_SAMPLING = 16000
SIZE = 200

def real_time_implementation():
    recorder = AudioRecorder(F_SAMPLING, SIZE)
    preprocessor = RealTimeMFSCPreprocessor(F_SAMPLING, SIZE)

    thread_recorder = threading.Thread(target=recorder.start_recording, daemon=True)
    thread_preprocessor = threading.Thread(target=preprocessor.start_preprocessing, args=(recorder.recordings,), daemon=True)

    thread_recorder.start()
    thread_preprocessor.start()

    time.sleep(5)

    recorder.stop_recording()
    preprocessor.stop_preprocessing()
    thread_recorder.join()
    thread_preprocessor.join()
