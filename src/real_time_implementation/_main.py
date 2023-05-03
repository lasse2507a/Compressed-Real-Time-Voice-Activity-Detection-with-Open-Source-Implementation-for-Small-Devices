import threading
import time
from real_time_implementation.audio_recorder import AudioRecorder
from real_time_implementation.real_time_preprocessing import RealTimeMFSCPreprocessor


F_SAMPLING = 16000
SIZE = 200


def real_time_implementation():
    recorder = AudioRecorder(F_SAMPLING, SIZE)
    preprocessor = RealTimeMFSCPreprocessor(F_SAMPLING, SIZE)
    #model = Model()

    thread_recorder = threading.Thread(target=recorder.start_recording, daemon=True)
    thread_preprocessor = threading.Thread(target=preprocessor.start_preprocessing, args=(recorder.recordings,), daemon=True)
    #thread_model = threading.Thread(target=model.start, args=(preprocessor.images_MFSC,), daemon=True)

    thread_recorder.start()
    thread_preprocessor.start()
    #thread_model.start()

    time.sleep(10)

    recorder.stop_recording()
    preprocessor.stop_preprocessing()
    #model.stop()
    thread_recorder.join()
    thread_preprocessor.join()
    #model.join()


if __name__ == '__main__':
    real_time_implementation()
