import threading
from queue import Queue
import time
from real_time_implementation.audio_recorder import AudioRecorder
from real_time_implementation.preprocessing import RealTimeMFSCPreprocessor
from real_time_implementation.inference_lite import RealTimeInferenceLite
from real_time_implementation.gui import GUI

F_SAMPLING = 16000
SIZE = 200


def real_time_implementation():
    recorder = AudioRecorder(F_SAMPLING, SIZE)
    preprocessor = RealTimeMFSCPreprocessor(F_SAMPLING, SIZE)
    model = RealTimeInferenceLite('cnn_model_v4_25(12,8,5).tflite')
    preds = Queue()
    gui = GUI(0.5)

    thread_recorder = threading.Thread(target=recorder.start_recording, daemon=True)
    thread_preprocessor = threading.Thread(target=preprocessor.start_preprocessing, args=(recorder.recordings,), daemon=True)
    thread_model = threading.Thread(target=model.start_inference, args=(preprocessor.images_MFSC, preds), daemon=True)

    thread_recorder.start()
    thread_preprocessor.start()
    thread_model.start()

    try:
        while True:
            gui.update_color(preds)
            time.sleep(0.01)
    except KeyboardInterrupt:
        recorder.stop_recording()
        preprocessor.stop_preprocessing()
        model.stop_inference()

        thread_recorder.join()
        thread_preprocessor.join()
        thread_model.join()
