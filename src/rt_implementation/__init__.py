import os
import threading
from queue import Queue
import psutil
from rt_implementation.audio_recorder import AudioRecorder
from rt_implementation.preprocessing import RealTimeMFSCPreprocessor
from rt_implementation.inference_lite import RealTimeInferenceLite
#from rt_implementation.inference import RealTimeInference
from rt_implementation.gui_plot import GUIPlot


F_SAMPLING = 16000
SIZE = 200
THRESHOLD = 0.5


def real_time_implementation():
    recorder = AudioRecorder(F_SAMPLING, SIZE)
    preprocessor = RealTimeMFSCPreprocessor(F_SAMPLING, SIZE)
    model = RealTimeInferenceLite('cnn_model_v4_25(12,8,5).tflite')
    #model = RealTimeInference('cnn_model_original_25(12,8,5).h5')
    preds = Queue()
    gui = GUIPlot(THRESHOLD)

    thread_recorder = threading.Thread(target=recorder.start_recording, daemon=True)
    thread_preprocessor = threading.Thread(target=preprocessor.start_preprocessing, args=(recorder.recordings,), daemon=True)
    thread_model = threading.Thread(target=model.start_inference, args=(preprocessor.images_MFSC, preds), daemon=True)

    thread_recorder.start()
    thread_preprocessor.start()
    thread_model.start()

    pid = os.getpid()
    process = psutil.Process(pid)
    usage_counter = 0
    cpu_percent = 0
    mem = 0
    try:
        while True:
            usage_counter += 1
            cpu_percent += process.cpu_percent()/100
            mem += process.memory_info().rss/(1024*1024)/100
            if usage_counter == 100:
                print(f"CPU usage: {cpu_percent/100:.2f}% | Memory usage: {mem:.2f} MB")
                usage_counter = 0
                cpu_percent = 0
                mem = 0

            gui.update_color(preds)

    except KeyboardInterrupt:
        recorder.stop_recording()
        preprocessor.stop_preprocessing()
        model.stop_inference()

        thread_recorder.join()
        thread_preprocessor.join()
        thread_model.join()
