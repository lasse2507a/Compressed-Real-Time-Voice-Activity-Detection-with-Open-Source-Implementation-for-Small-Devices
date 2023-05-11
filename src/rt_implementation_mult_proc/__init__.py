import os
import multiprocessing as mp
import psutil
from rt_implementation_mult_proc.audio_recorder import AudioRecorder
from rt_implementation_mult_proc.preprocessing import RealTimeMFSCPreprocessor
from rt_implementation_mult_proc.inference_lite import RealTimeInferenceLite
from rt_implementation_mult_proc.gui_plot import GUIPlot

F_SAMPLING = 16000
SIZE = 200
THRESHOLD = 0.5

def real_time_implementation():
    stop_event = mp.Event()

    recorder = AudioRecorder(F_SAMPLING, SIZE)
    preprocessor = RealTimeMFSCPreprocessor(F_SAMPLING, SIZE)
    model = RealTimeInferenceLite('cnn_model_v4_25(12,8,5).tflite')
    #model = RealTimeInference('cnn_model_original_25(12,8,5).h5')
    #gui = GUIPlot(THRESHOLD)

    process_recorder = mp.Process(target=recorder.start_recording)
    process_preprocessor = mp.Process(target=preprocessor.start_preprocessing, args=(recorder.recordings,))
    process_model = mp.Process(target=model.start_inference, args=(preprocessor.images_MFSC,))
    #process_gui = mp.Process(target=gui.update_color, args=(model.preds,))

    process_recorder.start()
    process_preprocessor.start()
    process_model.start()
    #process_gui.start()

    pid = os.getpid()
    process = psutil.Process(pid)
    usage_counter = 0
    cpu_percent = 0
    mem = 0
    try:
        while not stop_event.is_set():
            usage_counter += 1
            cpu_percent += process.cpu_percent()/100
            mem += process.memory_info().rss/(1024*1024)/100
            if usage_counter == 100:
                print(f"CPU usage: {cpu_percent/100:.2f}% | Memory usage: {mem:.2f} MB")
                usage_counter = 0
                cpu_percent = 0
                mem = 0

    except KeyboardInterrupt:
        stop_event.set()
        recorder.stop_recording()
        preprocessor.stop_preprocessing()
        model.stop_inference()

        process_recorder.terminate()
        process_preprocessor.terminate()
        process_model.terminate()
        #process_gui.terminate()
