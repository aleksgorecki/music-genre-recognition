import pathlib
import tkinter
import tkinter as tk
from tkinter import font

import audiofile
import matplotlib.pyplot as plt
import pygame.mixer
import tensorflow as tf
import numpy as np

from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import audio_processing
import inference
import visual
from gtzan_utils import GTZANLabels
import threading
import time


class GlobalEventFunctions:

    @staticmethod
    def pause_song():
        pygame.mixer.music.pause()

    @staticmethod
    def unpause_song():
        pygame.mixer.music.unpause()

    @staticmethod
    def play_song():
        pygame.mixer.music.play(-1)

    @staticmethod
    def set_song_position(position: float):
        pygame.mixer.music.set_pos(position)

    @staticmethod
    def open_song_file():
        pass

    @staticmethod
    def change_song():
        pass

    @staticmethod
    def next_song():
        pass

    @staticmethod
    def previous_song():
        pass

    @staticmethod
    def update_class_plot():
        pass

    @staticmethod
    def change_model():
        pass

    @staticmethod
    def change_volume(volume: float):
        pygame.mixer.music.set_volume(volume)


#class PlayerButtonMenuBar(tk.Frame)


class CustomScale(tk.Scale):
    def __init__(self, master, **kwargs):
        tk.Scale.__init__(self, master=master, **kwargs)
        self.bind("<Button-1>", self.snap_to_left_click)
        self.configure(activebackground="white", background="white", showvalue=False, highlightbackground="white", highlightcolor="white", border=0, borderwidth=0)

    def snap_to_left_click(self, event):
        self.event_generate("<Button-2>", x=event.x, y=event.y)



class InfoHeaderFrame(tk.Frame):

    def __init__(self, song_name: str):
        super().__init__()

        self.configure(background="white")

        self.song_frame = tk.Frame(background="white")
        self.song_frame.grid(column=1, row=0, padx=0, pady=0, sticky='')

        self.loading_gif_path = "./resources/spinner.gif"
        self.loading_gif = tk.PhotoImage(file=self.loading_gif_path, format="gif")

        self.song_title = tk.Label(text="No file loaded", master=self.song_frame, background="white")
        self.song_title.grid(column=0, row=0, padx=0, pady=0, sticky='')

        self.class_plot = None

    def display_class_plot(self, plot_img):
        self.class_plot = tk.Label(image=plot_img, background="white")
        #self.class_plot = FigureCanvasTkAgg(figure=fig).get_tk_widget()
        self.class_plot.grid(column=0, row=1, padx=0, pady=0, sticky='')



class PlayerWidget(tk.Frame):

    def __configure_left_frame_widgets(self):
        pass

    def __configure_right_frame_widgets(self):
        pass

    def __configure_center_frame_widgets(self):
        pass

    def play_pause_click(self):
        if self.playing:
            self.play_pause_button.configure(image=self.images["play"])
            GlobalEventFunctions.pause_song()
        else:
            self.play_pause_button.configure(image=self.images["pause"])
            if self.started:
                GlobalEventFunctions.unpause_song()
            else:
                self.started = True
                GlobalEventFunctions.play_song()

        self.playing = not self.playing

    def volume_button_click(self):
        if self.muted:
            self.volume_button.configure(image=self.images["volume"])
            GlobalEventFunctions.change_volume(self.previous_volume)
            self.volume_slider_position.set(self.previous_volume * 100)
        else:
            self.volume_button.configure(image=self.images["muted"])
            self.volume_slider_position.set(0)
            self.previous_volume = pygame.mixer.music.get_volume()
            GlobalEventFunctions.change_volume(0)
        self.muted = not self.muted

    def volume_slider_changed(self):
        volume = float(self.volume_slider.get() / 100)
        if not self.muted:
            GlobalEventFunctions.change_volume(volume)

    def time_slider_changed(self):
        #pygame.mixer.music.rewind()
        self.start_position = self.timeslider.get()
        new_position = float(self.timeslider.get() / 1000) * self.song_duration
        pygame.mixer.music.play(start=new_position)
        if not self.playing:
            pygame.mixer.music.pause()
        #GlobalEventFunctions.set_song_position(new_position)

    def update_timeslider(self):

        if self.song_duration == 0:
            self.time_slider_position.set(0)
        else:
            self.time_slider_position.set(self.start_position + (pygame.mixer.music.get_pos() / self.song_duration))

        current_time_str = time.strftime("%M:%S", time.gmtime(int((self.timeslider.get() / 1000) * self.song_duration)))
        duration_str = time.strftime("%M:%S", time.gmtime(int(self.song_duration)))
        self.current_time_label.configure(text=current_time_str)
        self.song_duration_label.configure(text=duration_str)

        if self.time_slider_position.get() > 998.9:
            self.start_position = 0
            pygame.mixer.music.stop()
            pygame.mixer.music.play()

        self.after_id = self.after(ms=100, func=self.update_timeslider)


    def reset(self):
        self.started = False
        self.playing = False
        self.song_file = None
        self.song_duration = 0
        self.play_pause_button.configure(image=self.images["play"])
        self.start_position = 0
        pygame.mixer.music.play()
        pygame.mixer.music.stop()
        self.update_timeslider()

    def __init__(self, master):
        super().__init__()
        self.configure(background="white")

        image_paths = {
            "play": "./resources/play.png",
            "pause": "./resources/pause.png",
            "next": "./resources/next.png",
            "previous": "./resources/previous.png"
        }

        target_w = 15
        target_h = 20

        self.images = {
            "play": ImageTk.PhotoImage(Image.open("./resources/play.png").resize((target_w, target_h))),
            "pause": ImageTk.PhotoImage(Image.open("./resources/pause.png").resize((target_w, target_h))),
            "next": ImageTk.PhotoImage(Image.open("./resources/next.png").resize((target_w, target_h))),
            "previous": ImageTk.PhotoImage(Image.open("./resources/previous.png").resize((target_w, target_h))),
            "volume": ImageTk.PhotoImage(Image.open("./resources/volume.png").resize((target_w, target_h))),
            "muted": ImageTk.PhotoImage(Image.open("./resources/muted.png").resize((target_w, target_h)))
        }

        self.started = False
        self.playing = False
        self.muted = False
        self.previous_volume = 0.3
        self.song_file: str = None
        self.song_duration: float = 0.0
        self.time_slider_position = tk.DoubleVar()
        self.volume_slider_position = tk.DoubleVar()
        self.volume_slider_position.set(self.previous_volume * 100)
        self.after_id = None
        self.start_position = 0


        self.after(ms=50, func=self.update_timeslider)


        self.left_frame = tk.Frame(background="white")
        self.left_frame.grid(column=0, row=1, padx=10, pady=10, sticky='w')
        self.center_frame = tk.Frame(background="white")
        self.center_frame.grid(column=1, row=1, padx=0, pady=10, sticky='')
        self.right_frame = tk.Frame(background="white")
        self.right_frame.grid(column=2, row=1, padx=10, pady=10, sticky='e')

        self.previous_button = tk.Button(self.left_frame, image=self.images["previous"], highlightthickness=0, bd=0, bg="white", activebackground="white")
        self.previous_button.grid(column=0, row=0, padx=10, pady=0, sticky='')
        self.play_pause_button = tk.Button(self.left_frame, image=self.images["play"], highlightthickness=0, bd=0, bg="white", activebackground="white", command=lambda: PlayerWidget.play_pause_click(self))
        self.play_pause_button.grid(column=1, row=0, padx=10, pady=0, sticky='')
        self.next_button = tk.Button(self.left_frame, image=self.images["next"], highlightthickness=0, bd=0, bg="white", activebackground="white")
        self.next_button.grid(column=2, row=0, padx=10, pady=0, sticky='')

        self.current_time_label = tk.Label(master=self.center_frame, text="0:00", highlightthickness=0, bd=0, bg="white", activebackground="white", borderwidth=0, border=0)
        self.current_time_label.grid(column=0, row=0, padx=0, pady=0, sticky='')

        self.timeslider = CustomScale(master=self.center_frame, from_=0, to=1000, orient=tk.HORIZONTAL, length=300, sliderlength=15, command=lambda dummy: self.time_slider_changed(), variable=self.time_slider_position)
        self.timeslider.grid(column=1, row=0, padx=10, pady=0, sticky='')

        self.song_duration_label = tk.Label(self.center_frame, text="0:00", highlightthickness=0, bd=0, bg="white", activebackground="white", borderwidth=0, border=0)
        self.song_duration_label.grid(column=2, row=0, padx=0, pady=0, sticky='')


        self.volume_button = tk.Button(self.right_frame, image=self.images["volume"], highlightthickness=0, bd=0, bg="white",
                                       activebackground="white", command=lambda: PlayerWidget.volume_button_click(self))
        self.volume_button.grid(column=0, row=0, padx=0, pady=0, sticky='')

        self.volume_slider = CustomScale(self.right_frame, from_=0, to=100, orient=tk.HORIZONTAL, sliderlength=15, length=70, command=lambda dummy: self.volume_slider_changed(), variable=self.volume_slider_position)
        self.volume_slider.grid(column=1, row=0, padx=10, pady=0, sticky='we')


        self.info_header_frame = InfoHeaderFrame(self.song_file)
        self.info_header_frame.grid(column=1, row=0, padx=10, pady=0, sticky='we')


        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg="white")
        self.file = None
        self.file_duration = None
        self.style = ttk.Style().theme_use("clam")
        self.prediction_thread: threading.Thread = None
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)

        self.class_img = None

        # root window
        self.title('Music genre recognition')
        self.wm_minsize(700, 400)
        self.resizable(False, False)
        self.style = ttk.Style(self)
        self.working_fig, self.working_ax = plt.subplots()

        self.player_widget = PlayerWidget(self)
        self.player_widget.grid(column=0, row=0, padx=0, pady=0, sticky='nsew')

        pygame.mixer.init()
        pygame.mixer.music.set_volume(self.player_widget.previous_volume)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        #self.grid_rowconfigure(1, weight=1)
        #self.grid_rowconfigure(2, weight=1)

        self.right_panel_background = tk.Label(borderwidth=2, relief="raised")
        self.right_panel_background.grid(column=3, row=0, padx=5, pady=5, sticky='nsew', rowspan=3)


        self.open_file_button = tk.Button(text="Load a file", command=lambda: App.open_file(self))
        self.open_file_button.grid(column=0, row=2, padx=5, pady=5, sticky='e')
        self.predict_genre_button = tk.Button(text="Run genre recognition", command=lambda: self.start_prediction_thread())
        self.predict_genre_button.grid(column=1, row=2, padx=5, pady=5, sticky='we')
        self.open_directory_button = tk.Button(text="Settings")
        self.open_directory_button.grid(column=2, row=2, padx=5, pady=5, sticky='w')

        self.open_directory_button = tk.Button(text="Load a directory")
        self.open_directory_button.grid(column=3, row=2, padx=10, pady=10, sticky='')


        self.directory_list_box = tk.Listbox(relief="sunken", borderwidth=2, background="white")
        self.directory_list_box.grid(column=3, row=0, padx=10, pady=10, sticky='nsew')


        self.protocol("WM_DELETE_WINDOW", self.on_close_event)

        self.load_model()

        # self.after(ms=20000, func=lambda: self.player_widget.info_header_frame.display_class_plot(self.class_img))


    def load_model(self):
        self.model = tf.keras.models.load_model("/home/aleksy/checkpoints50-256/90-0.88")

    def on_close_event(self):
        print(threading.active_count())
        tf.keras.backend.clear_session()
        if self.player_widget.after_id is not None:
            self.player_widget.after_cancel(self.player_widget.after_id)
        pygame.mixer.music.stop()
        self.quit()
        self.destroy()

    def open_file(self):
        filepath = filedialog.askopenfilename(filetypes=(("mp3 audio files", (".mp3")),), title="Select a file", initialdir="/home/aleksy/Full_Songs")
        if filepath != "":
            self.file = filepath
            pygame.mixer.music.load(filepath)
            self.file_duration = audiofile.duration(filepath)
            self.player_widget.reset()
            self.player_widget.song_file = self.file
            self.player_widget.song_duration = self.file_duration
            self.player_widget.info_header_frame.song_title.configure(text=str(pathlib.Path(self.file).stem))

    def start_prediction_thread(self):
        if self.prediction_thread is not None:
            if self.prediction_thread.is_alive():
                print("Stil running.")
                return
        self.prediction_thread = threading.Thread(target=self.predict_genre, daemon=True)
        self.prediction_thread.setDaemon(True)
        self.prediction_thread.start()
        # proc = multiprocessing.Process(target=self.predict_genre)
        # proc.start()
        # proc.join()
        # self.player_widget.info_header_frame.display_loading_animation()

    def draw_class_distribution(self):
        pass

    def predict_genre(self):
        print(self.file)
        if self.file is not None and self.file != ():
            model = self.model
            audio = audio_processing.load_to_mono(self.file)
            plots_interval_sec = 30
            audio_in_sec = len(audio.timeseries) / audio.sr
            plot_points = np.arange(0, len(audio.timeseries) / audio.sr - plots_interval_sec,
                                    plots_interval_sec)
            mels = list()
            for pp in plot_points:
                audio_sample = audio_processing.get_fragment_of_timeseries(audio, offset_sec=pp,
                                                                           fragment_duration_sec=5.0)
                sample_mel = audio_processing.mel_from_timeseries(audio_sample, mel_bands=256)
                mels.append(sample_mel)
            fig, ax = self.working_fig, self.working_ax
            images = []
            for mel in mels:
                visual.mel_only_on_ax(mel, ax)
                img = inference.fig_to_array(fig)
                img = inference.preprocess_img_array(img)
                images.append(img)

            outputs = []
            for image in images:
                output = model.predict(x=image, batch_size=1)
                outputs.append(output)

            sum_output = np.sum(outputs, axis=0)
            mean_output = sum_output / len(outputs)

            output = mean_output

            label_idx_dict = dict()
            GTZAN_classes = GTZANLabels
            for idx, label in enumerate(sorted(GTZAN_classes)):
                label_idx_dict.update({idx: label})
            probability_dict = dict()
            for idx, prob in enumerate(list(output[0])):
                probability_dict.update({label_idx_dict[idx]: prob})

            for item in probability_dict.items():
                print(item)

            print("=========")

            for item in probability_dict.items():
                if item[1] > 0.09:
                    proc = item[1] * 100
                    print(item[0], proc)

            print("==========")
            print(max(probability_dict.items(), key=lambda item: item[1])[0])

            outputs_scaled = [int(x * 100) for x in output[0]]
            labels = [x.capitalize() for x in sorted(GTZAN_classes)]

            d = {
                "class": labels,
                "prob": outputs_scaled
            }

            # fig, ax = plt.subplots()
            # visual.prepare_class_for_class_distribution(fig, ax)
            # visual.draw_class_distribution(ax, labels, outputs_scaled)
            # arr = inference.fig_to_array(fig=fig)
            # self.class_img = ImageTk.PhotoImage(Image.fromarray(arr).resize(size=(300, 200)))
        return


if __name__ == "__main__":
    app = App()
    app.mainloop()
