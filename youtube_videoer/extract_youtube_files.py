import csv
import subprocess


def download_audio():
    with open('liste_af_youtube_excel_sidste.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        for row in (csv_reader):
            subprocess.call(f"ffmpeg -ss 00:15:00 -t 00:15:00 -i $(youtube-dl --verbose -f worst -g 'https://www.youtube.com/watch?v={row[0]}') audio/{row[0]}.wav", shell=True)
download_audio()
#"ffmpeg -ss 00:15:00 -t 00:15:00 -i $(youtube-dl -verbose -f 140 -g 'https://www.youtube.com/watch?v={row[0]}') {row[0]}.wav""