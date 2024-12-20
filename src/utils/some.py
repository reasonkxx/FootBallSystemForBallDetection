from moviepy.video.io.VideoFileClip import VideoFileClip

# Путь к исходному видеофайлу
input_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/data/input_video/test4.mp4"
# Путь к сохраненному видеофайлу
output_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/data/input_video/test5.mp4"

# Загружаем видеофайл
clip = VideoFileClip(input_path)

# Вычисляем длительность видео без последних 16 секунд
new_duration = max(clip.duration - 16, 0)

# Обрезаем видео
trimmed_clip = clip.subclip(0, new_duration)

# Сохраняем результат
trimmed_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Закрываем ресурсы
clip.close()
trimmed_clip.close()
