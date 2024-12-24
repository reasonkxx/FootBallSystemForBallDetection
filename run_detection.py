from src.detection.detect_ball import detect_ball_in_video

# Шляхи до файлів
model_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/runs/detect/train26/weights/best.pt"
video_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/data/input_video/test2.mp4"
output_video_path = "D:/university/KursovaWork/FootBallSystemForBallDetection/data/output_video/output_test3.mp4"


detect_ball_in_video(model_path, video_path, output_video_path)