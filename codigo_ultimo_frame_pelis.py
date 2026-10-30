
import cv2

# 1. Rutas de entrada y salida
video_path_1 = r"C:\Users\ivana\Dropbox\Ivana Ducrey\Doctorado- Drosophila melanogaster\Berlin Lisa\10_29_2026\10_29_2026_movie_0.mp4"
output_path_1 = r"C:\Users\ivana\Dropbox\Ivana Ducrey\Doctorado- Drosophila melanogaster\Berlin Lisa\10_29_2026\ultimo_frame_0.png"

video_path_2 = r"C:\Users\ivana\Dropbox\Ivana Ducrey\Doctorado- Drosophila melanogaster\Berlin Lisa\10_29_2026\10_29_2026_movie_1.mp4"
output_path_2 = r"C:\Users\ivana\Dropbox\Ivana Ducrey\Doctorado- Drosophila melanogaster\Berlin Lisa\10_29_2026\ultimo_frame_1.png"

videos = [video_path_1, video_path_2]
outputs = [output_path_1, output_path_2]

# 2. Procesar cada video
for video_path, output_path in zip(videos, outputs):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: no se pudo abrir el video: {video_path}")
        continue

    # 3. Obtener la cantidad total de frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"Error: no se pudo determinar la cantidad de frames: {video_path}")
    else:
        # 4. Seleccionar el último frame
        video.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

        # 5. Leer el frame seleccionado
        success, frame = video.read()

        if success:
            # 6. Guardar el frame como imagen
            saved = cv2.imwrite(output_path, frame)

            if saved:
                print(f"Frame guardado en: {output_path}")
            else:
                print(f"Error al guardar la imagen: {output_path}")
        else:
            print(f"Error: no se pudo leer el último frame: {video_path}")

    # 7. Cerrar el video
    video.release()

  