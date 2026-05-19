import time

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections
PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections

MODEL_PATH = "holistic_landmarker.task"
OUTPUT_PATH = "holistic_landmark_output.mp4"
RECORD_SECONDS = 10  # 録画時間(秒)


def draw_landmarks(image, results):
    """検出された Holistic のランドマークをカメラ画像に重ねて描画する。"""
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
    return image


def main():
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
    )

    # Webカメラから入力
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return

    # 出力動画の設定。カメラの解像度に合わせる。
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:  # カメラが FPS を報告しない場合のフォールバック
        fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print(f"{RECORD_SECONDS}秒間録画します... (Esc キーで途中終了)")
    start_time = time.time()

    with HolisticLandmarker.create_from_options(options) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("空のカメラフレームを無視します。")
                continue

            # BGR -> RGB に変換して MediaPipe の Image にする
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # 検出を実行し、結果をフレームに描画
            results = holistic.detect(mp_image)
            image = draw_landmarks(image, results)

            # 描画済みフレームを MP4 へ書き出し
            writer.write(image)

            # プレビュー表示(左右反転して鏡像表示)
            cv2.imshow("MediaPipe Holistic", cv2.flip(image, 1))

            elapsed = time.time() - start_time
            if elapsed >= RECORD_SECONDS:
                print(f"録画完了: {OUTPUT_PATH}")
                break
            if cv2.waitKey(5) & 0xFF == 27:  # Esc キーで途中終了
                print(f"録画を中断しました: {OUTPUT_PATH}")
                break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
