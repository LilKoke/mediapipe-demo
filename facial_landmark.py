import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"
OUTPUT_PATH = "face_landmark_output.mp4"
RECORD_SECONDS = 10  # 録画時間(秒)

# このビルドの mediapipe には framework / solutions が無いため、
# 描画は drawing_utils を使わず OpenCV だけで行う。
# 接続トポロジ(線の繋ぎ方)は Tasks API の定数から取得する。
_conn = vision.FaceLandmarksConnections

# 部位ごとの (接続リスト, BGR色) 。テッセレーションは細い灰色の網目。
DRAW_SPEC = [
    (_conn.FACE_LANDMARKS_TESSELATION, (200, 200, 200), 1),
    (_conn.FACE_LANDMARKS_FACE_OVAL, (255, 255, 255), 1),
    (_conn.FACE_LANDMARKS_LIPS, (0, 0, 255), 1),
    (_conn.FACE_LANDMARKS_LEFT_EYE, (0, 255, 0), 1),
    (_conn.FACE_LANDMARKS_RIGHT_EYE, (0, 255, 0), 1),
    (_conn.FACE_LANDMARKS_LEFT_EYEBROW, (0, 255, 0), 1),
    (_conn.FACE_LANDMARKS_RIGHT_EYEBROW, (0, 255, 0), 1),
    (_conn.FACE_LANDMARKS_LEFT_IRIS, (255, 0, 0), 1),
    (_conn.FACE_LANDMARKS_RIGHT_IRIS, (255, 0, 0), 1),
]

# LIVE_STREAM モードのコールバックで受け取る最新の検出結果を保持する
latest_result = None


def result_callback(result, output_image, timestamp_ms):
    """FaceLandmarker が非同期に検出結果を返したときに呼ばれる。"""
    global latest_result
    latest_result = result


def draw_landmarks(image, detection_result):
    """検出結果のフェイスメッシュをカメラ画像に重ねて描画する。"""
    if detection_result is None:
        return image

    h, w = image.shape[:2]
    for face_landmarks in detection_result.face_landmarks:
        # 正規化座標(0-1)をピクセル座標へ変換
        points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

        # 接続線を描画
        for connections, color, thickness in DRAW_SPEC:
            for c in connections:
                if c.start < len(points) and c.end < len(points):
                    cv2.line(image, points[c.start], points[c.end], color, thickness)

    return image


def main():
    # Face Landmarker を LIVE_STREAM モードで設定
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=1,
        output_face_blendshapes=False,
        result_callback=result_callback,
    )

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

    with vision.FaceLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("空のカメラフレームを無視します。")
                continue

            # BGR -> RGB に変換して MediaPipe の Image にする
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 単調増加するタイムスタンプ(ミリ秒)で非同期検出を実行
            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(mp_image, timestamp_ms)

            # 最新の検出結果をフレームに描画
            frame = draw_landmarks(frame, latest_result)

            # 描画済みフレームを MP4 へ書き出し
            writer.write(frame)

            # プレビュー表示(左右反転して鏡像表示)
            cv2.imshow("MediaPipe Face Landmarker", cv2.flip(frame, 1))

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
