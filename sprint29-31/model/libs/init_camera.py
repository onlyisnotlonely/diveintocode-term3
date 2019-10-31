import cv2


def init_camera(raspi_flg = True):
    """
    ラズベリーパイカメラを初期化する関数
    """
    cap = None
    res = False

    while res is False:
        if raspi_flg:
            import picamera

            cap = picamera.PiCamera()
            # cap.start_preview()
            cap.resolution = (640, 480)
            cap.framerate = 33
            cv2.waitKey(1000)
            res = True
            print('Setting the picamera ..')

        else:
            cap = cv2.VideoCapture(0)
            res, _ = cap.read()
            #pass
            cv2.waitKey(1000)
            print('Setting the build-in-camera ..')

    return cap


def get_image(cap, raspi_flg = True):
    """
    ラズベリーパイで撮影した画像のデータを出力する関数
    """
    c_frame = None

    if raspi_flg:
        import picamera.array

        with picamera.array.PiRGBArray(cap, size=(640, 480)) as stream:
            c_frame = cap.capture(stream, 'bgr')
            c_frame = stream.array
            if c_frame is None:
                exit

    else:
        end_flag, c_frame = cap.read()
        if end_flag is False or c_frame is None:
            exit

    return c_frame
