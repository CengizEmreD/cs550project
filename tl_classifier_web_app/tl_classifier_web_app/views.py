import time
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
from .aws_helper import aws_rekognition_classify

# ---------- Secrets ----------
aws_access_key_id = ''
aws_secret_access_key = ''
# -----------------------------

# test_bucket_name = 'cs550-project-test-0'
# test_image_name = 'test_image.jpg'
# save_s3_as = 'test.jpg'

# Model v1
# rekognition_project_version_arn = 'arn:aws:rekognition:eu-central-1:627666257074:project/cs550-tl-classification/version/cs550-tl-classification.2021-11-01T12.29.09/1635758949344'

# Model v2
# rekognition_project_version_arn = 'arn:aws:rekognition:eu-central-1:627666257074:project/tl_classification_v2/version/tl_classification_v2.2021-12-30T19.20.49/1640881249626'

# Model v3
rekognition_project_version_arn = 'arn:aws:rekognition:eu-central-1:627666257074:project/tl_classification_v3/version/tl_classification_v3.2022-01-03T09.30.43/1641191443524'

font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()
        self.current_pred = list()
        self.previous_pred = list()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame

        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)

        try:
            response = aws_rekognition_classify(aws_access_key_id, aws_secret_access_key,
                                                rekognition_project_version_arn, image)
        except:
            response = {'CustomLabels': [{'Name': 'Rekognition service is offline!'}]}

        if len(response['CustomLabels']) > 0:
            self.current_pred = response['CustomLabels'][0]['Name']
        else:
            self.current_pred = "-"

        prediction = "-"
        if self.current_pred == self.previous_pred:
            prediction = self.current_pred

        cv2.putText(image, str(prediction), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

        print([self.current_pred, self.previous_pred])

        self.previous_pred = self.current_pred

        _, jpeg = cv2.imencode('.jpg', image)

        time.sleep(0.5)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def Home(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        return None
