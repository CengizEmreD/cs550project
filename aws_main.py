import cv2
from aws_helper import upload_to_s3, aws_rekognition_classify

# ---------- Secrets ----------
aws_access_key_id = ''
aws_secret_access_key = ''
# -----------------------------

test_bucket_name = 'cs550-project-test-0'
test_image_name = 'test_image.jpg'
save_s3_as = 'test.jpg'
rekognition_project_version_arn = 'arn:aws:rekognition:eu-central-1:627666257074:project/cs550-tl-classification/version/cs550-tl-classification.2021-11-01T12.29.09/1635758949344'

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        try:
            # upload_to_s3(aws_access_key_id, aws_secret_access_key, test_bucket_name, test_image_name, save_s3_as)
            response = aws_rekognition_classify(aws_access_key_id, aws_secret_access_key,
                                                rekognition_project_version_arn, frame)
        except:
            response = {'CustomLabels': 'Rekognition service is offline!'}

        cv2.putText(frame, str(response['CustomLabels']), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

        cv2.imshow("test", frame)

    cam.release()

    cv2.destroyAllWindows()
