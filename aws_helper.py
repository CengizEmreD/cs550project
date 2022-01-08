import boto3
import base64
import cv2


def upload_to_s3(aws_access_key_id, aws_secret_access_key, bucket_name, file_name, save_as):
    s3 = boto3.resource('s3',
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key)

    s3.meta.client.upload_file(file_name, bucket_name, save_as)


def aws_rekognition_classify(aws_access_key_id, aws_secret_access_key, rekognition_project_version_arn, image):
    rek_client = boto3.client('rekognition', region_name='eu-central-1',
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key)

    _, image_jpg = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(image_jpg)
    base_64_binary = base64.decodebytes(base64_image)

    response = rek_client.detect_custom_labels(
        ProjectVersionArn=rekognition_project_version_arn,
        Image={
            'Bytes': base_64_binary
        },
        MaxResults=1,
        MinConfidence=55
    )

    return response
