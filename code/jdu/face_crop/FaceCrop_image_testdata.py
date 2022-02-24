import cv2
import mediapipe as mp
import time
import os


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            self.minDetectionCon)

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, detection)
                # print(detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    #cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=6, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left x,y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right x1,y
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)        
        # Bottom Left x,y1
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), t)   
        # Bottom Right x1,y1
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)           

        return img


def main():
    a, b = 0, 0
    # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    sample_img_path = 'data/train/eval/images/3bb99aab2a2ab3a96ce6b044d7b7e3eb70cf52b7.jpg'
    
    idx = sample_img_path[::-1].find("/")
    dot_idx = sample_img_path[::-1].find(".")

    file_name = sample_img_path[-idx:-dot_idx-1]

    img = cv2.imread(sample_img_path)
    # cv2.imshow("Image", img)
    detector = FaceDetector(0.7)
    
    img, bboxs = detector.findFaces(img, draw=False)
    # cv2.imshow("Image", img)
    add_size = 30

    if len(bboxs) != 0:
        
        a = bboxs[0][1][0] - add_size
        b = bboxs[0][1][1] - add_size - 40
        if (a<0) : a=0
        if (b<0) : b=0
        c = bboxs[0][1][2] + (2*add_size)
        d = bboxs[0][1][3] + (2*add_size) + 40
        print(b,b+d, a,a+c)
        img = cv2.circle(img, (a,b), 5, (255,255,255), 2)
        img = cv2.circle(img, (a+c,b+d), 5, (255,255,255), 2)
        
        cropped_face = img[b:b+d, a:a+c]


        # cTime = time.time()
        # fps = 1/(cTime-pTime)
        # pTime = cTime
        # cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
        #             3, (0, 255, 0), 2)
        # cv2.imshow("Image", img)
        cv2.imshow("cropped", cropped_face)
        output_dir_path = "output_image/eval/images/"
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        try:
            cv2.imwrite(output_dir_path + "/" + file_name +".jpg", cropped_face)
            print("success save :  " + output_dir_path + "/" + file_name +".jpg")
        except:
            print("IMAGE LOAD FAILED : ", file_name)
    else:
        print("DETECTION FAILED : ", file_name)

        
    print("Finish!")

    cv2.waitKey(1)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
