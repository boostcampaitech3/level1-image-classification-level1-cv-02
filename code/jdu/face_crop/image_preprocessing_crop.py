import os, sys
import cv2
import mediapipe as mp


image_path = "data/train/eval/images/"
image_dir_list = [i for i in os.listdir(image_path) if i[:2] != "._" ]
# image_dir_list

image_list = []

for image_dir in image_dir_list:
    # for train data
    # for element in [image_path + image_dir + "/" +i for i in os.listdir(image_path + image_dir) if i[:2] != "._" ]:
    #     image_list.append(element)

    # for eval data
    for element in [image_path + image_dir]:
        image_list.append(element)


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
    # img_path = 'data/train/train/images/000001_female_Asian_45/mask3.jpg'
    for img_path in image_list:
    
        idx = img_path[::-1].find("/")

        dot_idx = img_path[::-1].find(".")

        dir_name, file_name = img_path[25:-idx-1], img_path[-idx:-dot_idx-1]

        img = cv2.imread(img_path)
        
        detector = FaceDetector(0.4)
        
        img, bboxs = detector.findFaces(img, draw=False)

        add_size = 30

        if len(bboxs) != 0:
            a = bboxs[0][1][0] - add_size
            b = bboxs[0][1][1] - add_size - 40
            if(a<0): a=0
            if(b<0): b=0
            c = bboxs[0][1][2] + (2*add_size)
            d = bboxs[0][1][3] + (2*add_size) + 40
            
            cropped_face = img[b:b+d, a:a+c]

            # cTime = time.time()
            # fps = 1/(cTime-pTime)
            # pTime = cTime
            # cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
            #             3, (0, 255, 0), 2)
            # cv2.imshow("Image", img)
            # cv2.imshow("cropped", cropped_face)
            output_dir_path = "output_image/" + dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            try:
                cv2.imwrite("output_image/"+dir_name+"/"+ file_name +".jpg", cropped_face)
            except:
                print("IMAGE LOAD FAILED : "+dir_name + "/" + file_name)
        else:
            print("FAILED : "+dir_name + "/" + file_name)
    print("Finish!")

if __name__ == "__main__":
    main()