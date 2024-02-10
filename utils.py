import cv2
import numpy as np
from datetime import datetime
import os


class LoadDataset:
    def __init__(self, input_fol, output_fol) -> None:
        datetim = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        self.input_fol = input_fol
        self.input_images = os.listdir(input_fol)
        os.makedirs(output_fol + datetim)
        self.output_fol = output_fol + datetim + "/"
    def get_image_data(self):
        image_data = []
        for image in self.input_images:
            image_path = self.input_fol+image
            image_data.append(image_path)
        
        return image_data
    
    def save_result(self, image_name, image):
        save_path = self.output_fol+image_name
        cv2.imwrite(save_path, image) 

class Detect:
    def __init__(self, image_path) -> None:
        self.image = cv2.imread(image_path)
        img_blur = cv2.blur(self.image,(3,3)) 
        self.gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        self.num_contours = 0

    def detect_contours(self):
        #setting threshold and drawing contours
        _, threshold = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = 0
        for contour in contours:
            if num_contours == 0:     #since the first contour is the whole image itself
                num_contours+=1 
                continue
        # drawing the contours
            cv2.drawContours(self.image, [contour], 0, (0, 255, 0), 2)
            # epsilon = 0.02 * cv2.arcLength(contour, True)                       #this approach was also researched. but not included in the final approach.
            # approx = cv2.approxPolyDP(contour, epsilon, True)

            # # Iterate through each point in the contour
            # for point in approx:
            #     x, y = point[0]
            #     print(f"Coordinate: ({x}, {y})")
            #     cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
            num_contours+=1

        self.num_contours = num_contours-1


    
    def detect_corner(self):
        corners = cv2.goodFeaturesToTrack(self.gray, self.num_contours*10, 0.01, 10)    #assuming a contour can have max of 10 corner points
        corners = np.intp(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(self.image, (x, y), 4, (0,0,255), -1)

    def show_edge_length(self):
        # Apply edge detection using Canny
        edges = cv2.Canny(self.gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            for i in range(len(approx)-1):
                point1 = approx[i][0]
                point2 = approx[i + 1][0]
                length = int(np.linalg.norm(np.array(point2) - np.array(point1)))
                cv2.putText(self.image, str(length), ((point1[0]+point2[0])//2, (point1[1]+point2[1])//2), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(self.image, str(length), ((approx[0][0][0]+approx[-1][0][0])//2, (approx[0][0][1]+approx[-1][0][1])//2), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (0,0,0), 1, cv2.LINE_AA)
    def get_results(self):
        return self.image, self.num_contours
