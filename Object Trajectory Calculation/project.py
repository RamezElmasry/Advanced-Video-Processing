import cv2
import numpy as np
import math
import PySimpleGUI as sg
from PIL import Image, ImageTk
import io
import os

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object is detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already before in center points dict
            # Decide if it is the same object based on a certain distance threshold
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 85:
                    cv2.line(background, (cx,cy), (pt[0],pt[1]), (255, 0, 0), 1)
                    #assign id to the new position
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # Different objects
            # New object is detected, so we assign a new ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by removing center points of the IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Assign current centers to be previous so we check on them on the next iteration
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

def Background_Modeling(source):
    # Open Video
    cap = cv2.VideoCapture(source)
    beta = 0.1

    # read the frames from the camera
    ret, frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    bg = (gray1 + gray2)/2
    # loop runs if capturing has been initialized. 
    while True:
        # reads frames
        ret, frame3 = cap.read()
        if frame3 is None:
            cap.release()
            break

        gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
        bg = updateBackground(bg, gray3, beta)

    cv2.imwrite('Ruska25A_Background.jpg', bg)
    return bg

#####################################################################    
def updateBackground(bg, current_frame, beta):
  return ((1-beta) * bg) + (beta * current_frame)
#####################################################################

def object_detection_tracking(source, bg):
  cap = cv2.VideoCapture(source)
  bg = np.uint8(bg)
  tracker = EuclideanDistTracker()
  count = 1

  while True:
    ret, current_frame = cap.read()
    
    if current_frame is None:
        cap.release()
        break
    if count > 87:
        continue
    #...................Object Detection...................#
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray, bg)
    binary = cv2.threshold(frame_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.dilate(binary, None, iterations=2)
    binary = cv2.erode(binary, None, iterations=2)
    mask = cv2.inRange(binary, 255, 255)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:
            (x, y, w, h) = cv2.boundingRect(contour)
            #cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            detections.append([x, y, w, h])

    #..................Object Tracking...................#
    #print(detections)
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        cv2.circle(current_frame, (cx,cy), 5, (255, 0, 0), -1)
        cv2.putText(current_frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(background, (cx,cy), 5, (255, 0, 0), -1)
        cv2.putText(background, str(id), (cx - 5, cy), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        

    cv2.imwrite('output/' + str(count) + '.png', current_frame)  
    #cv2.imwrite("gt frames/GT0"+ str(count) + ".jpg", current_frame)
    count += 1
    key = cv2.waitKey(0)
    if key == 27:
        break

  cv2.imwrite('trajectory.png', background)
  cap.release()
  cv2.destroyAllWindows()


# Generate image data using PIL
def get_img_data(f, maxsize=(1200, 850), first=False):
    img = Image.open(f)
    img.thumbnail(maxsize)
    # tkinter is inactive the first time showing img so it shows an error reading the img
    if first: 
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

def GUI():
    sg.theme('DarkPurple1')
    trajectory = get_img_data('trajectory.png', first = True)

    shows = []
    files_in_path = os.listdir('output/')
    files = [file for file in files_in_path if file.endswith(".png")]
    sorted_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    
    for file in sorted_files:
        img = get_img_data(os.path.join('output/', file), first = True)
        shows.append(img)

    
    layout = [[sg.Image(data=shows[0], size = (640,480), key='-IMAGE-', right_click_menu=['UNUSED', 'Exit'])],
            [sg.Button('Back', key='Back'), sg.Button('Forward', key='Forward'), 
            sg.Button('Show Trajectory', key='TRAJECTORY')]]

    window = sg.Window('Image Viewer', layout ,resizable = False ,element_justification='center')

    offset = 0

    show = shows[0]
    while True:  # Event Loop
        event, values = window.read()
        if event in (None, 'Exit', 'Cancel'):
            break

        if event == 'TRAJECTORY':
            show = trajectory

        if event == 'Forward':
            offset += 1 
            if offset <= len(shows) - 1 :
                show = shows[offset]

        if event == 'Forward':
            if offset > len(shows) - 1:
                offset = 0
                show = shows[offset]

        if event == 'Back':
            offset -= 1
            if offset >= 0:
                show = shows[offset]

        if event == 'Back':
            if offset < 0:
                offset = len(shows) - 1
                show = shows[offset]

        # update the image in the window
        window['-IMAGE-'].update(show, size = (640,480))

if __name__ == '__main__':
    source = 'RuskaUfer25A.avi'
    bg = Background_Modeling(source)
    background = cv2.imread('Ruska25A_Background.jpg')
    object_detection_tracking(source, bg)
    GUI()