import face_recognition
import os, sys
import cv2
import numpy as np
from datetime import datetime
import pickle
import openpyxl 


#open workbook
wb = openpyxl.load_workbook("FacialRecogAttendance.xlsx")

ws = wb['Attendance']

os.makedirs('captured_faces', exist_ok=True)


color = (0,0,255)

class FaceRecognition:
    face_location = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
        
        try:
            with open('face_encodings.pkl', 'rb') as file:
                known_face_encodings_dict = pickle.load(file)
                self.known_face_encodings = list(known_face_encodings_dict.values())
                self.known_face_names = list(known_face_encodings_dict.keys())
        except FileNotFoundError:
            self.known_face_encodings = []
            self.known_face_names = []

    #read and encode images from reference_images
    def encode_faces(self):
        
        try:
            with open('face_encodings.pkl', 'rb') as file:
                existing_face_encodings = pickle.load(file)
        except FileNotFoundError:
            existing_face_encodings = {}
            
        for image in os.listdir('reference_images'):
            filename, extension = os.path.splitext(image)

            if filename not in existing_face_encodings:
                face_image = face_recognition.load_image_file(f'reference_images/{image}')
                face_encoding = face_recognition.face_encodings(face_image)[0]

                existing_face_encodings[filename] = face_encoding

                print(filename)

        # Save the updated face encodings back to the file
        with open('face_encodings.pkl', 'wb') as file:
            pickle.dump(existing_face_encodings, file)
      
        
    #run facial recognition
    def run_recognition(self):
        
        #change number of cameras
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            sys.exit('Video Source Not Found...')
            
        counterPic = ws.max_row
        
        limiter = 0
        previous_face = None
        current_face = None
        
        while True:
            ret, frame = video_capture.read()
            
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
                
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
            
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                
                self.face_names = []
                
                colorFrame = (0, 0, 255)
            
                for face_encoding in self.face_encodings:
                    color = (0, 0, 255)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, 0.4)
                    
                    name = "Saved..."  

                    if self.known_face_encodings:  
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        # condition if face matches from reference_images folder
                        if matches[best_match_index]:
                        
                            print(limiter)
                            current_face = self.known_face_names[best_match_index]
                            
                            if limiter == 0 or current_face != previous_face:   
                                name = self.known_face_names[best_match_index]
                                date_time = datetime.now()                     
                                formatted_date_time = date_time.strftime("%Y-%m-%d %H:%M:%S")                        
                                output_path = f"captured_faces/{counterPic}_{name}.jpg"                                  
                                cv2.imwrite(f"captured_faces/{counterPic}_{name}.jpg", frame)
                                
                                counterPic += 1                                  
                                color = (0, 255, 0)
                                            
                                # add new attendance record
                                attendance = [name, formatted_date_time, output_path]                      
                                ws.append(attendance)                       
                                wb.save('FacialRecogAttendance.xlsx')
                                
                                limiter += 1
                                print(limiter)
                                
                            elif 0 < limiter < 10:
                                print('Not saving data')
                                limiter += 1
                                
                            else:
                                limiter = 0
                        else:
                            name = "Visitor..." 
                        
                        self.face_names.append(f'{name}')
                                             
                        colorFrame = color
                        previous_face = current_face
                            
                    else:
                        print("No known face encodings available.")
                        name = "Visitor..."  
                
                colorFrame1 = colorFrame
            
            colorVerify = colorFrame1    
            
            self.process_current_frame = not self.process_current_frame
            
            # places green/red frame around user face with text       
            for(top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                cv2.rectangle(frame, (left, top), (right, bottom), colorVerify, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), colorVerify, -1) 
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
                
            cv2.imshow('Face Recognition', frame)
            
            #key to terminate program
            if cv2.waitKey(1) == ord('q'):
                break
            
        video_capture.release()
        cv2.destroyAllWindows()
        

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()