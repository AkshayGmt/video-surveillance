from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pandas as pd
from collections import deque
from moviepy.editor import *
import cv2
import numpy as np
import cv2
import numpy as np
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



            # Load Yolo

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
 
# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 30
 
# Specify the directory containing the UCF50 dataset. 
DATASET_DIR = "Dataset"
 
# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["abuse","arrest","arson","assault","burglary","explosions", "fights","roadaccident","robbery","running","shooting","shopifting","stealing","vandalism","walking"]
#CLASSES_LIST = ["walking", "fights", "running"]


model = load_model('Suspicious_Human_Activity_Detection_LRCN_Model.h5')
video_file_path=["gn.mp4"]

for i in range(len(video_file_path)):
    def predict_single_action(video_file_path, SEQUENCE_LENGTH):
        '''
        This function will perform single action recognition prediction on a video using the LRCN model.
        Args:
        video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
        SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
        '''

        # Initialize the VideoCapture object to read from the video file.
        video_reader = cv2.VideoCapture(video_file_path[i])

        # Get the width and height of the video.
        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Declare a list to store video frames we will extract.
        frames_list = []
        
        # Initialize a variable to store the predicted action being performed in the video.
        predicted_class_name = ''

        # Get the number of frames in the video.
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the interval after which frames will be added to the list.
        skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)
        success, frame = video_reader.read()

        # Iterating the number of times equal to the fixed length of sequence.
        for frame_counter in range(SEQUENCE_LENGTH):

            # Set the current frame position of the video.
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

            # Read a frame.
            
           

            # Check if frame is not read properly then break the loop.
            if not success:
                break

            # Resize the Frame to fixed Dimensions.
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            
            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
            normalized_frame = resized_frame / 255
            
            # Appending the pre-processed frame into the frames list
            frames_list.append(normalized_frame)

        # Passing the  pre-processed frames to the model and get the predicted probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis = 0))[0]

        # Get the index of class with highest probability.
        predicted_label = np.argmax(predicted_labels_probabilities)

        # Get the class name using the retrieved index.
        predicted_class_name = CLASSES_LIST[predicted_label]
        if predicted_class_name=="normal":
            print("Not detected Suspicious activity:",predicted_class_name)
        elif predicted_class_name=="roadaccident":
            print(" detected Suspicious activity:",predicted_class_name)
            classes = ["Car"]
            #cap = cv2.VideoCapture("gn.mp4")
            a=len(frame)
            while True:
                       
                height, width, channels = frame.shape 
            
                # width = 512
                # height = 512

                #Detecting objects
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                
                layer_names = net.getLayerNames()

                output_layers = [layer_names[n - 1] for n in net.getUnconnectedOutLayers()]
                colors = np.random.uniform(0, 255, size=(len(classes), 3))
                outs = net.forward(output_layers)

                # Showing information on the screen
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    print(indexes)
                    if indexes == 1: print("weapon detected in frame")
                    font = cv2.FONT_HERSHEY_PLAIN
                    for n in range(len(boxes)):
                        if n in indexes:
                            x, y, w, h = boxes[n]
                            label = str(classes[class_ids[n]])
                            color = colors[class_ids[n]]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

                    #frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    cv2.imshow("Image", frame)
                    if cv2.waitKey(300) == ord("q"):
                            break
                    
                video_reader.release()
                cv2.destroyAllWindows()
                break

        elif predicted_class_name=="fight":
            classes = ["Weapon"]
            #cap = cv2.VideoCapture("gn.mp4")
            a=len(frame)
            while True:
                       
                height, width, channels = frame.shape 
            
                # width = 512
                # height = 512

                #Detecting objects
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                
                layer_names = net.getLayerNames()

                output_layers = [layer_names[n - 1] for n in net.getUnconnectedOutLayers()]
                colors = np.random.uniform(0, 255, size=(len(classes), 3))
                outs = net.forward(output_layers)

                # Showing information on the screen
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    print(indexes)
                    if indexes == 1: print("weapon detected in frame")
                    font = cv2.FONT_HERSHEY_PLAIN
                    for n in range(len(boxes)):
                        if n in indexes:
                            x, y, w, h = boxes[n]
                            label = str(classes[class_ids[n]])
                            color = colors[class_ids[n]]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    cv2.imshow("Image", frame)
                    if cv2.waitKey(1) == ord("q"):
                            break
                    
                video_reader.release()
                cv2.destroyAllWindows()

            print("nothing")
        elif predicted_class_name=="robbery":
            classes = ["Weapon"]
            #cap = cv2.VideoCapture("gn.mp4")
            a=len(frame)
            while True:
                       
                height, width, channels = frame.shape 
            
                # width = 512
                # height = 512

                #Detecting objects
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                
                layer_names = net.getLayerNames()

                output_layers = [layer_names[n - 1] for n in net.getUnconnectedOutLayers()]
                colors = np.random.uniform(0, 255, size=(len(classes), 3))
                outs = net.forward(output_layers)

                # Showing information on the screen
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    print(indexes)
                    if indexes == 1: print("weapon detected in frame")
                    font = cv2.FONT_HERSHEY_PLAIN
                    for n in range(len(boxes)):
                        if n in indexes:
                            x, y, w, h = boxes[n]
                            label = str(classes[class_ids[n]])
                            color = colors[class_ids[n]]
                     
                     
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    cv2.imshow("Image", frame)
                    if cv2.waitKey(1) == ord("q"):
                            break
                    
                video_reader.release()
                cv2.destroyAllWindows()

        elif predicted_class_name=="arrest":
            classes = ["person"]
            #cap = cv2.VideoCapture("gn.mp4")
            a=len(frame)
            while True:
                       
                height, width, channels = frame.shape 
            
                # width = 512
                # height = 512

                #Detecting objects
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                
                layer_names = net.getLayerNames()

                output_layers = [layer_names[n - 1] for n in net.getUnconnectedOutLayers()]
                colors = np.random.uniform(0, 255, size=(len(classes), 3))
                outs = net.forward(output_layers)

                # Showing information on the screen
                class_ids = []
                confidences = []
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    print(indexes)
                    if indexes == 1: print("weapon detected in frame")
                    font = cv2.FONT_HERSHEY_PLAIN
                    for n in range(len(boxes)):
                        if n in indexes:
                            x, y, w, h = boxes[n]
                            label = str(classes[class_ids[n]])
                            color = colors[class_ids[n]]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    cv2.imshow("Image", frame)
                    if cv2.waitKey(1) == ord("q"):
                            break
                    
                video_reader.release()
                cv2.destroyAllWindows()

                

                        
        cv2.putText(frame, predicted_class_name, (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 
                        lineType=cv2.LINE_AA)

        
        
        # Display the predicted action along with the prediction confidence.
        print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        cv2.imshow('image', frame)
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        #     break  
        # # Release the VideoCapture object. 
        video_reader.release()
    predict_single_action(video_file_path,SEQUENCE_LENGTH)
