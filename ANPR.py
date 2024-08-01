from ultralytics import YOLO

import cv2

from PIL import Image, ExifTags
import os

import numpy as np
import easyocr
from datetime import datetime

import csv

predModelANPR = YOLO("<path of included best.pt>")

# path of image
truckNo = '2'
ext = '.jpg'

inputDir = './DemoInputs'
inputSide = inputDir + '/Truck' + truckNo + 'side' + ext

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def crop_single_image(input_path, output_left_path, output_right_path):
    # Open an image file
    with Image.open(input_path) as img:
        # Correct the orientation of the image
        img = correct_orientation(img)

        # Get image dimensions
        width, height = img.size
        # Define box for left half (left, upper, right, lower)
        left_half = img.crop((0, 0, width // 2, height))
        # Define box for right half (left, upper, right, lower)
        right_half = img.crop((width // 2, 0, width, height))

        # Save the cropped images
        if not (os.path.exists(output_left_path)):
            left_half.save(output_left_path)
            right_half.save(output_right_path)

# Example usage
leftSide = inputDir + '/Truck' + truckNo + 'sideLeft' + ext
rightSide = inputDir + '/Truck' + truckNo + 'sideRight' + ext

crop_single_image(inputSide, leftSide, rightSide)

def correct_text(text):
    char_mapping = {
        'O': '0', '0': 'O',
        'o': '0',
        'l': '1',
        'I': '1', '1': 'I',
        'i': '1',
        'j': '1',
        'L': '1',
        'J': '1',
        'Z': '2', '2': 'Z',
        'E': '3', '3': 'E',
        'A': '4', '4': 'A',
        'S': '5', '5': 'S',
        's': '5',
        'G': '6', '6': 'G',
        'b': '6',
        'T': '7', '7': 'T',
        'Z': '7',
        'F': '7',
        'B': '8', '8': 'B',
        'g': '9', '9': 'g'
    }
    
    final_text = ""
    for char in text:
        if char.isalnum():
            final_text += char
            
    size = len(final_text)
    corrected_text = ''
    
    for i in range(2):
        if not final_text[i].isalpha():
            corrected_text += char_mapping[final_text[i]].upper()
        else:
            corrected_text += final_text[i].upper()
            
    for i in range(2, 4):
        if not final_text[i].isdigit():
            corrected_text += char_mapping[final_text[i]].upper()
        else:
            corrected_text += final_text[i].upper()
    
    var = -1
    if size == 11:
        var = 3
    elif size == 10:
        var = 2
    elif size == 9:
        var = 1
    else:
        var = 0
    
    for i in range(4, 4 + var):
        if not final_text[i].isalpha():
            corrected_text += char_mapping[final_text[i]].upper()
        else:
            corrected_text += final_text[i].upper()
    
    for i in range(4 + var, size):
        if not final_text[i].isdigit():
            corrected_text += char_mapping[final_text[i]].upper()
        else:
            corrected_text += final_text[i].upper()
    
    return corrected_text


# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Run YOLOv8 detection
results = predModelANPR(leftSide)
image = cv2.imread(leftSide)
if (len(results[0].boxes) == 0):
  results = predModelANPR(rightSide)
  image = cv2.imread(rightSide)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.equalizeHist(gray)

# Extract bounding boxes and confidences
box = results[0].boxes[0]
final_text = ""
x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
confidence = box.conf[0]

# Crop the region of interest (ROI) from the image
roi = image[y1:y2, x1:x2]

# Use EasyOCR to detect text in the ROI
text_results = reader.readtext(roi)

# Draw bounding box and text on the image
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
for (bbox, text, prob) in text_results:
    (tl, tr, br, bl) = bbox
    tl = tuple(map(int, tl))
    tr = tuple(map(int, tr))
    br = tuple(map(int, br))
    bl = tuple(map(int, bl))
    final_text += text
    cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.polylines(image, [np.array([tl, tr, br, bl], np.int32)], True, (0, 255, 255), 2)

plate_number = correct_text(final_text)

def append_to_csv(image_path_side, plate_number, csv_file='output.csv'):
    # Get the current date and time
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Define the row to be appended
    row = [image_path_side, current_datetime, plate_number]

    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Check if the file is empty to write the header
        if file.tell() == 0:
            header = ['Image Path Top', 'Image Path Side', 'Date and Time', 'Plate Number', 'Count']
            writer.writerow(header)

        # Append the row to the CSV file
        writer.writerow(row)

append_to_csv(inputSide, plate_number)
