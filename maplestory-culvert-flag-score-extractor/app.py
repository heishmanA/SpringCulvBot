from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np
from pytesseract import pytesseract
import glob
from fuzzywuzzy import process
from werkzeug.utils import secure_filename
import csv



app = Flask(__name__)
upload_folder = "uploads/"
app.config['UPLOAD_FOLDER'] = upload_folder
IMG_PATH = 'uploads/images'
SS_PATH = 'uploads/images/ss'
PROCESSED_PATH = "uploads/extracted/"
IMG_PATHS = [IMG_PATH, PROCESSED_PATH]


class ImageSaveError(Exception):
    pass

@app.route("/", methods=['POST'])

def upload_image():

    # create the paths if they do not exist
    for path in IMG_PATHS:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # routing / upload request
    if request.method == 'POST':
        ign = request.files['ign']
        vid = request.files['vid']
        if ign.filename == '' or vid.filename == '':
            return redirect(request.url)
        ign.filename = 'ign.txt'
        vid.filename = 'vid.mp4'
        ign_filename = secure_filename(ign.filename)
        vid_filename = secure_filename(vid.filename)
        ign.save(os.path.join(app.config['UPLOAD_FOLDER'], ign_filename))
        vid.save(os.path.join(app.config['UPLOAD_FOLDER'], vid_filename))

    # delete any old images that may have been created
    for pathh in IMG_PATHS:
        files = glob.glob(f"{path}/*.png")
        for file in files:
            os.remove(file)
    # load and create images
    cap = cv2.VideoCapture('uploads/vid.mp4')
    i=0
    ret, frame_prev = cap.read()
    
    # create starting frame
    result = cv2.imwrite('uploads/images/ss0.png',frame_prev)
    if result is False:
        raise ImageSaveError("Error while saving initial image.")
    i=1
    
    # creates the individual images 
    while(cap.isOpened()):
        ret, frame_cur = cap.read()
        if ret is False:
            break
        diff = cv2.absdiff(frame_prev,frame_cur)
        mean_diff = np.mean(diff)
        if mean_diff > 3:
            #cv2.imwrite('uploads/images/ss'+str(i)+'.png',frame_cur)
            cv2.imwrite(f"{SS_PATH}{i}.png", frame_cur)
            frame_prev = frame_cur
        i+=1

    cap.release()
    cv2.destroyAllWindows()

    path_to_tesseract = r"Tesseract-OCR/tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    # Put path to list of IGNS here
    path_to_igns = "uploads/ign.txt"
    text = ""
    i=0
    # Read list of IGNs
    igns = open(path_to_igns, "r")
    list_of_igns = igns.read()
    list_of_igns = list_of_igns.splitlines()
    # Iterating through every screenshot that was taken and performing
    # image processing to make the images easier to parse
    for x in glob.glob("uploads/images/*.png"):
        img = cv2.imread(x)
        # Resizing and making the images bigger
        img = cv2.resize(img, None, fx=10, fy=10)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Applying Gaussian blur
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # Using Otsu thresholding to binarize, partitions image into foreground
        # and background
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # Applying erosion
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        cv2.imwrite('uploads/extracted/processed'+str(i)+'.png', img)
        i+=1
        # Extract text from the images and put it all into one string,
        # I found psm 6 to be the best at parsing the columns and 
        # putting the data into rows
        text += pytesseract.image_to_string(img, config='--psm 6 -l eng') + "\n"


    # Writes unfiltered results to log.txt
    with open('log.txt', "w", encoding="utf-8") as f:
        f.write(text)

    # Formatting to prepare to extract data
    array = text.splitlines()

    # Remove empty entries
    array = list(filter(None, array))

    seen_igns = []
    res = []
    errors = []
    dupes = []
    
    # Extracts IGN, Culvert, and Flag Race numbers and compares parsed
    # IGN to the list of IGNs in the guild and finds the most similar match
    for x in range(0, len(array)):
        array[x] = array[x].split()
        ign = array[x][0]
        match, percent = process.extractOne(ign, list_of_igns)
        # Makes sure there are no dupes, in case multiple screenshots were taken
        # of the same set of members and IGNs match by 70%, if not then send to errors
        if match not in seen_igns and percent > 70:
            # Appends matched IGNs in format IGN Culvert Flag
            res.append([match, array[x][-2], array[x][-1]])
            seen_igns.append(match)
        # Duped IGNS
        elif match in seen_igns:
            dupes.append(array[x])
        else:    # Put IGNS that couldn't be matched into a list for debugging/manual solving
            errors.append(array[x])
    
    write_to_file(res, 'results.csv', True)
    write_to_file(errors, 'errors.csv', True)
    write_to_file(dupes, 'dupes.csv', True)
    
    return render_template('done.html')


def write_to_file(output: list, filename: str, delim=',', clean=False):
    """ Writes the individual lists of character culv info into csv format
    Args:
        v (list): The list containing all the lists of files 
        filename (str): the filename to write to
        delim (str): the delimeter for writing ',' by default
    """
    with open(filename, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        if clean is True:
            for row in output:
                cleaned_output = [cell.replace('..', ' ').replace('.', ',').replace('1]', '0').replace('1}', '0').replace('\n', '') for cell in row]
                writer.writerow(cleaned_output)
        else:
            writer.writerows(output)

@app.route("/", methods=['GET'])
def land():
    return render_template('main.html')

@app.route("/done", methods=['GET'])
def done():
    return render_template('done.html')
    
app.run(port=5000)