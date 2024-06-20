from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np
from pytesseract import pytesseract
import glob
from fuzzywuzzy import process
from werkzeug.utils import secure_filename
import csv


absolute_path = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
upload_folder = "uploads/"
app.config["UPLOAD_FOLDER"] = upload_folder
IMG_PATH = f"{absolute_path}/uploads/images"

PROCESSED_PATH = f"{absolute_path}/uploads/extracted/"
IMG_PATHS = [IMG_PATH, PROCESSED_PATH]


class ImageSaveError(Exception):
    print(f"{Exception}")
    pass


def check_img_path():
    """
    Checks to make sure the proper paths have been created. Creates them if they don't exist
    """
    for path in IMG_PATHS:
        if not os.path.exists(path):
            os.makedirs(path)


def route():
    """
    Routing process for file upload
    """
    # routing / upload request
    if request.method == "POST":
        ign = request.files["ign"]
        vid = request.files["vid"]
        if ign.filename == "" or vid.filename == "":
            return redirect(request.url)
        ign.filename = "ign.txt"
        vid.filename = "vid.mp4"
        ign_filename = secure_filename(ign.filename)
        vid_filename = secure_filename(vid.filename)
        ign.save(os.path.join(app.config["UPLOAD_FOLDER"], ign_filename))
        vid.save(os.path.join(app.config["UPLOAD_FOLDER"], vid_filename))


def delete_old_images():
    """
    Deletes any png files from a previous run
    """
    for path in IMG_PATHS:
        files = glob.glob(f"{path}/*.png")
        for file in files:
            os.remove(file)


def create_images():
    """
        Loads vid.mp4 and creates separate images from them as separate screenshots to be evaluated
    Raises:
        ImageSaveError: Exception if an image is unable to be saved
    """
    # load and create images
    video_path = f"{absolute_path}/uploads/vid.mp4"
    cap = cv2.VideoCapture(video_path)
    i = 0
    ret, frame_prev = cap.read()

    # create starting frame
    result = cv2.imwrite(f"{absolute_path}/uploads/images/ss0.png", frame_prev)
    if result is False:
        raise ImageSaveError("Error while saving initial image.")
    i = 1

    screenshot_path = f"{absolute_path}uploads/images/ss"
    # creates the individual images
    while cap.isOpened():
        ret, frame_cur = cap.read()
        if ret is False:
            break
        diff = cv2.absdiff(frame_prev, frame_cur)
        mean_diff = np.mean(diff)
        if mean_diff > 3:
            cv2.imwrite(f"{screenshot_path}{i}.png", frame_cur)
            frame_prev = frame_cur
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def get_igns() -> list[str]:
    """
        Reads igns from path/to/ign.txt. Returns as a list.
    Returns:
        list[str]: the list of igns
    """
    path_to_igns = f"{absolute_path}/uploads/ign.txt"
    result = []
    with open(path_to_igns, "r", encoding="utf-8") as file:
        for line in file:
            if len(line) > 0 and line != '\n':
                result.append(line.replace("\n", ""))
    return result
        
        
def read_images() -> list[str]:
    """
        Reads and processes the screenshots located in uploads/images and extracts text from each image
        using the Tesseract OCR engine.
        Writes unfiltered results to log.txt
    Returns:
        list[str]: a list of all the processed guild member information
    """
    path_to_tesseract = r"Tesseract-OCR/tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    # Put path to list of IGNS here
    i: int = 0
    result = []
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
        cv2.imwrite(
            f"{absolute_path}/uploads/extracted/processed" + str(i) + ".png", img
        )
        i += 1
        # Extract text from the images and put it all into one string,
        # I found psm 6 to be the best at parsing the columns and
        # putting the data into rows
        # ----------------------------------------------------------------------
        # Changed original process to split the text return by pytesseract
        # and append the result to a text (Aren)
        # ----------------------------------------------------------------------
        text = pytesseract.image_to_string(img, config="--psm 6 -l eng")
        result += text.splitlines()

    # Writes unfiltered results to log.txt
    with open("log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(result))

    return list(filter(None, result))



def process_igns(list_of_igns: list[str]):
    """
        Processes the given list of igns and creates a .csv file containing each guild member with their
        culvert and flag scores
    Args:
        list_of_igns (list[str]): The list of igns to process
    """
    # Remove empty entries
    guild_info = read_images()

    seen_igns = []
    correct_results = []
    error_results = []
    duplicate_results = []
    ign_position = 0
    culvert_position = -2
    flag_position = -1

    # Extracts IGN, Culvert, and Flag Race numbers and compares parsed
    # IGN to the list of IGNs in the guild and finds the most similar match
    for x in range(0, len(guild_info)):
        player_info = guild_info[x].split(" ")
        ign = player_info[ign_position]

        # match, percent, index  = get_match_and_index(ign, indexed_igns)
        match, percent = process.extractOne(ign, list_of_igns)
        # Makes sure there are no dupes, in case multiple screenshots were taken
        # of the same set of members and IGNs match by 70%, if not then send to errors
        if match in seen_igns:
            continue
        
        if (len(ign) == 4 and percent >= 65) or percent >= 70:
            # Appends matched IGNs in format IGN Culvert Flag
            correct_results.append(
                [match, player_info[culvert_position], player_info[flag_position]]
            )
            seen_igns.append(match)
        # Duped IGNS
        # - Not sure if Dupes even matters, not sure what this tells us (Aren)
        elif match in seen_igns:
            duplicate_results.append(guild_info[x])
        else:  # Put IGNS that couldn't be matched into a list for debugging/manual solving
            error_results.append(guild_info[x])

    write_to_file(correct_results, "results.csv", True)
    write_to_file(error_results, "errors.csv")
    write_to_file(duplicate_results, "dupes.csv")


@app.route("/", methods=["POST"])
def upload_image() -> str: # not sure if this function name needs to stay as upload_images() (Aren)
    """
    This function serves as the core of the program, designed to extract and analyze data from uploaded images.

    The primary purpose of this function is to extract information about culvert and flag scores from images, 
    which are assumed to be frames extracted from a video. The extracted scores are then compared with a list 
    of in-game names (IGNs) provided in the 'igns.txt' file.

    The comparison results, along with the extracted scores, are compiled into a structured format and written 
    to a CSV file. This CSV file serves as the output of the function, providing a comprehensive record of the 
    analyzed scores matched with the corresponding IGNs.

    Returns:
        str: A string of rendered HTML indicating the completion of the image upload, processing, and analysis.
    """
    check_img_path()
    route()
    create_images()
    process_igns(get_igns())  # holds the entire list of igns found in ign.txt
    return render_template("done.html")


def write_to_file(output: list, filename: str, delim=",", clean=False):
    """Writes the individual lists of character culv info into csv format
    Args:
        v (list): The list containing all the lists of files
        filename (str): the filename to write to
        delim (str): the delimeter for writing ',' by default
    """
    with open(filename, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        if clean is True:
            for row in output:
                cleaned_output = [
                    cell.replace("..", " ")
                    .replace(".", ",")
                    .replace("1]", "0")
                    .replace("1}", "0")
                    .replace("\n", "")
                    for cell in row
                ]
                writer.writerow(cleaned_output)
        else:
            writer.writerows(output)


@app.route("/", methods=["GET"])
def land():
    return render_template("main.html")


@app.route("/done", methods=["GET"])
def done():
    return render_template("done.html")


app.run(port=5000)
