import sys
import getopt
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import os
import csv



characters = ['0','1','2','3','4','5','6','7','8','9',
'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


# Take a path from the command line
def get_input_path(argv):
    arg_input = ""

    try:
        opts, args = getopt.getopt(argv[1:], "hi:", ["help", "input="])
    except getopt.GetoptError:
        print("Usage: {} -i <input>".format(argv[0]))
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: {} -i <input>".format(argv[0]))  # Print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            arg_input = arg

    return arg_input


# make a prediction for one picture
def predict_single_image(path_to_pic, picture_name, model):
    
    user_test = os.path.join(path_to_pic, picture_name)
    col = Image.open(user_test)
    gray = col.convert('L')
    bw = gray.point(lambda x: 0 if x<100 else 255, '1')
    path_to_bw_pic = os.path.join(path_to_pic, 'bw', picture_name)
    bw.save(path_to_bw_pic)
    
    img_array = cv2.imread(path_to_bw_pic, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.bitwise_not(img_array)

    img_size = 28
    new_array = cv2.resize(img_array, (img_size,img_size))
     
    new_array = np.expand_dims(new_array, axis=0)
    user_test = tf.keras.utils.normalize(new_array, axis = 1)
    
    # verbose param hides the prediction progress bar
    predicted = model.predict([[user_test]], verbose=0)
    pred_char = ord(characters[np.argmax(predicted[0])])

    return '{:0>3}, {}, {}'.format(pred_char, characters[np.argmax(predicted[0])], os.path.join(path_to_pic_def, picture_name).replace("\\", "/"))
    

# takes one CLI argument that is path to directory with image samples and print output to console in CSV format
# Program must find all images in directory
if __name__ == "__main__":
    # load trained model
    model = tf.keras.models.load_model('./app/model.h5')

    # Current dir
    current_directory = os.path.abspath(os.getcwd()).replace("\\", "/")

    # Get the input path
    #  “[character ASCII index in decimal format], [POSIX path to image sample]”
    path_to_pic_def = get_input_path(sys.argv)

    # full path
    path_dir = current_directory + path_to_pic_def

    predictions = []
    for filename in os.listdir(path_dir):
        f = os.path.join(path_dir, filename).replace("\\", "/")

        # checking if it is a file
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']:
            prediction = predict_single_image(path_to_pic=path_dir, picture_name=filename, model=model)
            print(prediction)
            predictions.append(prediction)

    # Save predictions to a CSV file
    output_file = 'predictions.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Prediction', 'Image Path'])
        for prediction in predictions:
            writer.writerow(prediction.split(', '))

    print("Predictions saved to", output_file)
