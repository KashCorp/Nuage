import cv2
import numpy as np
import time

# Define reference images and thresholds
template1 = cv2.imread('1.jpg', 0)
template2 = cv2.imread('2.jpg', 0)
template3 = cv2.imread('3.jpg', 0)
template4 = cv2.imread('4.jpg', 0)
templateShapes = [template1, template2, template3, template4]
thresholds = [0.1, 0.1, 0.1, 0.1]

# Define kernel for morphology operation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Define function to preprocess image
def preprocessImage(img):
    assert isinstance(img, np.ndarray), "Invalid input image type"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq_gray = cv2.equalizeHist(gray)
    closed = cv2.morphologyEx(eq_gray, cv2.MORPH_CLOSE, kernel)
    return closed

# Define function to recognize image
def recognizeImage(closed):
    result_images = []
    for i, template in enumerate(templateShapes):
        # Check if template image has contours
        contours_template = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(contours_template) > 0:
            contours_template = contours_template[0]
        else:
            print("No contours found in template image")
            continue  # Skip to the next template
        
        # Check if closed image has contours
        contours_closed = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(contours_closed) > 0:
            contours_closed = contours_closed[0]
        else:
            print("No contours found in closed image")
            continue  # Skip to the next template

        # Check if contours have the expected shape
        if len(contours_closed.shape) != 2 or len(contours_template.shape) != 2:
            print("Invalid contours shape")
            continue  # Skip to the next template

        # Check if input image has the expected type
        if closed.ndim != 2 or template.ndim != 2:
            print("Invalid input image type")
            continue  # Skip to the next template

        score = cv2.matchShapes(contours_closed, contours_template, cv2.CONTOURS_MATCH_I2, 0)
        if score < thresholds[i]:
            if i == 0:
                result_images.append(cv2.putText(closed.copy(), "Storm", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2))
                print("Storm detected")
            elif i == 1:
                result_images.append(cv2.putText(closed.copy(), "Thunder", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2))
                print("Thunder detected")
            elif i == 2:
                result_images.append(cv2.putText(closed.copy(), "Striking", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2))
                print("Striking detected")
            elif i == 3:
                result_images.append(cv2.putText(closed.copy(), "Rain", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2))
                print("Rain detected")
    return result_images

# Define function to crop image to 1:1 ratio
def cropImage(img):
    height, width = img.shape[:2]
    if height > width:
        start = int((height - width) / 2)
        cropped = img[start:start+width, :]
    else:
        start = int((width - height) / 2)
        cropped = img[:, start:start+height]
    return cropped

# Initialize video capture
cap = cv2.VideoCapture(0)

# Loop to capture image every 5 seconds and recognize shape
while True:
    # Capture image from camera
    ret, img = cap.read()
    assert ret, "Error capturing image from camera"
    
    # Check if image was successfully captured
    if not ret:
        print("Error capturing image from camera")
        break
    
    # Crop image to 1:1 aspect ratio
    height, width = img.shape[:2]
    crop_size = min(height, width)
    x = int((width - crop_size) / 2)
    y = int((height - crop_size) / 2)
    cropped_img = img[y:y+crop_size, x:x+crop_size]
    
    # Save the captured image
    cv2.imwrite('captured_image.jpg', cropped_img)

    # Crop image to 512x512
    cropped_image_resized = cv2.resize(cropped_img, (512, 512))
    
    # Preprocess captured image
    closed = preprocessImage(cropped_image_resized)
    
    # Save the preprocessed image
    cv2.imwrite('preprocessed_image.jpg', closed)
    
    # Recognize shape in the preprocessed image
    result_images = recognizeImage(closed)
    
    # Save the result images
    if result_images is not None:
        for i, result_image in enumerate(result_images):
            cv2.imwrite('result_image_{}.jpg'.format(i), result_image)
    
    # Wait for 5 seconds before taking the next image
    time.sleep(5)
    
# Release video capture
cap.release()
