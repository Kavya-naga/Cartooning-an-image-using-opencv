import cv2
import numpy as np

# Function to cartoonify an image with adjustable parameters
def cartoonify_image(image, ddepth=cv2.CV_8U, ksize=9, sigmaColor=150, sigmaSpace=150, edge_threshold=40):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur to smoothen the image
    gray = cv2.medianBlur(gray, 5)
    
    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ksize, sigmaColor)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    color = cv2.bilateralFilter(image, ddepth, sigmaColor, sigmaSpace)
    
    # Combine the edges with the color image using bitwise AND
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    return cartoon

# Load the input image
image_path = 'C:/Users/mohit/Downloads/faces1.jpg'  # Replace 'example_image.jpg' with the path to your image file
image = cv2.imread(image_path)

# Adjust parameters for clarity
cartoon_image = cartoonify_image(image, sigmaColor=200, sigmaSpace=200, edge_threshold=30)

# Display the cartoonified image
cv2.imshow('Cartoonified Image', cartoon_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
