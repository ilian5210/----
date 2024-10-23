import numpy as np
import cv2

def conv(input, m, n, filter, bias, padding):
    height = len(input)
    width = len(input[0])
    
    if padding == -1:
        paddingimg = np.copy(input)
    elif padding == 0:
        paddingimg = np.pad(input, ((1, 1), (1, 1)), 'constant', constant_values=0)
    elif padding == 1:
        paddingimg = np.pad(input, ((1, 1), (1, 1)), 'constant', constant_values=255)
    
    output = np.zeros((height, width), dtype=np.float64)  # 使用新的 output
    
    for i in range(height):
        for j in range(width):
            conv_sum = 0
            for k in range(m):
                for l in range(n):
                    if i + k < height and j + l < width:
                        conv_sum += paddingimg[i + k][j + l] * filter[k][l]
            output[i][j] = conv_sum + bias
    
    # Clip and convert to uint8 for display
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output

def maxpooling(input):
    height = len(input)
    width = len(input[0])
    output = np.zeros((height // 2, width // 2), dtype=np.float64)
    
    for i in range(0, height - 1, 2):
        for j in range(0, width - 1, 2):
            output[i//2][j//2] = max(input[i][j], input[i][j+1], input[i+1][j], input[i+1][j+1])
    
    return output.astype(np.uint8)

def avgpooling(input):
    height = len(input)
    width = len(input[0])
    output = np.zeros((height // 2, width // 2), dtype=np.float64)
    
    for i in range(0, height - 1, 2):
        for j in range(0, width - 1, 2):
            output[i//2][j//2] = (input[i][j] + input[i][j+1] + input[i+1][j] + input[i+1][j+1]) / 4
    
    return output.astype(np.uint8)

# Main logic remains the same, but added normalization for display
img = cv2.imread('imgs/IMG_5552.JPG', 0)
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

avg = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
sobel = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]])
gaussian = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])

for padding in range(-1, 2):
    output = conv(img, 3, 3, avg, 0, padding)
    cv2.imshow(f'Average Filter with Padding {padding}', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = conv(img, 3, 3, sobel, 0, padding)
    cv2.imshow(f'Sobel Filter with Padding {padding}', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = conv(img, 3, 3, gaussian, 0, padding)
    cv2.imshow(f'Gaussian Filter with Padding {padding}', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

output = maxpooling(img)
cv2.imshow('Maxpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

output = avgpooling(img)
cv2.imshow('Avgpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
