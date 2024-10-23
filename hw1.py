import numpy as np
import cv2


def conv(input,m,n,filter,bias,padding):
    #=========================== padding ===========================
    height = len(input)
    width = len(input[0])
    if padding == -1:
        paddingimg = np.copy(input)
        
    elif padding == 0:
        paddingimg = np.zeros((height + 2, width + 2), dtype=np.float64)
        for i in range(height + 2):
            for j in range(width + 2):
                if i == 0 or i == height + 1 or j == 0 or j == width + 1:
                    paddingimg[i][j] = 0
                else:
                    paddingimg[i][j] = input[i-1][j-1]

    elif padding == 1:
        paddingimg = np.zeros((height + 2, width + 2), dtype=np.float64)
        for i in range(height + 2):
            for j in range(width + 2):
                if i == 0 or i == height + 1 or j == 0 or j == width + 1:
                    paddingimg[i][j] = 255
                else:
                    paddingimg[i][j] = input[i-1][j-1]
                
    #=========================== convolution ===========================
    height = len(paddingimg)
    width = len(paddingimg[0])
    for i in range(height - m + 1):
        for j in range(width - n + 1):
            conv_sum = 0
            for k in range(m):
                for l in range(n):
                    conv_sum += paddingimg[i + k][j + l] * filter[k][l]
            paddingimg[i][j] = conv_sum + bias
    
    return paddingimg

#===========================maxpooling===========================
def maxpooling(input):
    height = len(input)
    width = len(input[0])
    output = np.zeros((height // 2, width // 2), dtype=np.float64)
    if height % 2 == 1:
        height -= 1
    if width % 2 == 1:
        width -= 1
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            output[i//2][j//2] = max(input[i][j] , input[i][j+1] , input[i+1][j] , input[i+1][j+1])
    return output


#===========================avgpooling===========================

def avgpooling(input_img):
    height = len(input_img)
    width = len(input_img[0])
    output = np.zeros((height // 2, width // 2), dtype=np.float64)
    if height % 2 == 1:
        height -= 1
    if width % 2 == 1:
        width -= 1
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            output[i//2][j//2] = (input_img[i][j] + input_img[i][j+1] + input_img[i+1][j] + input_img[i+1][j+1]) / 4
    return output
#=========================== mian ===========================


img = cv2.imread('imgs/IMG_5552.JPG', 0)


img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


avg = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
    ])

sobel = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
])

gaussian = np.array([
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
])

for padding in range(-1, 2):
    input_img = img
    m = 3
    n = 3
    bias = 0

    output = conv(input_img, m, n, avg, bias, padding)
    cv2.imshow('Average Filter with Padding {}'.format(padding), output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = conv(input_img, m, n, sobel, bias, padding)
    cv2.imshow('Sobel Filter with Padding {}'.format(padding), output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = conv(input_img, m, n, gaussian, bias, padding)
    cv2.imshow('Gaussian Filter with Padding {}'.format(padding), output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Padding = ', padding)

input_img = img

output = maxpooling(input_img)
cv2.imshow('maxpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

output = avgpooling(input_img)
cv2.imshow('avgpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

