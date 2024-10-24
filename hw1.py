import numpy as np
import cv2

def signmatrix(sign):
    height = len(sign)
    width = len(sign[0])
    for i in range(height):
        for j in range(width):
            if sign[i][j] < 128:
                sign[i][j] = 0
            else:
                sign[i][j] = 1
    return sign

def printsign(sign, output, filter):
    height = len(output)
    width = len(output[0])
    signheight = len(sign)
    signwidth = len(sign[0])
    for i in range(signheight):
        for j in range(signwidth):
            if sign[i][j] == 1:
                if filter == 'sobel' or filter == 'avgpooling':
                    output[height - signheight + i][width - signwidth + j] = 255
                else:
                    output[height - signheight + i][width - signwidth + j] = 0
    return output

def conv(input, m, n, filter, bias, padding):
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

def pooling(input, size, stride, type):
    
    if type == 'max':
        height = len(input)
        width = len(input[0])
        output = np.zeros((height // size, width // size), dtype=np.float64)
        
        for i in range(0, height - 1, stride):
            for j in range(0, width - 1, stride):
                output[i//2][j//2] = max(input[i][j], input[i][j+1], input[i+1][j], input[i+1][j+1])
        
        return output.astype(np.uint8)

    elif type == 'avg':
        height = len(input)
        width = len(input[0])
        output = np.zeros((height // size, width // size), dtype=np.float64)
        
        for i in range(0, height - 1, stride):
            for j in range(0, width - 1, stride):
                output[i//2][j//2] = (input[i][j] + input[i][j+1] + input[i+1][j] + input[i+1][j+1]) / 4
        
        return output.astype(np.uint8)

# Main logic remains the same, but added normalization for display

sign = cv2.imread('imgs/sign1.png', 0)
sign = signmatrix(sign)

img = cv2.imread('imgs/IMG_5552-2.JPG', 1)
img = printsign(sign,img,'original')
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = cv2.imread('imgs/IMG_5552-2.JPG', 0)

img1 = cv2.imread('imgs/tempImageEAu4Vg.jpg', 1)
img1 = printsign(sign,img1,'original')
cv2.imshow('Original Image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
img1 = cv2.imread('imgs/tempImageEAu4Vg.jpg', 0)

img3 = cv2.imread('imgs/IMG_5912.JPG', 1)
img3 = printsign(sign,img3,'original')
cv2.imshow('Original Image', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
img3 = cv2.imread('imgs/IMG_5912.JPG', 0)


avg = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
sobel = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]])
gaussian = np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]])

for padding in range(-1, 2):
    output = conv(img, 3, 3, avg, 0, padding)
    output = printsign(sign,output,'avg')
    cv2.imshow(f'Average Filter with Padding {padding}', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = conv(img, 3, 3, sobel, 0, padding)
    output = printsign(sign,output,'sobel')
    cv2.imshow(f'Sobel Filter with Padding {padding}', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = conv(img, 3, 3, gaussian, 0, padding)
    output = printsign(sign,output,'gaussian')
    cv2.imshow(f'Gaussian Filter with Padding {padding}', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = conv(img1, 3, 3, avg, 0, padding)
    output = printsign(sign,output,'avg')
    cv2.imshow(f'Average Filter with Padding {padding}', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = conv(img1, 3, 3, sobel, 0, padding)
    output = printsign(sign,output,'sobel')
    cv2.imshow(f'Sobel Filter with Padding {padding}', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output = conv(img1, 3, 3, gaussian, 0, padding)
    output = printsign(sign,output,'gaussian')
    cv2.imshow(f'Gaussian Filter with Padding {padding}', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out = conv(img3, 3, 3, avg, 0, padding)
    out = printsign(sign,out,'avg')
    cv2.imshow(f'Average Filter with Padding {padding}', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out = conv(img3, 3, 3, sobel, 0, padding)
    out = printsign(sign,out,'sobel')
    cv2.imshow(f'Sobel Filter with Padding {padding}', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out = conv(img3, 3, 3, gaussian, 0, padding)
    out = printsign(sign,out,'gaussian')
    cv2.imshow(f'Gaussian Filter with Padding {padding}', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

output = pooling(img,2,2,'max')
output = printsign(sign,output,'maxpooling')
cv2.imshow('Maxpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

output = pooling(img,2,2,'avg')
output = printsign(sign,output,'avgpooling')
cv2.imshow('Avgpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

output = pooling(img1,2,2,'max')
output = printsign(sign,output,'maxpooling')
cv2.imshow('Maxpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

output = pooling(img1,2,2,'avg')
output = printsign(sign,output,'avgpooling')
cv2.imshow('Avgpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

output = pooling(img3,2,2,'max')
output = printsign(sign,output,'maxpooling')
cv2.imshow('Maxpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

output = pooling(img3,2,2,'avg')
output = printsign(sign,output,'avgpooling')
cv2.imshow('Avgpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


