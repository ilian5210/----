import numpy as np
import cv2

def signmatrix(sign):
    #把sign矩阵转换为0，1矩阵
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
    #把sign矩阵印在output右下角
    height = len(output)
    width = len(output[0])
    signheight = len(sign)
    signwidth = len(sign[0])
    for i in range(signheight):
        for j in range(signwidth):
            if sign[i][j] == 1:
                if filter == 'sobel':
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
img = cv2.imread('imgs\IMG_5552.JPG', 0)
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

sign = cv2.imread('imgs\IMG_3652.png', 0)
#save as new picture
# cv2.imwrite('imgs\sign.png',sign)

sign = signmatrix(sign)




avg = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
sobel = np.array([[-1/9,  0,  1/9], [-2/9,  0,  2/9], [-1/9,  0,  1/9]])
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

output = maxpooling(img)
cv2.imshow('Maxpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

output = avgpooling(img)
cv2.imshow('Avgpooling', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
