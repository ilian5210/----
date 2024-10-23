import numpy as np
import cv2
def padding(input, padding):
    height = len(input)
    width = len(input[0])
    if padding == -1:
        paddingimg = np.copy(input)
        
    elif padding == 0:
        paddingimg = np.zeros((height + 2, width + 2), dtype=np.float64)
        for i in range(height + 2):
            for j in range(width + 2):
                if i == 0 or i == height + 1 or j == 0 or j == width + 1:
                    paddingimg[i][j] = 255
                else:
                    paddingimg[i][j] = input[i-1][j-1]

    elif padding == 1:
        paddingimg = np.zeros((height + 2, width + 2), dtype=np.float64)
        for i in range(height + 2):
            for j in range(width + 2):
                if i == 0 or i == height + 1 or j == 0 or j == width + 1:
                    paddingimg[i][j] = 0
                else:
                    paddingimg[i][j] = input[i-1][j-1]

    return paddingimg


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
            output[i//2][j//2] = max(input[i][j], input[i][j+1], input[i+1][j], input[i+1][j+1])
    return output

def avgpooling(input):
    height = len(input)
    width = len(input[0])
    output = np.zeros((height // 2, width // 2), dtype=np.float64)
    if height % 2 == 1:
        height -= 1
    if width % 2 == 1:
        width -= 1
    for i in range(0, height, 2):
        for j in range(0, width, 2):
            output[i//2][j//2] = (input[i][j] + input[i][j+1] + input[i+1][j] + input[i+1][j+1]) / 4
    return output

# 11*11 test
intput_img = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                   
                     [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],

                        [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],

                            [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],

                                [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],

                                    [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],

                                        [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77],

                                            [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88],

                                                [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],

                                                    [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],

                                                        [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]])


output = maxpooling(intput_img)
print(output)
output = avgpooling(intput_img)
print(output)

#odd padding test bigger than 10*10

print()
output = padding(intput_img, -1)
print(output)
print()
output = padding(intput_img, 0)
print(output)
print() 
output = padding(intput_img, 1)
print(output)
print()

sign = cv2.imread('imgs\IMG_3652.png', 0)
sign = cv2.resize(sign, (0, 0), fx=0.1, fy=.1)
#save as new picture
cv2.imwrite('imgs\sign1.png',sign)



