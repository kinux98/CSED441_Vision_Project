import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import gc 
from IPython.display import clear_output


def load_img(filename):
    img = cv2.imread(filename)
    data = np.array(img)
    w, h, _ = img.shape
    return data, w, h

def recover_img(prev_data, pix_data, x, y, h1, h2):
    data = prev_data
    i = 0
    for col in range(x, x+h2): 
        j = 0
        for row in range(y, y+h1):
            data[row][col][0]= pix_data[j][i][0]
            data[row][col][1]= pix_data[j][i][1]
            data[row][col][2]= pix_data[j][i][2]
            j += 1
        i += 1 
    return data

def get_random_crop(image, crop_height, crop_width, h, w):

    max_x = w - crop_width
    max_y = h - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    #print(w, h)
    #print(crop_width, crop_height)
    #print(max_x, max_y)
    #print(x, x+crop_width, ", ", y, y+crop_height)

    return crop, x, y

class DataGenerator():
    def __init__(self, dir_list):
        self.dir_list = dir_list

    def pixelate(self):
        for dir_name in self.dir_list:
            img_dir = "./cropped_result/"+dir_name
            file_list = os.listdir(img_dir)
            print(img_dir)
            for cnt, file_name in enumerate(file_list):
                if "png" in file_name:
                    print(img_dir+file_name)
                    tmp = load_img(img_dir+file_name)
                    data = tmp[0]
                    original = data
                    w = tmp[1]
                    h = tmp[2]

                    print(cnt, " of ", len(file_list))
                    if(cnt % 10 == 0 and cnt != 0):
                        clear_output()

                    r_w = np.random.randint(int(w*0.3), int(w*0.75), 1)[0]
                    r_h = np.random.randint(int(h*0.3), int(h*0.75), 1)[0]

                    random_crop=None
                    while True:
                        random_crop, x_l, y_b = get_random_crop(data, r_h, r_w, h, w)
                        if(np.shape(random_crop)[0] == 0 or np.shape(random_crop)[1] == 0 or np.shape(random_crop)[0]== 0):
                            print("Something was wrong!, try again...")
                            continue
                        else:
                            break

                    #print(np.shape(random_crop))
                    h1 = np.shape(random_crop)[0]
                    h2 = np.shape(random_crop)[1]
                    pix_w =  max([int(round(h2 * 0.1)), 1])
                    pix_h =  max([int(round(h1 * 0.1)), 1])

                    temp1 = cv2.resize(random_crop, (pix_w, pix_h), interpolation=cv2.INTER_LINEAR)
                    temp2 = cv2.resize(random_crop, (pix_w, pix_h), interpolation=cv2.INTER_AREA)
                    temp3 = cv2.resize(random_crop, (pix_w, pix_h), interpolation=cv2.INTER_CUBIC)
                    temp4 = cv2.resize(random_crop, (pix_w, pix_h), interpolation=cv2.INTER_NEAREST)

                    output1 = cv2.resize(temp1, (h2, h1), interpolation=cv2.INTER_NEAREST)
                    output2 = cv2.resize(temp2, (h2, h1), interpolation=cv2.INTER_NEAREST)
                    output3 = cv2.resize(temp3, (h2, h1), interpolation=cv2.INTER_NEAREST)
                    output4 = cv2.resize(temp4, (h2, h1), interpolation=cv2.INTER_NEAREST)

                    if not(os.path.isdir("resize_result")):
                        os.makedirs(os.path.join("resize_result")) 
                    if not(os.path.isdir("resize_result/"+dir_name)):
                        os.makedirs(os.path.join("resize_result/"+dir_name)) 

                    cv2.imwrite("resize_result/" + dir_name + file_name +"_resize1.png", output1)
                    cv2.imwrite("resize_result/" + dir_name + file_name +"_resize2.png", output2)
                    cv2.imwrite("resize_result/" + dir_name + file_name +"_resize3.png", output3)
                    cv2.imwrite("resize_result/" + dir_name + file_name +"_resize4.png", output4)

                    if not(os.path.isdir("pixelate_result")):
                        os.makedirs(os.path.join("pixelate_result")) 
                    if not(os.path.isdir("pixelate_result/"+dir_name)):
                        os.makedirs(os.path.join("pixelate_result/"+dir_name)) 

                    rec_out1 = recover_img(original, cv2.imread("resize_result/"+dir_name + file_name +"_resize1.png"), x_l, y_b, h1, h2)
                    cv2.imwrite("pixelate_result/" +dir_name + file_name +"_pixelate1.png", rec_out1)

                    rec_out2 = recover_img(original, cv2.imread("resize_result/"+dir_name + file_name +"_resize2.png"), x_l, y_b, h1, h2)
                    cv2.imwrite("pixelate_result/" +dir_name + file_name +"_pixelate2.png", rec_out2)

                    rec_out3 = recover_img(original, cv2.imread("resize_result/"+dir_name + file_name +"_resize3.png"), x_l, y_b, h1, h2)
                    cv2.imwrite("pixelate_result/" +dir_name + file_name +"_pixelate3.png", rec_out3)

                    rec_out4 = recover_img(original, cv2.imread("resize_result/"+dir_name + file_name +"_resize4.png"), x_l, y_b, h1, h2)
                    cv2.imwrite("pixelate_result/" +dir_name + file_name +"_pixelate4.png", rec_out4)

                    del random_crop
                    del x_l 
                    del y_b
                    del tmp  
                    del data 
                    del original 
                    del w  
                    del h 

                    del temp1
                    del temp2
                    del temp3
                    del temp4 

                    del output1
                    del output2
                    del output3
                    del output4

                    del rec_out1
                    del rec_out2
                    del rec_out3
                    del rec_out4

                    del r_w
                    del r_h

                
            del file_list
    
    def randcrop_generate(self, crop_size, count):
        for dir_name in self.dir_list:
            file_list = os.listdir(dir_name)
            print(dir_name)
            for cnt, file_name in enumerate(file_list):
                if "png" in file_name:
                    print(dir_name+file_name)
                    tmp = load_img(dir_name+file_name)
                    data = tmp[0]
                    original = data
                    w = tmp[1]
                    h = tmp[2] 

                    print(cnt, " of ", len(file_list))
                    if(cnt % 10 == 0 and cnt != 0):
                        clear_output()

                    r_w = crop_size
                    r_h = crop_size
                    
                    for i in range(count):
                        random_crop=None
                        while True:
                            random_crop, x_l, y_b = get_random_crop(data, r_h, r_w, h, w)
                            if(np.shape(random_crop)[0] != crop_size or np.shape(random_crop)[1] != crop_size):
                                print("Something was wrong!, try again...")
                                continue
                            else:
                                break
                        
                        if not(os.path.isdir("cropped_result")):
                            os.makedirs(os.path.join("cropped_result")) 
                        if not(os.path.isdir("cropped_result/"+dir_name)):
                            os.makedirs(os.path.join("cropped_result/"+dir_name))

                        cv2.imwrite("cropped_result/" + dir_name + file_name[:-4] +"_"+str(i)+".png", random_crop)

                        del random_crop
                        del x_l 
                        del y_b

                    del tmp  
                    del data 
                    del original 
                    del w  
                    del h

                    del r_w
                    del r_h

            del file_list


new_data = DataGenerator(["./test_set/"])
new_data.pixelate()