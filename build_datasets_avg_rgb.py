from PIL import Image, ImageEnhance
import numpy as np
import os
import pandas as pd
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='AVG RGB Dataset Builder')
    parser.add_argument('--SOURCE_PATH', type=str, required=True, help='Path to source images folder')
    return parser.parse_args()

valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

def build_dataframe_average_rgb():
    args = get_args()

    df=pd.DataFrame(columns=['filename','avg_r','avg_g','avg_b'])
    
    source = args.SOURCE_PATH
    _, _, filenames = next(os.walk(source))
    
    length=len(filenames)
    index=0
    
    print('')
    for filename in filenames:
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            print(f"Skipping non-image file: {filename}")
            continue
        try:
            img = Image.open(source+filename)
            print(f"Image Format: {img.format}")
            print(f"Image Mode: {img.mode}")  # e.g., RGB, L, CMYK
            print(f"Image Size: {img.size}")  # Width x Height

            print(f"Opening image: {filename}, Dimensions: {img.size}")
            img_array = np.array(img)
            print(f"Image Array Shape: {img_array.shape}")
            print(f"Image Array (Sample Data):\n{img_array[:5, :5]}")
            # Check if the image is loaded and has 3 dimensions (height, width, channels)
            if img_array.ndim != 3 or img_array.shape[2] != 3:
                print(f"Skipping {filename}: Not a valid RGB image or corrupted.")
                continue
            #Get the average value of Red, Green, and Blue
            #Original Image
            df = pd.concat([df, pd.DataFrame({'filename': [filename],
                                              'avg_r': [np.mean(img_array[:, :, 0])],
                                              'avg_g': [np.mean(img_array[:, :, 1])],
                                              'avg_b': [np.mean(img_array[:, :, 2])],
                                              })], ignore_index=True)
            # RGB -> BGR Image
            bgr_img_array = img_array[:, :, ::-1]
            df = pd.concat([df, pd.DataFrame({'filename': ['bgr_' + filename],
                                              'avg_r': [np.mean(bgr_img_array[:, :, 0])],
                                              'avg_g': [np.mean(bgr_img_array[:, :, 1])],
                                              'avg_b': [np.mean(bgr_img_array[:, :, 2])],
                                              })], ignore_index=True)
            bgr_img = Image.fromarray(bgr_img_array)
            bgr_img.save(source + 'bgr_' + filename)

            # Enhanced Image
            img_enh = ImageEnhance.Contrast(img)
            img_enh = img_enh.enhance(1.8)
            img_enh_array = np.array(img_enh)
            df = pd.concat([df, pd.DataFrame({'filename': ['enh_' + filename],
                                              'avg_r': [np.mean(img_enh_array[:, :, 0])],
                                              'avg_g': [np.mean(img_enh_array[:, :, 1])],
                                              'avg_b': [np.mean(img_enh_array[:, :, 2])],
                                              })], ignore_index=True)
            img_enh.save(source + 'enh_' + filename)

            # Grayscale Image
            grey_img = img.convert('L')
            grey_img_array = np.array(grey_img)
            df = pd.concat([df, pd.DataFrame({'filename': ['gray_' + filename],
                                              'avg_r': [np.mean(grey_img_array)],
                                              'avg_g': [np.mean(grey_img_array)],
                                              'avg_b': [np.mean(grey_img_array)],
                                              })], ignore_index=True)
            grey_img.save(source + 'gray_' + filename)

            index+=1
            print(('%.4f percents done \r')%(index*100/length),end='')
        except Exception as e:
            print(f"\n Image Error with {filename}: {e}")
            index += 1
    print('')
    df.to_csv('Avg_RGB_dataset.csv',index=False)

if __name__=='__main__':
    build_dataframe_average_rgb()