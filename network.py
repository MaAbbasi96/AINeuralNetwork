import os
from PIL import Image

alphabet_to_int = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9
}

def get_sample_data(folder_name):
    X = []
    Y = []
    path = os.getcwd() + '/' + folder_name
    for folder_name in os.listdir(path):
        folder_path = path + '/' + folder_name
        for index, image_name in enumerate(os.listdir(folder_path)):
            if index % 1801 == 0:
                image_path = folder_path + '/' + image_name
                image = Image.open(image_path)
                image = image.convert('1')
                data = [0 for i in range(10)]
                data[alphabet_to_int[folder_name]] = 1
                Y.append(data)
                pixels = image.load()
                data = []
                for i in range(28):
                    for j in range(28):
                        data.append(pixels[i,j] if pixels[i,j] == 0 else 1)
                X.append(data)
    return X, Y

def main():
    input_data, result = get_sample_data('notMNIST_small')
    print input_data[0], result[0]

if __name__ == '__main__':
    main()