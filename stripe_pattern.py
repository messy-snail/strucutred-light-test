import numpy as np


def split(imgs):
    return list(imgs.transpose(2, 0, 1))

def generate(dsize, option):
    width, height = dsize

    # -2: 0b 문자 날림
    width_num = len(bin(width - 1)) - 2
    height_num = len(bin(height - 1)) - 2

    if option=='b':
        print('바이너리로 생성')
        imgs_code_vertical = 255 * np.fromfunction(lambda y, x, n: (x & (1 << (width_num - 1 - n)) != 0), (height, width, width_num),
                                          dtype=int).astype(np.uint8)
        imgs_code_horizontal = 255 * np.fromfunction(lambda y, x, n: (y & (1 << (height_num - 1 - n)) != 0), (height, width, height_num),
                                          dtype=int).astype(np.uint8)

    elif option == 'g':
        print('그레이로 생성')
        imgs_code_vertical = 255 * np.fromfunction(lambda y, x, n: ((x ^ (x >> 1)) & (1 << (width_num - 1 - n)) != 0),
                                                   (height, width, width_num), dtype=int).astype(np.uint8)

        imgs_code_horizontal = 255 * np.fromfunction(
            lambda y, x, n: ((y ^ (y >> 1)) & (1 << (height_num - 1 - n)) != 0),
            (height, width, height_num), dtype=int).astype(np.uint8)

    img_vertical_list = split(imgs_code_vertical)

    img_final_list = []
    for img in img_vertical_list:
        img_final_list.append(img)
        img = np.where(img==255, 0, 255)
        img_final_list.append(img)

    for img in imgs_code_horizontal:
        img_final_list.append(img)
        img = np.where(img==255, 0, 255)
        img_final_list.append(img)

    return img_final_list
