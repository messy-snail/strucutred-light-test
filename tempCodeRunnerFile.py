def generate(dsize):
    width, height = dsize
    num = len(bin(width - 1)) - 2

    # y,x,n을 입력으로,
    # x &
    imgs_code = 255 * np.fromfunction(lambda y, x, n: (x & (1 << (num - 1 - n)) != 0), (height, width, num),
                                      dtype=int).astype(np.uint8)

    imlist = split(imgs_code)
    return imlist