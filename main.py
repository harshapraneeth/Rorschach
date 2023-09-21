import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def show_image(image: np.ndarray, grid: bool = True) -> None:

    plt.imshow(image, cmap="gray")
    
    if grid:

        plt.xticks([i-0.5 for i in range(image.shape[0])], labels=[])
        plt.yticks([i-0.5 for i in range(image.shape[0])], labels=[])
        plt.grid(visible=True, axis="both", color="red")

    plt.show()


def to_binary(x: int, n: int = 8) -> str:

    return (
        "0"*n + 
        bin(x).replace('b', '0')
    )[-n:]


def to_decimal(x: str) -> int:

    return int(x, base=2)


def print_output(output: str) -> None:

    m, n = 5, np.ceil(len(output)/8)

    i = 0
    while i < n:

        j = 0
        while j<m and i<n:

            print(output[i*8 : (i+1)*8], end=" ")
            i, j = i+1, j+1

        print()


def polynomial_multiplication(X: list, Y: list) -> list:

    nx, ny = len(X), len(Y)
    n = nx + ny - 1
    res = [0]*n

    for i in range(n):
        for ix in range(i+1):
            iy = i-ix

            if ix<nx and iy<ny and X[ix] and Y[iy]:
                res[i] ^= ANTILOG[(LOG[X[ix]] + LOG[Y[iy]]) % 255]

    return res


def polynomial_division(X: list, Y: list) -> list:

    nx = len(X)
    n = nx - len(Y) + 1
    res = X[:]

    for i in range(n):

        nr = len(res)
        if nr <=0 : break

        if res[0]:

            factor = ANTILOG[
                (LOG[res[0]] + LOG[Y[0]] * 254) % 255
            ]

            subtr = (
                polynomial_multiplication(Y, [factor]) + 
                [0]*nr
            )

            res = list(
                map(
                    lambda x: x[0] ^ x[1],
                    zip(res, subtr)
                )
            )

        res = res[1:]

    return res


if __name__ == "__main__":

    '''
    INPUT = input(
        "Enter input: "
    )
    '''

    INPUT = "https://www.qrcode.com/"
    output = ""

    # ========================================================

    MODE = (
        "NUMERIC" if re.search("^\d*$", INPUT) is not None
        else "ALPHANUMERIC" if re.search("^[\dA-Z $%*+\-./:]*$", INPUT) is not None
        else "BYTE" if re.search("^[\x00-\xff]*$", INPUT) is not None
        else None
    )
    
    match MODE:

        case "NUMERIC": output += "0001"
        case "ALPHANUMERIC": output += "0010"
        case "BYTE": output += "0100"
        case _: 

            print("Input mode is not supported.")
            raise ValueError
    
    print("Selected mode: %s" % MODE)

    # ========================================================

    INPUT_LEN = len(INPUT)

    capacities = pd.read_csv(
        "tables/capacities.csv",
        sep = ' '
    )

    temp = capacities[MODE] >= INPUT_LEN
    temp = capacities[temp].sort_values(by=MODE)
    
    # VERSION = temp["VERSION"].iloc[0]
    # ECLEVEL = temp["ECLEVEL"].iloc[0]

    VERSION = 2
    ECLEVEL = "M"

    print("Selecte version: %s" % VERSION)
    print("Selected ec level: %s" % ECLEVEL)

    # ========================================================

    length_indicators = pd.read_csv(
        "tables/length_indicators.csv",
        sep = ' '
    )

    temp = length_indicators["VERSION"] == VERSION
    temp = length_indicators[temp][MODE].iloc[0]

    LENGTH_INDICATOR = temp
    output += to_binary(INPUT_LEN, LENGTH_INDICATOR)
    
    # ========================================================

    match MODE:

        case "NUMERIC":

            print("Input mode is not supported.")
            raise ValueError

        case "ALPHANUMERIC":

            encoding = pd.read_csv(
                "tables/alphanumeric_encoding.csv",
                sep = ','
            )

            for i in range(0, INPUT_LEN, 2):

                x = INPUT[i:i+2]

                temp = encoding["CHARACTER"] == x[0]
                temp = encoding[temp]["VALUE"]
                a = temp.iloc[0]

                if len(x)==2:

                    temp = encoding["CHARACTER"] == x[1]
                    temp = encoding[temp]["VALUE"]
                    b = temp.iloc[0]

                    output += to_binary(a*45 + b, 11)

                else:

                    output += to_binary(a, 6)

        case "BYTE":
            
            for x in INPUT:

                output += to_binary(
                    int(
                        x.encode(encoding="iso-8859-1").hex(),
                        base = 16
                    )
                )

        case _: 

            print("Input mode is not supported.")
            raise ValueError
        
    # ========================================================

    ecc_info = pd.read_csv(
        "tables/ecc_info.csv",
        sep = ' '
    )

    temp = ecc_info[ecc_info["VERSION"] == VERSION]
    temp = temp[temp["ECLEVEL"] == ECLEVEL]

    NUMBITS = temp["DATACODEWORDS"].iloc[0] * 8
    
    output += "0"*4
    output = output[:NUMBITS]

    temp = len(output) % 8
    if temp > 0: 

        output += "0" * (8-temp)

    temp = len(output)
    while temp < NUMBITS:

        output += "1110110000010001"
        temp += 16

    output = output[:NUMBITS]

    # ========================================================

    ecc_info = pd.read_csv(
        "tables/ecc_info.csv",
        sep = ' '
    )

    temp = ecc_info[ecc_info["VERSION"] == VERSION]
    temp = temp[temp["ECLEVEL"] == ECLEVEL]

    blocks = []
    for group in range(2):
        
        num_blocks = int(
            np.nan_to_num(
                temp["GROUP%dBLOCKS" % (group+1)].iloc[0]
            )
        )

        num_datacodewords = int(
            np.nan_to_num(
                temp["GROUP%dDATACODEWORDS" % (group+1)].iloc[0]
            )
        )

        for i in range(num_blocks):

            block = {
                "DATACODEWORDS": [],
                "ECCODEWORDS": []
            }

            for j in range(num_datacodewords):

                block["DATACODEWORDS"].append(output[:8])
                output = output[8:]

            blocks.append(block)
    
    LOG = [0] * 256
    ANTILOG = [0] * 256

    value = 1
    for exponent in range(1, 256):

        value = (
            (value << 1) ^ 285 if value>127 
            else value << 1
        )

        LOG[value] = exponent % 255
        ANTILOG[exponent % 255] = value
    
    # ========================================================

    ecc_info = pd.read_csv(
        "tables/ecc_info.csv",
        sep = ' '
    )

    temp = ecc_info[ecc_info["VERSION"] == VERSION]
    temp = temp[temp["ECLEVEL"] == ECLEVEL]

    NUMECC = temp["ECCODEWORDS"].iloc[0]

    GENPOLY = [1]
    for i in range(NUMECC):

        GENPOLY = polynomial_multiplication(
            GENPOLY, 
            [1, ANTILOG[i]]
        )

    for block in blocks:

        MSGPOLY = [
            to_decimal(x) 
            for x in block["DATACODEWORDS"]
        ]

        MSGPOLY += [0] * NUMECC

        block["ECCODEWORDS"] = [
            to_binary(x)
            for x in polynomial_division(MSGPOLY, GENPOLY)
        ]

    # ========================================================

    output = ""

    N = [len(block["DATACODEWORDS"]) for block in blocks]
    n = max(N)

    for i in range(n):
        for j, block in enumerate(blocks):

            if i>=N[j]: continue
            output += block["DATACODEWORDS"][i]

    N = [len(block["ECCODEWORDS"]) for block in blocks]
    n = max(N)

    for i in range(n):
        for j, block in enumerate(blocks):

            if i>=N[j]: continue
            output += block["ECCODEWORDS"][i]

    # ========================================================

    remainders = pd.read_csv(
        "tables/remainders.csv",
        sep = ' '
    )

    temp = remainders[remainders["VERSION"] == VERSION]
    REMAINDER = temp["REMAINDER"].iloc[0]

    output += "0" * REMAINDER

    # ========================================================

    SIZE = VERSION*4 + 17

    image = np.ones(shape = (SIZE, SIZE), dtype=bool)
    reserved = np.zeros(shape = image.shape, dtype=bool)

    finder_pattern = np.ones(shape = (9, 9), dtype=bool)
    finder_pattern[1:8, 1:8] = 0
    finder_pattern[2:7, 2:7] = 1
    finder_pattern[3:6, 3:6] = 0

    image[:8, :8] = finder_pattern[1:, 1:]
    image[:8, -8:] = finder_pattern[1:, :-1]
    image[-8:, :8] = finder_pattern[:-1, 1:]

    reserved[:8, :8] = 1
    reserved[:8, -8:] = 1
    reserved[-8:, :8] = 1

    image[6, 8:-8] = np.arange(SIZE-16) % 2
    image[8:-8, 6] = np.arange(SIZE-16) % 2

    reserved[6, 8:-8] = 1
    reserved[8:-8, 6] = 1

    image[-8, 8] = 0
    reserved[-8, 8] = 1

    if VERSION>1:

        alignment_pattern = np.zeros(shape = (5, 5), dtype=bool)
        alignment_pattern[1:4, 1:4] = 1
        alignment_pattern[2, 2] = 0

        alignments = pd.read_csv(
            "tables/alignments.csv",
            sep = ' '
        )

        temp = alignments["VERSION"] == VERSION
        temp = alignments[temp]["POSITIONS"].iloc[0]

        POSITIONS = list(
            map(
                lambda x: int(x) - 2, 
                temp.split(',')
            )
        )

        for row in POSITIONS:
            for col in POSITIONS:

                if row<8 and col<8: continue
                if row<8 and col+4>=SIZE-8: continue
                if row+4>=SIZE-8 and col<8: continue

                image[row:row+5, col:col+5] = alignment_pattern
                reserved[row:row+5, col:col+5] = 1

    
    reserved[:9, :9] = 1
    reserved[8, -8:] = 1
    reserved[-8:, 8] = 1

    if VERSION >= 7:

        reserved[:6, -11:] = 1
        reserved[-11:, :6] = 1

    # ========================================================

    i, j = 0, 0
    row, col = SIZE-1, SIZE-1
    drow = -1

    while col>=0:

        if not reserved[row, col]: 

            image[row, col] = output[j]=='0'
            j += 1

        if i%2:

            row += drow
            if row<0 or row>=SIZE:

                drow = -drow
                row += drow
                col -= 2 if col==7 else 1

            else: col += 1

        else: col -= 1

        i += 1

    show_image(image)

    # ========================================================

    MASKS = [
        lambda row, col: (row + col) % 2 == 0,
        lambda row, _: row % 2 == 0,
        lambda _, col: col % 3 == 0,
        lambda row, col: (row + col) % 3 == 0,
        lambda row, col: (np.floor(row/2) + np.floor(col/3)) % 2 == 0,
        lambda row, col: ((row * col)%2) + ((row * col)%3) == 0,
        lambda row, col: (((row * col)%2) + ((row * col)%3)) % 2 == 0,
        lambda row, col: (((row + col)%2) + ((row * col)%3)) % 2 == 0
    ]

    min_score = np.inf
    opt_mask = -1

    for i, mask in enumerate( MASKS):

        flags = np.array(
            [
                [
                    0 if reserved[row, col] else mask(row, col)
                    for col in range(SIZE)
                ]
                for row in range(SIZE)
            ], 
            dtype=bool
        )

        masked_image = np.logical_xor(image, flags)
        score = 0

        if score < min_score:

            min_score = score
            opt_mask = i

    # MASK = opt_mask
    MASK = 2

    flags = np.array([
        [
            0 if reserved[row, col] else  MASKS[MASK](row, col)
            for col in range(SIZE)
        ]
        for row in range(SIZE)
    ])

    image = np.logical_xor(image, flags)

    show_image(image)

    # ========================================================

    format_info = pd.read_csv(
        "tables/format_info.csv",
        sep = ' ',
        dtype=str
    )

    temp = format_info[format_info["ECLEVEL"] == ECLEVEL]
    temp = temp[temp["MASK"] == str(MASK)]

    FORMAT = np.array(
        list(
            map(
                lambda x: x=='0',
                str(temp["BITS"].iloc[0])
            )
        )
    )

    image[8, 0:6] = FORMAT[0:6]
    image[0:6, 8] = FORMAT[14:8:-1]

    image[8, 7] = FORMAT[6]
    image[8, 8] = FORMAT[7]
    image[7, 8] = FORMAT[8]

    image[SIZE-7:, 8] = FORMAT[6::-1]
    image[8, SIZE-8:] = FORMAT[7:]
    
    show_image(image, grid=False)





