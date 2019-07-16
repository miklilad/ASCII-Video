import cv2
from tkinter import *
import numpy as np
import pyscreenshot as ps
import time

text_c = ""
font_size = 64
encoding = 0


def letter_gray_val(font_size, num):
    win = Tk()
    win.title("letter")
    text = Text(win, width=1, height=1, font=("courier", font_size))
    text.pack()
    text.update()
    ar = []
    time.sleep(1)
    xcord = text.winfo_rootx() + 3
    ycord = text.winfo_rooty() + 3
    width = text.winfo_width() - 6
    height = text.winfo_height() - 6

    char = []
    if num == 0:
        for i in range(32, 47):
            char.append(chr(i))
        char.append('▓')
        char.append('▒')
        char.append('░')
        char.append('█')
        char.append('■')
    elif num == 1:
        for i in range(32, 127):
            char.append(chr(i))
    else:
        for i in range(ord('a'), ord('z') + 1):
            char.append(chr(i))
    for x in char:  # 32, 127
        text.delete(0.0, END)
        text.insert(END, x)
        text.update()
        box = (xcord, ycord, xcord + width, ycord + height)
        image = ps.grab(bbox=box).convert('L')
        image = np.array(image)
        mean = image.mean()
        ar.append([mean, x])

    ar.sort()
    win.destroy()
    return ar


def letter_size(font_size):
    win = Tk()
    win.title("letter_size")
    text = Text(win, width=1, height=1, font=("courier", font_size))
    text.pack()
    text.update()
    width = text.winfo_width() - 6
    height = text.winfo_height() - 6
    win.destroy()
    return width, height


def check_input(event):
    global text_c
    text_c = event.char
    return "break"


def find_closest_char(letters, numbers, val):
    lo = 0
    hi = len(numbers) - 1
    while True:
        mid = (lo + hi) // 2
        if mid == lo:
            break
        midval = numbers[mid]
        if midval < val:
            lo = mid
        elif midval > val:
            hi = mid
    if abs(numbers[lo] - val) < abs(numbers[hi] - val):
        return letters[lo]
    else:
        return letters[hi]


def make_interp(left_min, left_max, right_min, right_max):
    # Figure out how 'wide' each range is
    leftSpan = left_max - left_min
    rightSpan = right_max - right_min

    # Compute the scale factor between left and right values
    scaleFactor = float(rightSpan) / float(leftSpan)

    # create interpolation function using pre-calculated scaleFactor
    def interp_fn(value):
        return right_min + (value - left_min) * scaleFactor

    return interp_fn


def load_letters_from_file(filename, num=0):
    try:
        file = open(filename, 'r')
    except:
        save_letters_to_file(filename)
        file = open(filename, 'r')
    try:
        letters = parse_file(file,num)
    except:
        save_letters_to_file(filename)
        file = open(filename, 'r')
        letters = parse_file(file, num)
    file.close()
    return letters


def save_letters_to_file(filename):
    file = open(filename, 'w')
    for i in range(0, 3):
        letters = letter_gray_val(font_size, i)
        if i != 0:
            file.write("::")
        for x in letters:
            file.write(str(x[0]) + "," + x[1] + "\n")
    file.close()


def parse_file(file, num=0):
    text = file.read()
    codings = text.split("::")
    lines = codings[num % len(codings)].split("\n")
    letters = []
    for l in lines:
        if not l:
            continue
        value, char = l.split(",", 1)
        letters.append([float(value), char[0]])
    letters.sort()
    return letters


def ar2np_ar(ar):
    numbers = []
    for x in ar:
        numbers.append(x[0])
    interp = make_interp(numbers[0], numbers[-1], 0, 255)
    numbers = list(map(interp, numbers))
    let = []
    for x in ar:
        let.append(x[1])
    letters = np.array(let)
    numbers = np.array(numbers)
    return letters, numbers


cap = cv2.VideoCapture(0)
ret, img = cap.read()
height, width, channels = img.shape


lwidth, lheight = letter_size(font_size)
win = Tk()
win.title("ASCII")
text = Text(win)
text.bind("<Key>", check_input)
text.pack()
text.config(font=("courier", font_size), width=width // lwidth, height=height // lheight)
label = Label(win, text="q: quit, p: pause, c: change encoding, '+'/'-': inc/dec resolution")
label.pack()
letters = load_letters_from_file("data", encoding)
letters, numbers = ar2np_ar(letters)

last_time = time.time()
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    partitioned = np.zeros((height, width))

    text.delete(0.0, END)
    for y in range(0, height // lheight):
        for x in range(0, width // lwidth):
            value = gray[y * lheight: y * lheight + lheight,
                    x * lwidth: x * lwidth + lwidth].mean()
            partitioned[y * lheight: y * lheight + lheight,
            x * lwidth: x * lwidth + lwidth] = value / 255
            text.insert(END, find_closest_char(letters, numbers, value))
        text.insert(END, '\n')

    #cv2.imshow("gray", gray)
    #cv2.imshow("partitioned", partitioned)

    text.update()
    c = cv2.waitKey(1) & 0xFF
    if c == ord('q') or text_c == 'q':
        break
    elif c == ord('c') or text_c == 'c':
        encoding += 1
        letters = load_letters_from_file("data", encoding)
        letters, numbers = ar2np_ar(letters)
        text_c = ""
    elif c == ord('p') or text_c == 'p':
        text_c = ""
        c = cv2.waitKey(1) & 0xFF

        while not (c == ord('p') or text_c == 'p' or c == ord('q') or text_c == 'q'):
            c = cv2.waitKey(1) & 0xFF
            text.update()
            continue
        if c == ord('q') or text_c == 'q':
            break
        text_c = ""
    elif c == ord('-') or text_c == '-':
        if font_size < width:
            font_size += 2
            lwidth, lheight = letter_size(font_size)
            text.config(font=("courier", font_size), width=width // lwidth, height=height // lheight)
        text_c = ""
    elif c == ord('+') or text_c == '+':
        if font_size > 3:
            font_size -= 2
            lwidth, lheight = letter_size(font_size)
            text.config(font=("courier", font_size), width=width // lwidth, height=height // lheight)
        text_c = ""
    print("FPS: " + str(1 / (time.time() - last_time)))
    last_time = time.time()

cap.release()
cv2.destroyAllWindows()
