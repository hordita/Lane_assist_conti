import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')
left_xs = []
left_ys = []
right_xs = []
right_ys = []
left_top_x = 0
left_bottom_x = 0
right_top_x = 0
right_bottom_x = 0

sobel_vertical = np.float32([[-1, -2, -1],
                             [0, 0, 0],
                             [+1, +2, +1]])

sobel_horizontal = np.transpose(sobel_vertical)

while True:

    ret, frame = cam.read()

    if ret is False:
        break
    # rezolutia mea: 3840x2160
    # rezolutie video:1280x720

    # 2
    h, w, _ = frame.shape
    frame = cv2.resize(frame, (int(w / 3), int(h / 3)))
    cadru_initial = frame

    # 3
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4
    h, w = frame.shape
    trapez = np.zeros((h, w), dtype=np.uint8)

    upper_right = (w * 0.55, h * 0.778)
    upper_left = (w * 0.42, h * 0.778)
    lower_left = (w * 0.045, h)
    lower_right = (w * 0.985, h)

    coord = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)
    cv2.fillConvexPoly(trapez, coord, 255)
    road = trapez * frame * trapez

    # 5
    coord = np.float32(coord)
    coord_standard = np.array([(w, 0), (0, 0), (0, h), (w, h)], dtype=np.float32)
    coord_magic = cv2.getPerspectiveTransform(np.float32(coord), coord_standard)
    top_down = cv2.warpPerspective(road, coord_magic, (w, h))

    # 6
    blur_frame = cv2.blur(top_down, ksize=(11, 11))

    # 7
    sobel1 = np.float32(blur_frame)
    sobel2 = sobel1
    sobel1 = cv2.filter2D(sobel1, -1, sobel_horizontal)
    sobel2 = cv2.filter2D(sobel2, -1, sobel_vertical)
    # sobel1 = cv2.convertScaleAbs(sobel1)
    sobel = ((sobel1 ** 2) + (sobel2 ** 2)) ** 1 / 2
    sobel = cv2.convertScaleAbs(sobel)

    # 8
    th, binarize = cv2.threshold(sobel, 240, 255, cv2.THRESH_OTSU)

    # 9
    c_binarize = binarize.copy()
    c_binarize[:, int(w - (w / 10)):w] = 1
    c_binarize[:, 0:25] = 1
    c_binarize[h - 10:h, :] = 1

    c_binarize_left = c_binarize[:, 0:int(w / 2)]
    arr_left_white = np.argwhere(c_binarize_left == 255)
    c_binarize_right = c_binarize[:, int(w / 2) + 1:w]
    arr_right_white = np.argwhere(c_binarize_right == 255)

    left_xs = arr_left_white[:, 1]
    left_ys = arr_left_white[:, 0]
    right_xs = arr_right_white[:, 1] + int(w / 2)
    right_ys = arr_right_white[:, 0]

    # 10
    left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    left_top_y = 0
    if 35 < ((left_top_y - left_line[0]) / left_line[1]) < 200:
        left_top_x = (left_top_y - left_line[0]) / left_line[1]

    left_bottom_y = h
    if 60 < ((left_bottom_y - left_line[0]) / left_line[1]) < 120:
        left_bottom_x = (left_bottom_y - left_line[0]) / left_line[1]
    right_top_y = 0
    right_top_x = (right_top_y - right_line[0]) / right_line[1]

    right_bottom_y = h
    if 400 > ((right_bottom_y - right_line[0]) / right_line[1]) > 250:
        right_bottom_x = (right_bottom_y - right_line[0]) / right_line[1]

    left_top = int(left_top_x), int(left_top_y)
    left_bottom = int(left_bottom_x), int(left_bottom_y)

    right_top = int(right_top_x), int(right_top_y)
    right_bottom = int(right_bottom_x), int(right_bottom_y)
    
    cv2.line(c_binarize, left_top, left_bottom,(200, 0, 0) , 7)
    #arr_left_white[:, [0, 1]] = arr_left_white[:, [1, 0]]
    #cv2.polylines(c_binarize, [arr_left_white], isClosed=True, color=(200, 0, 0),thickness=5)
    cv2.line(c_binarize, right_top, right_bottom, (100, 0, 0), 7)

    # 11
    color_line_left = np.zeros((h, w), dtype=np.uint8)
    #cv2.polylines(color_line_left, [arr_left_white], isClosed=True, color=(255, 0, 0), thickness=10)
    cv2.line(color_line_left, left_top, left_bottom, (255, 0, 0), 13)
    coord_magic = cv2.getPerspectiveTransform(coord_standard, coord)
    color_line_left = cv2.warpPerspective(color_line_left, coord_magic, (w, h))
    white_coord_line_trapez = np.argwhere(color_line_left == 255)
    for i in white_coord_line_trapez:
        cadru_initial[i[0], i[1]] = (50, 50, 250)

    color_line_right = np.zeros((h, w), dtype=np.uint8)
    cv2.line(color_line_right, right_top, right_bottom, (255, 0, 0), 13)
    coord_magic = cv2.getPerspectiveTransform(coord_standard, coord)
    color_line_right = cv2.warpPerspective(color_line_right, coord_magic, (w, h))
    white_coord_line_trapez = np.argwhere(color_line_right == 255)
    for i in white_coord_line_trapez:
        cadru_initial[i[0], i[1]] = (250, 0, 250)

    # Afisari
    cv2.imshow('Small', frame)
    cv2.imshow('Road', road)
    cv2.imshow('Blur', blur_frame)
    cv2.imshow('C_Binarize', c_binarize)
    cv2.imshow('Final', cadru_initial)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
