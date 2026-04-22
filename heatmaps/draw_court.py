import cv2
import numpy as np


def draw_court(court_x, court_y, scale=250):
    COURT_W = 9
    COURT_H = 18
    FREE = 3.5
    LINE_W = 5

    img_w = int((COURT_W + 2 * FREE) * scale)
    img_h = int((COURT_H + 2 * FREE) * scale)

    ORANGE = (60, 140, 220)
    GREEN = (80, 130, 80)
    WHITE = (255, 255, 255)

    court_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    court_img[:] = GREEN

    offset = int(FREE * scale)

    # main court
    cv2.rectangle(
        court_img,
        (offset, offset),
        (offset + int(COURT_W * scale), offset + int(COURT_H * scale)),
        ORANGE,
        -1
    )

    cv2.rectangle(
        court_img,
        (offset, offset),
        (offset + int(COURT_W * scale), offset + int(COURT_H * scale)),
        WHITE,
        LINE_W,
        lineType=cv2.LINE_AA
    )

    # net
    net_y = offset + int((COURT_H / 2) * scale)
    cv2.line(
        court_img,
        (offset, net_y),
        (offset + int(COURT_W * scale), net_y),
        WHITE,
        LINE_W,
        lineType=cv2.LINE_AA
    )

    # 3m lines
    attack_offset = int(3 * scale)
    y_top = net_y - attack_offset
    y_bot = net_y + attack_offset

    for y in [y_top, y_bot]:
        cv2.line(
            court_img,
            (offset, y),
            (offset + int(COURT_W * scale), y),
            WHITE,
            LINE_W,
            lineType=cv2.LINE_AA
        )

    # dashed extensions
    def draw_fixed_dashes(y, x_start, direction):
        DASH = int(0.2 * scale)
        GAP = int(0.2 * scale)
        COUNT = 5

        x = x_start
        for _ in range(COUNT):
            x_end = x + direction * DASH
            cv2.line(
                court_img,
                (int(x), int(y)),
                (int(x_end), int(y)),
                WHITE,
                LINE_W,
                lineType=cv2.LINE_AA
            )
            x = x_end + direction * GAP

    left_edge = offset
    right_edge = offset + int(COURT_W * scale)

    for y in [y_top, y_bot]:
        draw_fixed_dashes(y, left_edge, -1)
        draw_fixed_dashes(y, right_edge, 1)

    # landing point
    px = offset + int(court_x * scale)
    py = offset + int((COURT_H - court_y) * scale)

    
    return court_img
