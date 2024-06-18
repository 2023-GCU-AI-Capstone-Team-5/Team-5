import cv2
import dlib
import numpy as np

#image -> highlighted_image
def highlight_key_parts(highlighted_image, landmarks,scale_factor=1.0):
    highlighted_points = []
    for landmark in landmarks:
        # 각 랜드마크 영역의 경계 상자 계산
        (x, y, w, h) = cv2.boundingRect(landmark)
        # 랜드마크 영역의 가로 세로 크기 계산
        width = w
        height = h
        # 각 랜드마크 영역의 크기를 조정
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        # scale_factor가 1인 경우에만 조정
        if scale_factor == 1:
            new_width = width
            new_height = height

        # 랜드마크 영역 중심 계산
        center_x = x + width // 2
        center_y = y + height // 2
        # 새로운 크기를 이용하여 좌우 양옆 크기가 같은 bounding box 계산
        x = max(0, center_x - new_width // 2)
        y = max(0, center_y - new_height // 2)
        # 랜드마크 영역 내의 픽셀을 붉은색으로 변경
        roi = highlighted_image[y:y+new_height, x:x+new_width]
        roi[:] = [0, 0, 255] # 해당 랜드마크 영역의 픽셀을 붉은색(RGB)으로 변경
        # 바운딩 박스의 좌표를 출력
        print(f"Bounding box coordinates - x: {x}, y: {y}, width: {w}, height: {h}")
        highlighted_points.append([x,y,w,h])
        # 얼굴 랜드마크를 파란색으로 가시화
        # shpae -> landmark
        for (x, y) in landmark:
            cv2.circle(highlighted_image, (x, y), 2, (255, 0, 0), -1)
    return highlighted_image, highlighted_points


# 얼굴 랜드마크 좌표를 NumPy 배열로 변환하는 함수
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def Point_expansion(points, height, width):
    position = []
    for i in range(height):
        position.append([0] * width)
    for x, y, w, h in points:
        print(f"x: {x}, y: {y}, w: {w}, h: {h}")
        for j in range(y, y + h):
            for i in range(x, x + w):
                position[j][i] = 1

    return position