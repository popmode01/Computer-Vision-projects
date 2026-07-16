import cv2
import numpy as np


brush_color = (0, 255, 0)
brush_size = 8
last_point = None
drawing_enabled = False
toolbar_height = 90
frame_width = 1280
frame_height = 720

draw_colors = [
    ("GREEN", (0, 255, 0)),
    ("BLUE", (255, 0, 0)),
    ("RED", (0, 0, 255)),
    ("YELLOW", (0, 255, 255)),
    ("WHITE", (255, 255, 255)),
]


def find_blue_marker(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([95, 120, 100])
    upper_blue = np.array([125, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < 300 or area > 6000:
        return None, mask

    perimeter = cv2.arcLength(largest_contour, True)

    if perimeter == 0:
        return None, mask

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    if circularity < 0.35:
        return None, mask

    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    if radius < 6 or radius > 45:
        return None, mask

    return (int(x), int(y)), mask


def draw_toolbar(frame):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], toolbar_height), (35, 35, 35), cv2.FILLED)

    color_boxes = []
    box_size = 65
    gap = 15
    x = 15

    for name, color in draw_colors:
        x1 = x
        y1 = 12
        x2 = x + box_size
        y2 = y1 + box_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)

        border_color = (255, 255, 255) if color == brush_color else (120, 120, 120)
        border_size = 4 if color == brush_color else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_size)

        cv2.putText(frame, name[0], (x1 + 19, y1 + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        color_boxes.append((x1, y1, x2, y2, color))
        x += box_size + gap

    cv2.putText(frame, "Use BLUE marker | SPACE draw | C clear | S save | Q quit", (x + 10, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return color_boxes


def select_color(marker_point, color_boxes):
    global brush_color, last_point

    if marker_point is None:
        return False

    x, y = marker_point

    for x1, y1, x2, y2, color in color_boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            brush_color = color
            last_point = None
            return True

    return y < toolbar_height


def main():
    global last_point, drawing_enabled

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        return

    # Request a larger webcam frame for a bigger drawing window.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    canvas = None
    window_name = "Air Drawing"

    print("Important:")
    print("  Put blue tape, blue pen cap, or any blue object on your index finger.")
    print("Controls:")
    print("  Move blue marker to a color box to select color")
    print("  Press space to start/stop drawing")
    print("  Press c to clear")
    print("  Press s to save")
    print("  Press q to quit")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        marker_point, mask = find_blue_marker(frame)
        color_boxes = draw_toolbar(frame)
        selecting_toolbar = select_color(marker_point, color_boxes)

        if marker_point is not None:
            cv2.circle(frame, marker_point, 12, (255, 255, 255), cv2.FILLED)
            cv2.circle(frame, marker_point, 8, brush_color, cv2.FILLED)

            if drawing_enabled and not selecting_toolbar:
                if last_point is not None:
                    cv2.line(canvas, last_point, marker_point, brush_color, brush_size)
                last_point = marker_point
            else:
                last_point = None
        else:
            last_point = None

        canvas[:toolbar_height, :] = 0
        output = cv2.addWeighted(frame, 1, canvas, 1, 0)

        status = "DRAWING ON" if drawing_enabled else "DRAWING OFF - press SPACE"
        cv2.putText(output, status, (20, toolbar_height + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, brush_color, 2)

        cv2.imshow(window_name, output)
        cv2.imshow("Blue Marker Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            drawing_enabled = not drawing_enabled
            last_point = None

        elif key == ord("c"):
            canvas[:] = 0

        elif key == ord("s"):
            cv2.imwrite("air_drawing.png", output)
            print("Saved as air_drawing.png")

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


main()

