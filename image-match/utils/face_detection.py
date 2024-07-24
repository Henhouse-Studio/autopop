import cv2
import face_recognition


# To find the faces in the images
def get_face_encoding(image_path: str):

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return None, []

    encodings = face_recognition.face_encodings(image, face_locations)

    return encodings, face_locations


def on_close(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.destroyAllWindows()


def display_face_locations(image_path: str, face_locations: list):

    image = cv2.imread(image_path)
    for top, right, bottom, left in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    window_name = f"Faces found in {image_path}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_close)

    cv2.imshow(window_name, image)

    while True:
        key = cv2.waitKey(100)
        if (
            key == 27
            or key == ord("q")
            or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
        ):
            break

    cv2.destroyAllWindows()


# For Testing
if __name__ == "__main__":

    image_path = "images/rick_base.jpg"
    encodings, face_locations = get_face_encoding(image_path)

    if encodings is not None:
        print(f"{len(face_locations)} faces detected.")
        display_face_locations(image_path, face_locations)

    else:
        print("No faces found.")
