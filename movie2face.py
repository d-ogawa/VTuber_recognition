import cv2
import glob
import os

def movie_to_image(video_paths, out_image_path, num_cut=10):
    img_count = 0
    for video_path in video_paths:
        print(video_path)
        capture = cv2.VideoCapture(video_path)
        frame_count = 0
        while(capture.isOpened()):

            ret, frame = capture.read()
            if ret == False:
                break

            if frame_count % num_cut == 0:
                img_file_name = os.path.join(out_image_path, '{:05d}.jpg'.format(img_count))
                cv2.imwrite(img_file_name, frame)
                img_count += 1

            frame_count += 1

        capture.release()


def face_detect(out_face_path, img_list):

    # https://github.com/nagadomi/lbpcascade_animeface
    # wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml

    xml_path = './lbpcascade_animeface.xml'
    classifier = cv2.CascadeClassifier(xml_path)

    img_count = 0
    for img_path in img_list:

        org_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        face_points = classifier.detectMultiScale(gray_img,
                                                  scaleFactor=1.1,
                                                  minNeighbors=2,
                                                  minSize=(30,30))

        for points in face_points:

            x, y, width, height =  points

            dst_img = org_img[y:y+height, x:x+width]

            face_img = cv2.resize(dst_img, (128,128))
            new_img_name = os.path.join(out_face_path, '{:05d}.jpg'.format(img_count))
            cv2.imwrite(new_img_name, face_img)
            img_count += 1

if __name__ == '__main__':

    VTubers = ['KizunaAI', 'MiraiAkari', 'KaguyaLuna', 'Siro', 'NekoMas']
    for VTuber in VTubers:

        print(VTuber)
        video_dir = os.path.join('./video', VTuber)
        video_paths = glob.glob(os.path.join(video_dir, '*.mp4'))

        out_image_path = os.path.join('./image/', VTuber)   # './image/KizunaAI'
        out_face_path = os.path.join('./face/', VTuber)  # './face/KizunaAI'

        print('Movie to image ...')
        movie_to_image(video_paths, out_image_path, num_cut=10)

        images = glob.glob(out_image_path + '/*.jpg')
        print('image num', len(images))

        print('Save %s faces ...' %(VTuber))
        face_detect(out_face_path, images)

        faces = glob.glob(out_face_path + '/*.jpg')
        print('face  num', len(faces))
        print()
