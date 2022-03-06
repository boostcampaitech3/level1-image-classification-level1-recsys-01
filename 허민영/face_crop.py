# 나머지 코드는 '/opt/ml/main.py' 여기에 옮겨놈
# import zipfile
#
# try:
#     with zipfile.ZipFile("/opt/ml/P_level1/crop_images.zip") as zf:
#         zf.extractall()
#         print("uncompress success")
#
# except:
#     print("uncompress fail")
################################# 진짜 최종 코드
# apt-get install libgl1-mesa-glx
# pip install mtcnn
# pip install tensorflow
# pip uninstall opencv-python-headless
# pip install opencv-python-headless==4.1.2.30

import os
import cv2
import mtcnn
import matplotlib.pyplot as plt

data_dir = '/opt/ml/P_level1/images'
profiles = os.listdir('/opt/ml/P_level1/images')
for profile in profiles:
    if profile.startswith("."):
        continue

    img_folder = os.path.join(data_dir, profile)  # /opt/ml/input/data/train/images/  +  000001_female_Asian_45

    a = os.listdir('/opt/ml/P_level1/images(crop)/'+profile)
    if len(a) == 7 or len(a) == 8: # 아래의 imwrite 오류 때문에 코드 다시 실행시킬 때 7개의 사진을 이미 다 crop한 폴더는 pass
        print('pass')              # -> 7개만 있는 줄 알았는데 checkpoint라는 빈폴더가 있는 폴더도 있었음
        continue                   #   (왜있는지는 모르겠으나 8개로 간주해서 한번 더 저장시키기 때문에 시간이 낭비되는 것 같음. 따라서 8개가 있는 경우도 pass)

    # 다 수정함
    # 아래는 imwrite 오류 생기는 폴더들.. (나중에 수작업으로 처리할 것)  => 총 37개  => 이미지들 하나 하나 보니까 이 사진들은 crop안하는 게 나을거같음
    if img_folder == '/opt/ml/P_level1/images/001762_male_Asian_40' or \
            img_folder == '/opt/ml/P_level1/images/005453_female_Asian_60' or \
            img_folder == '/opt/ml/P_level1/images/001355_female_Asian_22' or \
            img_folder == '/opt/ml/P_level1/images/005457_female_Asian_36' or \
            img_folder == '/opt/ml/P_level1/images/003440_male_Asian_60' or \
            img_folder == '/opt/ml/P_level1/images/005545_male_Asian_53' or \
            img_folder == '/opt/ml/P_level1/images/001077_male_Asian_19' or \
            img_folder == '/opt/ml/P_level1/images/005451_male_Asian_32' or \
            img_folder == '/opt/ml/P_level1/images/001987_female_Asian_48' or \
            img_folder == '/opt/ml/P_level1/images/001766_female_Asian_53' or \
            img_folder == '/opt/ml/P_level1/images/005430_male_Asian_52' or \
            img_folder == '/opt/ml/P_level1/images/006927_male_Asian_19' or \
            img_folder == '/opt/ml/P_level1/images/005418_male_Asian_60' or \
            img_folder == '/opt/ml/P_level1/images/005527_female_Asian_40' or \
            img_folder == '/opt/ml/P_level1/images/005473_female_Asian_51' or \
            img_folder == '/opt/ml/P_level1/images/005475_female_Asian_44' or \
            img_folder == '/opt/ml/P_level1/images/005232_male_Asian_18' or \
            img_folder == '/opt/ml/P_level1/images/001059_female_Asian_25' or \
            img_folder == '/opt/ml/P_level1/images/005439_female_Asian_58' or \
            img_folder == '/opt/ml/P_level1/images/005450_female_Asian_48' or \
            img_folder == '/opt/ml/P_level1/images/005525_female_Asian_39' or \
            img_folder == '/opt/ml/P_level1/images/005407_female_Asian_26' or \
            img_folder == '/opt/ml/P_level1/images/005521_female_Asian_50' or  \
            img_folder == '/opt/ml/P_level1/images/005526_male_Asian_23' or \
            img_folder == '/opt/ml/P_level1/images/005458_male_Asian_52' or \
            img_folder == '/opt/ml/P_level1/images/005471_female_Asian_38' or \
            img_folder == '/opt/ml/P_level1/images/005408_male_Asian_28' or \
            img_folder == '/opt/ml/P_level1/images/001769_male_Asian_51' or \
            img_folder == '/opt/ml/P_level1/images/005459_male_Asian_60' or \
            img_folder == '/opt/ml/P_level1/images/006422_female_Asian_20' or \
            img_folder == '/opt/ml/P_level1/images/001359-1_female_Asian_24' or \
            img_folder == '/opt/ml/P_level1/images/005429_female_Asian_42' or \
            img_folder == '/opt/ml/P_level1/images/005499_male_Asian_55' or \
            img_folder == '/opt/ml/P_level1/images/005516_female_Asian_52' or \
            img_folder == '/opt/ml/P_level1/images/006213_male_Asian_19' or \
            img_folder == '/opt/ml/P_level1/images/005423_male_Asian_38' or \
            img_folder == '/opt/ml/P_level1/images/005405_female_Asian_20' or \
            img_folder == '/opt/ml/P_level1/images/003531_male_Asian_55':
        print('imwrite 오류 생기는 폴더')
        continue

    for file_name in os.listdir(img_folder):
        _file_name, ext = os.path.splitext(file_name)
        if _file_name.startswith("."):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시
            continue

        img_path = os.path.join(data_dir, profile, file_name)  # /opt/ml/input/data/train/images/  +  000001_female_Asian_45/  +  mask1.jpg

        ###----- 파일로부터 이미지 가져오기 -----###
        filename = img_path
        pixels = plt.imread(filename)
        img = cv2.imread(filename)

        ###----- 이미지에서 얼굴 box 검출 -----###
        detector = mtcnn.MTCNN()
        faces = detector.detect_faces(pixels)
        #print(faces)

        if len(faces)==0: # 바운딩 박스가 하나도 검출되지 않았을 경우를 대비해서 임의로 바운딩 박스를 만들어 줌 (아마 이정도 크기의 x,y,width,heigth면 박스 안에 얼굴 다 들어올 듯??)
            print("들어옴!!") # 참고로 7개의 이미지 파일 모두가 box처리 되지 않는 경우가 있어서 이에 해당되는 폴더들의 사진을 다 확인해보니 사람 얼굴에 그림자 진 사진이 대부분이였음. 또는 안경에 그림자 진 사람의 사진
            faces = [{'box': [0, 0, 384, 512],
                      'confidence': 0.9731148481369019,
                      'keypoints': {'left_eye': (142, 207),
                                    'right_eye': (232, 205),
                                    'nose': (178, 249),
                                    'mouth_left': (137, 297),
                                    'mouth_right': (215, 298)}}] # {'box': [86, 103, 207, 258],

        # cnt = 1
        # if len(faces) >= 1:
        #     for face in faces:
        #         print('검출된 바운딩 박스 %d개: ' %cnt, face)
        #         cnt += 1

        ###----- 바운딩 박스가 여러개 검출될 수도 있기 때문에 가장 큰 바운딩 박스(사람 얼굴)의 검출 결과가 있는 인덱스만 -----###
        if len(faces) == 1:  # 검출된 바운딩 박스가 1개면
            print("1개 검출됨!")
            index = 0
        elif len(faces) >= 2:  # 검출된 바운딩 박스가 2개 이상이면
            print('2개 검출됨!')
            crop_size_list = []
            for face in faces:
                x, y, width, height = face['box']
                crop_size = width * height
                #print(crop_size)
                crop_size_list.append(crop_size)
                #print(crop_size_list)
            index = crop_size_list.index(max(crop_size_list))
            #print(index)
        else: # 바운딩 박스가 하나도 검출 되지 않았을 때
            print('0개 검출됨!')
            index = 0


        def draw_facebox(filename, result_list):
            data = plt.imread(filename)
            plt.imshow(data)
            ax = plt.gca()
            #print(result['box'])

            ## 검출된 박스에 해당하는 x, y, width, height 추출
            print('result_list: ',result_list)
            if len(result_list) == 0: # 주의. 함수 안에 다시 정의 안해놓으면 오류뜸!!!
                result_list = [{'box': [0, 0, 384, 512],
                                'confidence': 0.9731148481369019,
                                'keypoints': {'left_eye': (142, 207),
                                              'right_eye': (232, 205),
                                              'nose': (178, 249),
                                              'mouth_left': (137, 297),
                                              'mouth_right': (215, 298)}}]
            result = result_list[index]
            x, y, width, height = result['box']

            if width*height < 13000: # 이상하게 잡히는 box는 다 엄청 작은 크기였음. -> 평균적인 box 크기의 절반사이즈 보다 작으면 잘못 검출된거라고 보고 처리될 수 있도록 임의로 box위치 정해줌
                x, y, width, height = 0, 0, 384, 512

            rect = plt.Rectangle((x, y), width, height, fill=False, color='orange')
            #print('rect: ',rect)
            ax.add_patch(rect)

            # [x:x+width, y:y+height]일 줄 알았는데 출력해보니 x,y 좌표 둘 다 왼쪽아래에서 시작하는게 아니라  x좌표는 왼쪽아래 y좌표는 왼쪽위에서 시작이라 두 개 바꿔서 적어야 하는 듯
            crop_img = img[y-50:y + height+100, x-50:x + width+100] # 얼굴 부분만 crop하니까 머리카락도 너무 많이 잘린 상태로 crop돼서 남자인지 여자인지 구분이 안될 것 같음 -> -40, +40
            #path = '/opt/ml/P_level1/crop_images'
            crop_img_save_path = os.path.join('/opt/ml/P_level1/images(crop)', profile, file_name)# 000001_female_Asian_45/  +  mask1.jpg

            print("저장 완료!")
            print(crop_img_save_path)
            print()
            cv2.imwrite(crop_img_save_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            #plt.show()

        faces = detector.detect_faces(pixels)
        draw_facebox(filename, faces)

