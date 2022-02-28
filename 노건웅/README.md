### 파일

**주의사항: 전체 dataset 결측치 수정**

-   data_eda.ipynb: class, 나이, 성별 갯수에 따른 데이터 분포
-   train.py: baseline 코드인 train.py에 f1 score를 추가한 버전
-   image_label2.csv: 새롭게 갱신된 dataset을 이용해 파일 이름과 경로 class label, 마스크 유무, 나이 등을 구분한 csv 파일

-   before_02-26: 2월 26일 이전에 갱신한 것
    -   new_data_eda
        -   data_augmentation.ipynb: 이미지 새로 추가 (따로 함수를 만들지 않아서 코드가 정돈되지 않았습니다.
        -   augmentation_label.csv: 새롭게 augmented한 이미지까지 추가한 전체 dataset, label 표시 csv 파일
    -   train3.csv: '001498-1', '004432'에서 성별이 잘못된 데이터를 변경
    -   image_label.csv
        -   id: id
        -   path: 파일 디렉토리 이름
        -   file_name: 이미지 파일 이름
        -   absolute_path: 이미지를 불러오기 위한 절대경로
        -   label: 0 ~ 17 중 어디에 해당하는지
    -   image_path_label.ipynb: image_label.csv 제작
    -   mask_dataset.ipynb: 주어진 이미지를 가져오는 간단한 Dataset 제작
