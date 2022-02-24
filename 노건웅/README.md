### 파일

**주의사항: '001498-1', '004432' 성별이 변경되었으니 해당 directory 이름 변경 필요**

-   new_data_eda
	- data_eda: class, 나이, 성별 갯수
	- data_augmentation.ipynb: 이미지 새로 추가 (따로 함수를 만들지 않아서 코드가 정돈되지 않았습니다.
	- augmentation_label.csv: 새롭게 augmented한 이미지까지 추가한 전체 dataset, label 표시 csv 파일
-   train3.csv: '001498-1', '004432'에서 성별이 잘못된 데이터를 변경
-   image_label.csv
    -   id: id
    -   path: 파일 디렉토리 이름
    -   file_name: 이미지 파일 이름
    -   absolute_path: 이미지를 불러오기 위한 절대경로
    -   label: 0 ~ 17 중 어디에 해당하는지
-   image_path_label.ipynb: image_label.csv 제작
-   mask_dataset.ipynb: 주어진 이미지를 가져오는 간단한 Dataset 제작
