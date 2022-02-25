'''
##### module   #####
torch.nn: 신경망을 만들고 훈련시키는 것을 돕기 위한 모듈


#####   class   #####
Object Detection
: 여러 object들을 Bounding Box를 통해 Localization(위치를 찾고)과 Classification(어떤 물체인지 분류)을 하는 작업
: https://herbwood.tistory.com/19

Object Detection은 크게 2가지 종류의 알고리즘이 있다.
1. R-CNN 계열의 two-stage detector
   : localization -> classification이 순차적으로 이루어짐
   : 정확도 성능으로는 two-stage detector가 좋지만 연산 속도가 오래 걸린다는 단점이 있음
2. YOLO, SSD 계열의 one-stage detector
   : localization과 classification을 동시에 처리함.
   : two-stage detector에 비해 "학습 중 클래스 불균형 문제"가 심하다는 문제점이 있음

class imbalance는 학습 시 두 가지 문제를 야기함
1. 대부분의 sample이 easy negative, 즉 모델이 class를 예측하기 쉬운 sample이기 때문에 유용한 기여를 하지 못해 학습이 비효율적으로 진행됨
2. easy negative의 수가 압도적으로 많기 때문에 학습에 끼치는 영향력이 커져 모델의 성능이 하락함
=> two-stage detector 계열의 모델은 이러한 문제를 해결하기 위해 여러가지 해결책을 적용했지만, 이러한 해결책은 one-stage detector에 적용하기 어려움
one-stage detector는 region proposal 과정이 없어 전체 이미지를 빽빽하게 순회하면서 sampling하는 dense sampling 방법을 수행하기 때문에 two-stage detector에 비해 훨씬 더 많은 후보 영역을 생성함. 다시 말해 class imbalance 문제가 two-stage detector보다 더 심각함.

Focal loss
: one-stage detector 모델에서 foreground와 background class 사이에 발생하는 극단적인 class imbalance(가령 1:1000)문제를 해결하는데 사용되며, 이진 분류에서 사용되는 Cross Entropy loss function으로부터 비롯되었음
: 마지막에 출력되는 각 클래스의 probability를 이용해 CE Loss에 통과된 최종 확률값이 큰 EASY 케이스의 Loss를 크게 줄이고 최종 확률 값이 낮은 HARD 케이스의 Loss를 낮게 줄이는 역할을 한다. 보통 CE는 확률이 낮은 케이스에 패널티를 주는 역할만 하고 확률이 높은 케이스에 어떠한 보상도 주지만 Focal Loss는 확률이 높은 케이스에는 확률이 낮은 케이스 보다 Loss를 더 크게 낮추는 보상을 주는 차이점이 있다.


LabelSmoothingLoss
: Multi-class 분류를 위한 cross entropy loss에서 목적 함수의 타겟으로 사용되는 라벨은 일반적으로 정확히 하나의 클래스만 명확히 표현하는(one-hot vector) hard 라벨이 사용된다. label smoothing 기법은 한 클래스가 전체를 모두 차지하는 hard 라벨을 정답 클래스의 비중을 약간 줄이고 나머지 클래스의 비중을 늘리는 soft 라벨로 변환하는 기법이다.
: hard target을 soft target으로 바꾸어 모델의 over confidence 문제를 해결할 수 있기에 모델의 일반화 성능이 향상
: 라벨스무딩을 cross entropy에 적용하는 방법은 단순합니다. 
  그저 일반적인 cross entropy식에서 truth label인 y(k)를 y(k)(1-α)+α/K의 식으로 구한 y′(k)로 바꾸면 됨
  즉, 정답 클래스 K가 차지하는 비중을 줄이면서 나머지 클래스에 대해서 α/K만큼의 uniform 분포를 줌으로써 발생할 수 있는 라벨 노이즈에 대해 사전 분포를 적용했다고 생각할 수 있음
: https://hongl.tistory.com/230


//재현율(Recall)- 실제 데이터가 True인 것 중에서 모델이 True라고 예측한 비율
  정밀도(Precision)- 모델이 True라고 예측한 정답 중에서 실제로 True인 비율
  서로의 확율이 반대이기 때문에(Trade-off관계) 어느 하나의 지표가 극단적으로 높은 경우에는 신뢰할만한 지표인지 잘 확인해보아야 함
  그러므로 데이터와 상황에 맞게 각각의 임계치를 정해두고 모델을 평가하도록 해야 함
  재현율과 정밀도의 임계치를 잘못 설정하면 극단적인 경우로 향할 수 이기 때문에 이를 방지하기 위해 F1loss(F1score)를 사용함

F1Loss(F1score)
: 재현율과 정밀도의 중요성이 같다고 가정하고, 두 지표의 조화평균으로 만들어진 지표
: 한쪽으로 치우치지 않는 모델을 만드는 데 유용할 수 있지만 이 지표의 수치가 높다고 해서 꼭 올바른 것만은 아님


### 다른 손실함수 찾아서 더 추가해놓기

'''


import torch
import torch.nn as nn
import torch.nn.functional as F


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim) # 먼저 .log_softmax 함수를 통해 log softmax를 구함 (나중에 cross entropy loss를 계산하기 위함)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1)) # α/K
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) # scatter_ 함수를 통해 target의 index에 해당하는 위치에 (1−α)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim)) # Log softmax와 target을 곱한 것의 음수를 취한 것이 cross entrophy loss가 됨


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
# [assert 조건] 이라 적었을 때 조건을 충족하지 않는다면 에러를 내라 할 때 사용
# if/else문이나 try/except문처럼 조건에 해당하지 않는 경우에 대응하지 않는 이유는 '에러가 절대 나지 않는다는 확신'을 갖고 있지만 일단 저것이 맞는지 검증하기 위한 용도로 사용하기 때문임
class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2   
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1) # softmax를 통해 함수에 들어오는 값들을 0~1의 확률값으로 바꿈

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32) #실제값 T, 예측값 P
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32) #실제값 T, 예측값 N
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32) #실제값 F, 예측값 P
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32) #실제값 F, 예측값 N

        precision = tp / (tp + fp + self.epsilon) #정밀도(모델이 True라고 예측한 정답 중에서 실제로 True인 비율)
        recall = tp / (tp + fn + self.epsilon) #재현율(실제 데이터가 True인 것 중에서 모델이 True라고 예측한 비율)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon) # 모든 요소를 [min, max]범위로 고정하여 Tensor로 출력
        return 1 - f1.mean()


_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'f1': F1Loss
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints

# **kwargs는 (키워드 = 특정 값) 형태로 함수를 호출할 수 있음. 결과값이 딕셔너리형태로 출력됨. 함수를 만들 때 키워드 인수는 가장 마지막으로 가야 함
def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion
