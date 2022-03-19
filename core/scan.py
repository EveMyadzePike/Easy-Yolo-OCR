
import yaml
from core.util import *
from core.general import *
from core.image_handler import ImagePack

import pandas as pd


# pt 모델 설정 세팅
def model_setting(model, half, imgz):
    if half:
        model.half()
    stride = int(model.stride.max())
    img_size = check_img_size(imgz, s=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    return model, stride, img_size, names


# pt 검출
def detecting(model, img, im0s, device, img_size, half, option, ciou=20):
    confidence, iou = option
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))

    # 이미지 정규화
    print("trying to call torch .from_numpy")
    img = torch.from_numpy(img).to(device)
    print("called torch from numpy")
    print("trying to call imgh.float")
    img = img.half() if half else img.float()
    print("called img float")
    img /= 255.0
    if img.ndimension() == 3:
        print("trying to call img.unsqueexe")
        img = img.unsqueeze(0)
        print("called img.unsqueeze")
        
    # 추론 & NMS 적용
    print("trying to call model")
    prediction = model(img, augment=False)[0]
    print("called model")
    print("trying to call nms")
    prediction = non_max_suppression(prediction, confidence, iou, classes=None, agnostic=False)
    print("called nms")
    
    detect = None
    for _, det in enumerate(prediction):
        obj, det[:, :4] = {}, scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
        print("trying to det.cpu \n")
        detect = det.cpu()
        print("called det.cpu \n")

    # 중복 상자 제거
    detList = []
    for *rect, conf, cls in detect:
        detList.append((rect, conf, cls))
    
    print("trying to call unsorted remove intersect box det \n")
    detect = unsorted_remove_intersect_box_det(detList, ciou)
    print("called unsoretd remove \n")
    
    return detect


def detection(det, names):
    det.sort(key=lambda x: x[2])
    label_list = []
    rect_list = []
    for *rect, conf, cls in det:
        rects = rect[0][0]
        label_list.append(names[int(cls)])
        rect_list.append([int(rects[0]), int(rects[2]), int(rects[1]), int(rects[3])])

    return rect_list, label_list


def pt_detect(path, device, models, ciou, reader, gray=False, byteMode=False):
    driver_weights = models

    half = device.type != 'cpu'
    # config 로드
    with open('config.yaml', 'r') as f:
        print("trying to load config.yaml")
        config = yaml.safe_load(f)
        print("loaded config.yaml")
    img_size, confidence, iou = config['detection-size'], config['detection-confidence'], config['detection-iou']
    detection_option = (img_size, confidence, iou)
    f.close()
    
    print("Trying to call model_setting")
    model, stride, img_size, names = model_setting(driver_weights, half, detection_option[0])
    print("successfully called model_settting")
    print("Trying to call Image Pack")
    image_pack = ImagePack(path, img_size, stride, byteMode=byteMode, gray=gray)
    print("successfuly called Image Pack")
    img, im0s = image_pack.getImg()
    print("trying to call detecting")
    det = detecting(model, img, im0s, device, img_size, half, detection_option[1:], ciou)
    print("successfully called detecting \n")
    print("trying to call detection")
    rect_list, label_list = detection(det, names)
    print("called detection")
    print("trying to call reader.recogss")
    result = reader.recogss(im0s, rect_list)
    print("called reader.recogss")
    
    result_line, i = [], 0
    print(f'----------- {path} -----------')
    for r in result:
        line = r[1]
        print(f'{label_list[i]} : {line}')
        result_line.append(line)
        i += 1
    print('-------------------------------')
