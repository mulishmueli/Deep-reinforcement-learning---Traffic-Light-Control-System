import argparse
import time
from numpy.linalg import norm
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    ref_frame_axies = []
    ref_frame_label= []
    cur_frame_axies = []
    cur_frame_label = []
    t_pre =0
    speed = 0

    min_distance = 50
    pixel_distance_in_meter = 0.024 #car length 4.5m resulation 1280*720

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader

    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


    for path, img, im0s, vid_cap in dataset:
        road_a = 0
        road_b = 0
        road_c = 0
        road_d = 0
        total_person = 0
        total_vehicle = 0


        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        #bus-0,car-1,motorcycle-2,person-3,truck-4
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=(0,1,2,3,4), agnostic=opt.agnostic_nms)#classes=opt.classes
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if (c==3) :
                        total_person= int(n)
                    else :
                        if ((c== 0) or(c== 1) or(c== 2) or (c== 4)):
                            total_vehicle= int(n)+total_vehicle
                # Write results
                cur_frame_axies = []
                cur_frame_label = []
                for *xyxy, conf, cls in reversed(det):
                    x=int(xyxy[0])
                    y=int(xyxy[1])
                    if ((cls== 0) or(cls== 1) or(cls== 2) or (cls== 4)):
                        if ((x < 318) and (y > 300) and (y < 460)):
                            road_a = road_a + 1
                        else:
                            if ((x > 360) and (x < 440) and (y < 100) or (
                                    (x > 425) and (x < 580) and (y > 100) and (y < 215)) or (
                                    (x > 440) and (x < 500) and (y > 50) and (y < 100))):
                                road_b = road_b + 1
                            else:
                                if ((x > 820) and (x < 1280) and (y > 145) and (y < 258)):
                                    road_c = road_c + 1
                                else:
                                    if ((x > 750) and (x < 1260) and (y > 400) and (y < 550) or (
                                            (x > 880) and (x < 1280) and (y > 550) and (y < 720))):
                                        road_d = road_d + 1


                    num_of_object = 0

                    if (len(ref_frame_label) > 0):
                        b = np.array([(int(xyxy[0]), int(xyxy[1]))])
                        a = np.array(ref_frame_axies)
                        distance = norm(a - b, axis=1)
                        min_value = distance.min()
                        if (min_value < min_distance):
                            idx = np.where(distance == min_value)[0][0]
                            num_of_object = ref_frame_label[idx]
                            if (num_of_object in cur_frame_label):
                                num_of_object=0
                            else:
                                t_delta = t2 - t1
                                speed=min_value*0.01*pixel_distance_in_meter*t_delta*3600
                    if (not (num_of_object)):
                        if (len(cur_frame_label)):
                            for j in range(1, max(cur_frame_label) + 2):
                                if (not (j in cur_frame_label)):
                                    num_of_object = j
                                    break
                        else:
                            num_of_object = 1
                    cur_frame_label.append(num_of_object)
                    cur_frame_axies.append((int(xyxy[0]), int(xyxy[1])))


                    label = f'{names[int(cls)]} {conf:.2f} {num_of_object} {speed:.0f}km/h'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            ref_frame_label = cur_frame_label
            ref_frame_axies = cur_frame_axies

            fps=1/(t2 - t1)

            print(f'{s}Done. ({t2 - t1:.3f}s)')
            cv2.putText(im0, f'FPS- {fps:.0f}  ', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2,
                        cv2.LINE_4)
            cv2.putText(im0, f'count of vehicle- {total_vehicle}  ', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2,
                        cv2.LINE_4)
            cv2.putText(im0, f'count of person- {total_person}  ', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2,
                        cv2.LINE_4)
            cv2.putText(im0, f' ROAD A- {road_a}  ', (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_4)
            cv2.putText(im0, f' ROAD B- {road_b}  ', (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_4)
            cv2.putText(im0, f' ROAD C- {road_c}  ', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_4)
            cv2.putText(im0, f' ROAD D- {road_d}  ', (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_4)
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
               # cv2.waitKey(1)  # 1 millisecond
                #time.sleep(0.06)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='0.005.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default="https://5e0d15ab12687.streamlock.net/live/AHISEMECH.stream/chunklist_w1349241609.m3u8", help='source')  # file/folder, 0 for webcam
    #parser.add_argument('--source', type=str,default="0",help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))


    detect()
