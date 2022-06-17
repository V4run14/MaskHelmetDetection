
# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import argparse
import time
import imutils

from resize import resize_image

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load

from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, \
    scale_coords, strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import os

class Arguments:
    def __init__(self, source, weights, cctv, img_size, conf_thres, iou_thres, device, project, name, augment, classes, agnostic_nms, update):
        self.source = source
        self.weights = weights
        self.cctv = cctv
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.project = project
        self.name = name
        self.augment = augment
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.update = update


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def detect():
    osource = 'data/images'
    oweights = 'runs/train/yolov5s_results/weights/best.pt'
    occtv = 0
    oimg_size = 416
    oconf_thres = 0.5
    oiou_thres = 0.45
    odevice = '0'
    oproject = 'runs/detect'
    oname = 'exp'
    oaugment = True
    oclasses = None
    oagnostic_nms = False
    oupdate = False
    opt = Arguments(osource, oweights, occtv, oimg_size, oconf_thres, oiou_thres, odevice, oproject, oname, oaugment,  oclasses,
                    oagnostic_nms, oupdate)
    check_requirements()
    source, weights, imgsz, cctv = opt.source, opt.weights, opt.img_size, opt.cctv

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
        modelc = load_classifier(name='resnet101', n=4)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    save_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference

    path = 'C:/Users/admin/Downloads/videoplayback (3).mp4'
    RTSP_URL = "rtsp://admin:Admin123$@10.11.25.53:554/user=admin_password='Admin123$'_channel='Streaming/Channels/'_stream=0.sdp"
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    #cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    #cap = cv2.VideoCapture('C:/Users/admin/Downloads/videoplayback (3).mp4')

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    fvs = FileVideoStream(path).start()
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 20)

    time.sleep(1.0)
    # start the FPS timer
    fps = FPS().start()
    count = 0
    prev_frame_time = 0
    new_frame_time = 0
    while fvs.more():
        frame = fvs.read()
        if count % 1 ==0:
            #cv2.imshow('RTSP stream', iframe)
            #success, iframe = cap.read()
            '''ret, buffer = cv2.imencode('.jpg', iframe)
            oframe = buffer.tobytes()
            #return Response(retStream(iframe), mimetype='multipart/x-mixed-replace; boundary=frame')
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + oframe + b'\r\n')  # concat frame one by one and show result'''
            #fps = cap.get(cv2.CAP_PROP_FPS)

            #print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
            img = resize_image(frame, 416, cv2.INTER_AREA)

            im0s = img  # BGR

            # Padded resize
            img = letterbox(im0s, 416, stride=stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0, frame = '', im0s, count

                # save_path = str(save_dir / 'images')  # img.jpg
                s += '%gx%g ' % img.shape[2:]  # print string
                #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
            new_frame_time = time.time()
            fpst = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            # converting the fps into integer
            fpst = int(fpst)
            # converting the fps to string so that we can display it on frame
            # by using putText function
            fpst = str(fpst)
            cv2.putText(im0, "FPS: {}".format(fpst),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            im0 = imutils.resize(im0, width=450)
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            im0 = np.dstack([im0, im0, im0])
            # display the size of the queue on the frame
            # show the frame and update the FPS counter
            #cv2.imshow("Frame", im0)

            #fps.update()

            ret, buffer = cv2.imencode('.jpg', im0)
            oframe = buffer.tobytes()
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + oframe + b'\r\n')  # concat frame one by one and show result

            fps.update()

            if cv2.waitKey(1) == 27:
                break
        count += 1
    #cap.release()
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    fvs.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/yolov5s_results/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--cctv', type=int, default=0, help='CCTV stream input') # 0 for image read, 1 for cctv read
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()