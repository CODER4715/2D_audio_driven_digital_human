import numpy as np


# batched detection
from PIL import Image
import cv2

def batched_transform(self, frames, use_origin_size):
    """
    Arguments:
        frames: a list of PIL.Image, or torch.Tensor(shape=[n, h, w, c],
            type=np.float32, BGR format).
        use_origin_size: whether to use origin size.
    """
    from_PIL = True if isinstance(frames[0], Image.Image) else False

    # convert to opencv format
    if from_PIL:
        frames = [cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR) for frame in frames]
        frames = np.asarray(frames, dtype=np.float32)

    # testing scale
    im_size_min = np.min(frames[0].shape[0:2])
    im_size_max = np.max(frames[0].shape[0:2])
    resize = float(self.target_size) / float(im_size_min)

    # prevent bigger axis from being more than max_size
    if np.round(resize * im_size_max) > self.max_size:
        resize = float(self.max_size) / float(im_size_max)
    resize = 1 if use_origin_size else resize

    # resize
    if resize != 1:
        if not from_PIL:
            frames = F.interpolate(frames, scale_factor=resize)
        else:
            frames = [
                cv2.resize(frame, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                for frame in frames
            ]

    # convert to torch.tensor format
    # if not from_PIL:
    #     frames = frames.transpose(1, 2).transpose(1, 3).contiguous()
    # else:
    #     frames = frames.transpose((0, 3, 1, 2))
    #     frames = torch.from_numpy(frames)
    frames = frames.transpose((0, 3, 1, 2))
    # frames = torch.from_numpy(frames)

    return frames, resize
def __detect_faces(inputs):
    # get scale
    height, width = inputs.shape[2:]
    # self.scale = torch.tensor([width, height, width, height], dtype=torch.float32).to(device)
    scale = np.array([width, height, width, height], dtype=np.float32)
    tmp = [width, height, width, height, width, height, width, height, width, height]
    # self.scale1 = torch.tensor(tmp, dtype=torch.float32).to(device)
    scale1 = np.array(tmp, dtype=np.float32)

    # forawrd
    # inputs = inputs.to(device)
    inputs = inputs
    # if self.half_inference:
    #     inputs = inputs.half()
    loc, conf, landmarks = self(inputs)

    # get priorbox
    priorbox = PriorBox(self.cfg, image_size=inputs.shape[2:])
    # priors = priorbox.forward().to(device)
    priors = priorbox.forward()

    return loc, conf, landmarks, priors

# def batch_detect(net, imgs, device):
def batch_detect(frames, conf_threshold = 0.8, nms_threshold = 0.4, use_origin_size = True):

    frames, resize = batched_transform(frames, use_origin_size)
    frames = frames
    frames = frames - np.array([104, 117, 123])

    b_loc, b_conf, b_landmarks, priors = self.__detect_faces(frames)

    final_bounding_boxes, final_landmarks = [], []

    # decode
    priors = priors.unsqueeze(0)
    b_loc = batched_decode(b_loc, priors, self.cfg['variance']) * self.scale / self.resize
    # b_landmarks = batched_decode_landm(b_landmarks, priors, self.cfg['variance']) * self.scale1 / self.resize
    b_conf = b_conf[:, :, 1]

    # index for selection
    b_indice = b_conf > conf_threshold

    # concat
    b_loc_and_conf = torch.cat((b_loc, b_conf.unsqueeze(-1)), dim=2).float()

    for pred, landm, inds in zip(b_loc_and_conf, b_landmarks, b_indice):

        # ignore low scores
        # pred, landm = pred[inds, :], landm[inds, :]
        pred = pred[inds, :]
        if pred.shape[0] == 0:
            final_bounding_boxes.append(np.array([], dtype=np.float32))
            # final_landmarks.append(np.array([], dtype=np.float32))
            continue

        # to CPU
        # bounding_boxes, landm = pred.cpu().numpy(), landm.cpu().numpy() #原本
        # bounding_boxes, landm = pred.cpu().detach().numpy(), landm.cpu().detach().numpy()
        bounding_boxes = pred.cpu().detach().numpy()

        # NMS
        keep = py_cpu_nms(bounding_boxes, nms_threshold)
        # bounding_boxes, landmarks = bounding_boxes[keep, :], landm[keep]
        bounding_boxes = bounding_boxes[keep, :]

        # append
        d = bounding_boxes[0]
        d = np.clip(d, 0, None)
        x1, y1, x2, y2 = map(int, d[:-1])
        final_bounding_boxes.append((x1, y1, x2, y2))
        # final_bounding_boxes.append(bounding_boxes)
        # final_landmarks.append(landmarks)
    # self.t['forward_pass'].toc(average=True)
    # self.batch_time += self.t['forward_pass'].diff
    # self.total_frame += len(frames)
    # print(self.batch_time / self.total_frame)

    return final_bounding_boxes


    imgs = imgs - np.array([104, 117, 123])
    imgs = imgs.transpose(0, 3, 1, 2)
    imgs = np.array(imgs, dtype=np.float32)
    # if 'cuda' in device:
    #     torch.backends.cudnn.benchmark = True

    # imgs = torch.from_numpy(imgs).float().to(device)
    BB, CC, HH, WW = imgs.shape
    # with torch.no_grad():
        # olist = net(imgs)
    olist = net.run(None, {'img': imgs})

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = softmax(olist[i * 2], axis=1)
    # olist = [oelem.data.cpu() for oelem in olist]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        # FB, FC, FH, FW = ocls.size()  # feature map size
        FB, FC, FH, FW = ocls.shape
        stride = 2**(i + 2)    # 4,8,16,32,64,128
        anchor = stride * 4
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls[:, 1, hindex, windex]
            loc = oreg[:, :, hindex, windex].reshape(BB, 1, 4)
            priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
            priors = priors.reshape(1, 1, 4)
            # priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]).view(1, 1, 4)
            variances = [0.1, 0.2]
            box = batch_decode(loc, priors, variances)
            box = box[:, 0] * 1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            score = np.expand_dims(score,axis=1)
            bboxlist.append(np.concatenate([box, score], 1))
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, BB, 5))

    return bboxlist