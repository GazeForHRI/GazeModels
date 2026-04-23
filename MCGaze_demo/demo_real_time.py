from mmcv.parallel import collate, scatter

import cv2
import torch
import numpy as np
import time
import sys
sys.path.insert(0, "..")
from mmdet.apis import init_detector

import pyrealsense2 as rs
from mmdet.datasets.pipelines import Compose

CLIP_SIZE = 3
# use gaze360 or l2cs by changing these paths
model = init_detector(
        '../configs/multiclue_gaze/multiclue_gaze_r50_l2cs.py',
        '../ckpts/multiclue_gaze_r50_l2cs.pth',
        device="cuda:0",
        cfg_options=None,)

#model = init_detector(
#        '../configs/multiclue_gaze/multiclue_gaze_r50_gaze360.py',
#        '../ckpts/multiclue_gaze_r50_gaze360.pth',
#        device="cuda:0",
#        cfg_options=None,)

cfg = model.cfg
#model = torch.load("../ckpts/multiclue_gaze_r50_l2cs.pth")

print(cfg.data.test.pipeline[1:])
test_pipeline = Compose(cfg.data.test.pipeline[1:])

def load_datas(data, test_pipeline, datas):
    datas.append(test_pipeline(data))

#img_metas =  [[{'ori_shape': (256, 256, 3), 'img_shape': (448, 448, 3), 'batch_input_shape': (448, 448, 3), 'pad_shape': (448, 448, 3), 'scale_factor': [1.75, 1.75, 1.75, 1.75], 'flip': False, #'flip_direction': None, 'img_norm_cfg': {'mean': [123.675, 116.28 , 103.53 ], 'std': [58.395, 57.12 , 57.375], 'to_rgb': True}}]]

def infer(cropped_img,model,clip,l, datas, clip_size=-1):
    if clip_size == -1:
        clip_size = CLIP_SIZE
    #cropped_img = scatter(cropped_img, ["cuda:0"])[0]
    #l = int(max(head_bboxes[3]-head_bboxes[1],head_bboxes[2]-head_bboxes[0])*0.8)
    #cropped_img = cropped_img.cpu().detach().numpy()[0].T        
    w_n,h_n,_ = cropped_img.shape
    cur_data = dict(filename=111,ori_filename=111,img=cropped_img,img_shape=(w_n,h_n,3),ori_shape=(2*l,2*l,3),img_fields=['img'])
    tmp_loaded_datas = []
    load_datas(cur_data,test_pipeline,tmp_loaded_datas)

    if len(datas)<clip_size:
        datas.extend(tmp_loaded_datas)
    else:
        datas.pop(0)
        datas.extend(tmp_loaded_datas)

    with torch.no_grad():
        datas = sorted(datas, key=lambda x:x['img_metas'].data['filename']) # 按帧顺序 img名称从小到大
        datas = collate(datas, samples_per_gpu=1) # 用来形成batch用的
        datas['img_metas'] = datas['img_metas'].data
        datas['img'] = datas['img'].data
        datas["img_metas"] = [{'filename': 0, 'ori_filename': 111, 'ori_shape': (254, 254, 3), 'img_shape': (448, 448, 3), 'pad_shape': (448, 448, 3), 'scale_factor': np.array([1.7637795, 1.7637795, 1.7637795, 1.7637795], dtype=float), 'flip': False, 'flip_direction': None, 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=float), 'std': np.array([58.395, 57.12 , 57.375], dtype=float), 'to_rgb': True}}]
        datas['img_metas'] = [[datas['img_metas'][0] for _ in range(clip_size)]] #[datas['img_metas'][0], datas['img_metas'][0], datas['img_metas'][0], datas['img_metas'][0], datas['img_metas'][0], datas['img_metas'][0], datas['img_metas'][0]]

        datas['img'] = [torch.stack([datas['img'][0][0] for _ in range(clip_size)], axis=0)]

        # print(len(datas["img_metas"]), len(datas["img_metas"][0]), len(datas["img_metas"][0][0]))
        # print(len(datas["img"]), len(datas["img"][0]), len(datas["img"][0][0]))
        datas = scatter(datas, ["cuda:0"])[0]
        #(det_bboxes, det_labels), det_gazes = model.simple_test(imgs=datas['img'], img_metas=datas['img_metas'])
        (det_bboxes, det_labels), det_gazes = model(
                return_loss=False,
                rescale=True,
                format=False,# 返回的bbox既包含face_bboxes也包含head_bboxes
                **datas)    # 返回的bbox格式是[x1,y1,x2,y2],根据return_loss函数来判断是forward_train还是forward_test.
    gaze_dim = det_gazes['gaze_score'].size(1)
    det_fusion_gaze = det_gazes['gaze_score'].view((det_gazes['gaze_score'].shape[0], 1, gaze_dim))
    clip['gaze_p'].append(det_fusion_gaze.cpu().numpy()) 

def infer_gaze(cur_img, l, frame_buffer, clip_size=-1):
    #w,h,_ = cur_img.shape
    if False:
        for j in range(len(head_bboxes)):
            for xy in head_bboxes[j]:
                xy = int(xy)
        head_center = [int(head_bboxes[j][1]+head_bboxes[j][3])//2,int(head_bboxes[j][0]+head_bboxes[j][2])//2]
        l = int(max(head_bboxes[j][3]-head_bboxes[j][1],head_bboxes[j][2]-head_bboxes[j][0])*0.8)
        head_crop = cur_img[max(0,head_center[0]-l):min(head_center[0]+l,w),max(0,head_center[1]-l):min(head_center[1]+l,h),:]
        w_n,h_n,_ = head_crop.shape
        # if frame==0:
        #     plt.imshow(head_crop)
        # print(head_crop.shape)
        cur_data = dict(filename=j,ori_filename=111,img=head_crop,img_shape=(w_n,h_n,3),ori_shape=(2*l,2*l,3),img_fields=['img'])
    clip = {"gaze_p": []}
    cropped_img = cur_img
    infer(cropped_img,model,clip,l, frame_buffer, clip_size)
    return clip

def process_real_time():
    #cap = cv2.VideoCapture('video_1.mp4')    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = np.asanyarray(frames.get_color_frame().get_data())
        color_frame = torch.tensor(color_frame).to(torch.device("cuda:0")).float()
        color_frame = torch.permute(color_frame, (2, 0, 1)).unsqueeze(0)
        clip = infer_gaze(color_frame, None)
        print(clip)

def visualize_gaze_mcgaze(cur_img, gaze, head_bboxes):
    gaze = gaze[0][0]
    for xy in head_bboxes:
        xy = int(xy)
    head_center = [int(head_bboxes[1]+head_bboxes[3])//2,int(head_bboxes[0]+head_bboxes[2])//2]
    l = int(max(head_bboxes[3]-head_bboxes[1],head_bboxes[2]-head_bboxes[0])*1)
    gaze_len = l*1.0
    thick = max(5,int(l*0.01))
    cv2.arrowedLine(cur_img,(head_center[1],head_center[0]),
                (int(head_center[1]-gaze_len*gaze[0]),int(head_center[0]-gaze_len*gaze[1])),
                (230,253,11),thickness=thick)    
    return cur_img

def visualize_gaze(frame, gaze_vector):
    """
    Visualize the gaze vector on the given frame.

    Args:
        frame: The image frame (numpy array).
        gaze_vector: The gaze estimation vector (numpy array).
    """
    h, w, _ = frame.shape
    center = (w // 2, h // 2)  # Assuming gaze vector points from center
    scale = 100  # Scale factor for visualization

    # Gaze vector components
    dx = int(gaze_vector[0][0][0] * scale)
    dy = int(gaze_vector[0][0][1] * scale)

    # End point of the vector
    end_point = (center[0] + dx, center[1] + dy)
    #l = int(max(w,h)*1)
    #gaze_len = l/20
    #end_point = (int(center[1]-gaze_len*dx),int(center[0]-gaze_len*dy))
    # Draw the vector on the frame
    cv2.arrowedLine(
        frame, center, end_point, (0, 255, 0), 2, tipLength=0.3
    )
    cv2.putText(
        frame,
        f"Gaze: [{gaze_vector[0][0][0]:.2f}, {gaze_vector[0][0][1]:.2f}, {gaze_vector[0][0][2]:.2f}]",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return frame

def process_real_time_with_visualization():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = np.asanyarray(frames.get_color_frame().get_data())
            
            # Preprocess the frame for inference
            color_frame_tensor = (
                torch.tensor(color_frame).to(torch.device("cuda:0")).float()
            )
            color_frame_tensor = torch.permute(color_frame_tensor, (2, 0, 1)).unsqueeze(0)

            # Perform gaze inference
            clip = infer_gaze(color_frame_tensor, None)
            if clip["gaze_p"]:
                gaze_vector = clip["gaze_p"][-1]  # Get the latest gaze vector

                # Visualize gaze on the original frame
                visualized_frame = visualize_gaze(color_frame.copy(), gaze_vector)

                # Show the visualized frame
                cv2.imshow("Gaze Estimation", visualized_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

#if __name__ == "__main__":
    # Run the visualization process
#    process_real_time_with_visualization()
