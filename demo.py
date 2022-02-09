# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import imutils
import itertools

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo


# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--saliency_mask",
        help="A file or directory containing the saliency/transparency mask",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def sad_calculation(mask, lookup):
    # mask = cv2.resize(mask.astype('float64'), (500,500))
    # lookup = cv2.resize(lookup, (500, 500))

    # assert(mask.shape == lookup.shape)
    # assert(mask.dtype == lookup.dtype)

    mse_diff = ((mask - lookup) ** 2).sum()
    sad_diff = np.abs(mask - lookup).sum()

    # print(sad_diff)

    return sad_diff, mse_diff
    # pass

# to check if r2 is inside r1
def contains(r1, r2):
    return r1[0] < r2[0] < r2[2] < r1[2] and r1[1] < r2[1] < r2[3] < r1[3]


def remove_contained_cnts(cnts):
    cnt_indexes = list(itertools.combinations(range(len(cnts)), 2))

    remove_indexes = []

    keep_indexes = []

    cnts_ = []

    for index_pair in cnt_indexes:
        indexes = [index_pair[0], index_pair[1]]
        cnt1, cnt2 = cnts[index_pair[0]], cnts[index_pair[1]]
        rect_0 = cv2.boundingRect(cnt1)
        rect_0 = (rect_0[0], rect_0[1], rect_0[0] + rect_0[2], rect_0[1] + rect_0[3])
        rect_1 = cv2.boundingRect(cnt2)
        rect_1 = (rect_1[0], rect_1[1], rect_1[0] + rect_1[2], rect_1[1] + rect_1[3])

        # rect1 inside rect0
        if contains(rect_0, rect_1):
            #             print(rect_0, rect_1)
            remove_indexes.append(index_pair[1])

        # rect0 inside rect1
        elif contains(rect_1, rect_0):
            #             print(rect_1, rect_0)
            remove_indexes.append(index_pair[0])

    keep_indexes = [ind for ind in range(len(cnts)) if ind not in remove_indexes]

    cnts_ = [cnts[ind] for ind in keep_indexes]

    return cnts_

def get_contours(mask):
    masks = []
    mask = mask * 255
    mask = mask.astype('uint8')
    # cv2.imwrite('current_mask.jpg', mask)
    
    """
    draw a black line on all 4 corners of the image,
    this is done to avoid open contours later on
    """
    height, width = mask.shape
    cv2.line(mask, (0, 0), (0, height), (0, 0, 0), thickness=5)
    cv2.line(mask, (0, 0), (width, 0), (0, 0, 0), thickness=5)
    cv2.line(mask, (0, height), (width, height), (0, 0, 0), thickness=5)
    cv2.line(mask, (width, 0), (width, height), (0, 0, 0), thickness=5)
    
    edged = cv2.Canny(mask, 10, 30)
    kernel = np.ones((5, 5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    cnts = remove_contained_cnts(cnts)
    
    for cnt in cnts:
        img_cnt = np.zeros((mask.shape[0], mask.shape[1]))  # create a single channel 200x200 pixel black image plt.imshow(img_cnt)
        img_cnt = cv2.fillPoly(img_cnt, pts=[cnt], color=(255, 255, 255))
        # masks.append((np.logical_and(img_cnt, mask)*255).astype('uint8'))
        masks.append(np.logical_and(img_cnt, mask))
        # cv2.imwrite('contour.jpg', np.logical_and(img_cnt, mask)*255)

    return masks

def process_im_path(path, saliency_path):
    saliency_mask = cv2.imread(saliency_path, 0)
    img = read_image(path, format="BGR")
    saliency_mask = cv2.resize(saliency_mask, (img.shape[1], img.shape[0]))
    # result = np.zeros(img.shape[:2])
    result = []
    encountered_pixels = []
    append_count = 0
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(img)
    
    masks = predictions['instances'].pred_masks.cpu().detach().numpy()
    
    
    
    for mask in masks:
        #function to split mask into separate contours
        cnts = get_contours(mask)
        
        for cnt in cnts:
            lookup = np.where(cnt==True, saliency_mask/255.0 , 0)
            total_pixels = np.count_nonzero(cnt)
            
            # lookup = np.where(lookup > 0.05, lookup, 0)
            
            if not np.all(lookup == 0.) and total_pixels > 0:
                sad_diff, mse_diff = sad_calculation(cnt, lookup)
                # cv2.imwrite('lookup.jpg', lookup*255)
                # cv2.imwrite('mask.jpg', cnt*255)
                # lookup = saliency_mask[mask]
                # lookup_pixels = np.count_nonzero(lookup)
                thresh = sad_diff/total_pixels
                confidence = 1. - thresh
                encountered_pixels.append(cnt)
                
                if confidence > 0.85:
                # if 0.7 < confidence < 0.85:
                    # result[mask] = mask
                    result.append(cnt)     
                    append_count += 1
            # temp = 0
    
    #combine encountered pixels and saliency mask
    if len(result) > 0:
        result = np.array(result)
        result = np.sum(result, axis=0)
        result = np.where(result>0, 1, 0)
        result = result*255
        result = result.astype('uint8')
    else:
        result = np.zeros(img.shape[:2])
    
    if len(encountered_pixels) > 0:
        encountered_pixels = np.array(encountered_pixels)
        encountered_pixels = np.sum(encountered_pixels, axis=0)
        encountered_pixels = np.where(encountered_pixels>0, 1, 0)
        encountered_pixels = encountered_pixels*255
        encountered_pixels = encountered_pixels.astype('uint8')
    else:
        encountered_pixels = np.zeros(img.shape[:2])
    
    corrected_result = np.where(encountered_pixels > 0, result , saliency_mask)
    
    # img_trans = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # img_trans[:,:,3]= mask.astype('uint8')
    # cv2.imwrite('lol.png', img_trans)
    logger.info(
        "{}: {} in {:.2f}s".format(
            path,
            "detected {} out of {} instances ".format(append_count, len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )
    

    if args.output:
        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            out_filename = os.path.join(args.output, os.path.basename(path))
        else:
            assert len(args.input) == 1, "Please specify a directory with args.output"
            out_filename = args.output
        # visualized_output.save(out_filename)
        # cv2.imwrite(out_filename, mask)
        cv2.imwrite(out_filename, corrected_result)
        # cv2.imwrite(saliency_path, saliency_mask)
        # cv2.imwrite('pixels_encountered.jpg', encountered_pixels)
        # cv2.imwrite('result.jpg', result)
        
    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
        return

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, args)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        
        for inp_path in tqdm.tqdm(args.input, disable=not args.output):
            if os.path.isdir(inp_path):
                # temp = 0
                for file_path in os.listdir(inp_path):
                    saliency_path = os.path.join(args.saliency_mask, file_path.replace('.jpg', '_sal_fuse.png'))
                    file_path = os.path.join(inp_path, file_path)
                    out_filename = os.path.join(args.output, os.path.basename(file_path))
                    if not os.path.exists(out_filename):
                        process_im_path(file_path, saliency_path)
                    else:
                        print('skipping ', file_path)
                    # process_im_path(file_path)
            else:
                # for path in tqdm.tqdm(args.input, disable=not args.output):
                process_im_path(inp_path, args.saliency_mask)
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
