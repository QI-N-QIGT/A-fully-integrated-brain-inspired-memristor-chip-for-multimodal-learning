import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import base64
import time
import rospy
import torchaudio
from std_msgs.msg import String
import os
import time
import cv2
import torchvision
from PIL import Image as PILImage
from about_mutimodal import *
import rospy
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import threading
bridge = CvBridge()
sample_rate = 16000
n_mfcc = 20
sample_length = 32
transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={'n_fft': 2048, 'hop_length': sample_rate // sample_length})
lock1 = threading.Lock()
lock2 = threading.Lock()
lock3 = threading.Lock()
event = threading.Event()
g_speed_publisher = None
g_radar = None
g_speech = None
g_img = None
g_radar_count = 0
g_speech_count = 0
g_img_count = 0

def lidar_callback1(msg):
    global g_radar
    global g_radar_count
    g_radar_count += 1
    if g_radar_count % 5 == 0:
        res = controller.get_radar_feature(msg, method='a111', chip_id=0)
        print()
        with lock1:
            g_radar = res
    event.set()

def lidar_callback2(msg):
    global g_speech
    global g_speech_count
    g_speech_count += 1
    if g_speech_count % 1 == 0:
        res = controller.get_speech_feature(msg, method='a111', chip_id=3)
        print()
        with lock2:
            g_speech = res
    event.set()

def lidar_callback3(msg):
    global g_img
    global g_img_count
    g_img_count += 1
    if g_img_count % 2 == 0:
        res = controller.get_img_feature(msg, method='a111', chip_id=6)
        print()
        with lock3:
            g_img = res
    event.set()

def listener():
    global g_speed_publisher
    rospy.init_node('a111_node')
    rospy.Subscriber('/scan', LaserScan, lidar_callback1)
    rospy.Subscriber('/audio_data', String, lidar_callback2)
    rospy.Subscriber('/output_video_frame', Image, lidar_callback3)
    g_speed_publisher = rospy.Publisher('/a111_car_speed', Float32MultiArray, queue_size=10)
    print()
    rospy.spin()

def publish_car_speed(speed_x, speed_y):
    global g_speed_publisher
    array_msg = Float32MultiArray()
    array_msg.layout.dim.append(MultiArrayDimension())
    array_msg.layout.dim[0].label = 'speed'
    array_msg.layout.dim[0].size = 2
    array_msg.layout.dim[0].stride = 1
    print()
    array_msg.data.extend([speed_x, speed_y])
    if not rospy.is_shutdown():
        print()
        g_speed_publisher.publish(array_msg)

def thread_fusion(name):
    global g_radar, g_speech, g_img
    while True:
        event.wait()
        if g_radar is not None and g_speech is not None and (g_img is not None):
            with lock1:
                with lock2:
                    with lock3:
                        print()
                        (speed_x, speed_y) = controller.get_mutimodal_speed(g_radar, g_speech, g_img, method='a111', chip_id=1)
            publish_car_speed(speed_x, speed_y)
        event.clear()

def main():
    threads = []
    thread = threading.Thread(target=thread_fusion, args=(1,))
    threads.append(thread)
    thread.start()
    listener()

class Evaluator:

    def __init__(self, modal, simulated_data_root, chip_num, timestamp, offset_row_begin=8, step=5):
        self.modal = modal
        self.simulated_data_root = simulated_data_root
        self.offset_row_begin = offset_row_begin
        self.chip_num = chip_num
        self.timestamp = timestamp
        rram_results = 'rram_results'
        save_root = os.path.join(rram_results, 'chip%d_%d' % (chip_num, timestamp))
        begin = time.time()
        (_, self.onnx_obj, self.onnx_layer_info, self.quant_info, self.each_layer_outs_s) = get_model_middle_results_use_tool_chains(simulated_data_root, begin_layer_name='graph_input', step=step)
        print()
        save_path = os.path.join(save_root, 'repaired_care_name_2_map_infos.npy')
        self.repaired_care_name_2_map_infos = np.load(save_path, allow_pickle=True).item()

    def get_cal_result_use_a111(self, input_data, input_node_name1='graph_input', output_node_name1='Relu_0', input_node_name2='Relu_0', output_node_name2='MatMul_2', chip_id=0):
        input_s = 1
        input_node_and_data = {input_node_name1: {'data': input_data, 's': input_s}}
        datas = get_output_from_specified_node(self.onnx_obj, input_node_and_data, output_node_name1, self.onnx_layer_info, callback, save_path=None, paras=self.quant_info)
        input_data_s = datas
        input_data = input_data_s['data']
        input_s = input_data_s['s']
        inputs = {input_node_name2: tuple([input_data, input_s])}
        outputs = cal_conv_fc_used_a111_chip_and_other_used_tool_chains(self.onnx_obj, self.onnx_layer_info, self.quant_info, self.each_layer_outs_s, self.repaired_care_name_2_map_infos, offset_row_begin=self.offset_row_begin, inputs=inputs, outputs=[output_node_name2], chip_id=chip_id)
        return outputs[output_node_name2][0]

    def get_cal_result_use_tool(self, input_data, input_node_name1='graph_input', output_node_name1='Relu_0', input_node_name2='Relu_0', output_node_name2='MatMul_2', chip_id=0):
        input_s = 1
        input_node_and_data = {input_node_name1: {'data': input_data, 's': input_s}}
        datas = get_output_from_specified_node(self.onnx_obj, input_node_and_data, output_node_name1, self.onnx_layer_info, callback, save_path=None, paras=self.quant_info)
        input_data_s = datas
        input_data = input_data_s['data']
        input_s = input_data_s['s']
        input_node_and_data = {input_node_name2: {'data': input_data, 's': input_s}}
        datas = get_output_from_specified_node(self.onnx_obj, input_node_and_data, output_node_name2, self.onnx_layer_info, callback, save_path=None, paras=self.quant_info)
        return datas['data']

class Controller:

    def __init__(self):
        print()
        chip_num = 4043
        timestamp = 202407151600
        simulated_data_root = 'simulated_data/radar_cls_quant_nat40_GE_HL4bit_20240708-073834'
        modal = 'radar'
        begin_layer_shape = [256]
        begin_layer_name = 'Relu_0'
        img_number = 1000
        step = 10
        debug_img_number = img_number // step
        self.radar = Evaluator(modal, simulated_data_root, chip_num, timestamp, step=step)
        print()
        print()
        print()
        print()
        chip_num = 4044
        timestamp = 202407161600
        simulated_data_root = 'simulated_data/speech_cls_quant_nat_GE_HL4bit_20240708-073737'
        modal = 'speech'
        begin_layer_shape = [256]
        begin_layer_name = 'Relu_0'
        img_number = 500
        step = 5
        debug_img_number = img_number // step
        self.speech = Evaluator(modal, simulated_data_root, chip_num, timestamp, step=step)
        self.speech.sample_rate = 16000
        self.speech.n_mfcc = 20
        self.speech.sample_length = 32
        self.speech.transform = torchaudio.transforms.MFCC(sample_rate=self.speech.sample_rate, n_mfcc=self.speech.n_mfcc, melkwargs={'n_fft': 2048, 'hop_length': self.speech.sample_rate // self.speech.sample_length})
        print()
        print()
        print()
        print()
        chip_num = 4045
        timestamp = 202407191000
        simulated_data_root = 'simulated_data/image_cls_quant_nat_GE_HL4bit_20240708-100228'
        modal = 'img'
        begin_layer_shape = [64, 28, 28]
        begin_layer_name = 'Relu_0'
        img_number = 1000
        step = 10
        debug_img_number = img_number // step
        self.img = Evaluator(modal, simulated_data_root, chip_num, timestamp, step=step)
        self.img.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((56, 56)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        print()
        print()
        print()
        print()
        chip_num = 4046
        timestamp = 202407171800
        simulated_data_root = 'simulated_data/multimodal_cls_quant_nat_GE_HL4bit_20240710-113200'
        modal = 'mutimodal'
        begin_layer_shape = [128]
        begin_layer_name = 'Relu_0'
        img_number = 1000
        step = 10
        debug_img_number = img_number // step
        self.mutimodal = Evaluator(modal, simulated_data_root, chip_num, timestamp, step=step)
        print()
        print()
        print()
        print()
        self.count = 0
        self.last_x = 0
        self.last_y = 0
        self.ladar_frequent = 5
        self.last_time = time.time()
        self.radar_time = time.time()
        self.speech_time = time.time()
        self.img_time = time.time()
        self.mutimodal_time = time.time()

    def get_radar_feature(self, msg, method='sim', chip_id=1):
        begin = time.time()
        dis = np.array(msg.ranges)
        index = np.array(msg.intensities) == 0
        dis[index] = np.max(dis[(1 - index).astype(np.bool8)])
        input_data = dis.reshape(1, -1).astype(np.float32)
        if method == 'sim':
            radar_fun = self.radar.get_cal_result_use_tool
        else:
            radar_fun = self.radar.get_cal_result_use_a111
        f_radar = radar_fun(input_data, input_node_name1='graph_input', output_node_name1='Relu_0', input_node_name2='Relu_0', output_node_name2='Relu_1', chip_id=chip_id)
        f_radar = f_radar * self.radar.each_layer_outs_s['Relu_1']['s']
        end = time.time()
        print()
        print()
        self.radar_time = time.time()
        return f_radar

    def get_speech_feature(self, msg, method='sim', chip_id=1):
        begin = time.time()
        data = msg.data
        a_data = base64.b64decode(data)
        data = np.frombuffer(a_data, dtype=np.int16).reshape(-1).astype(np.float32) / 2 ** 15
        data = torch.from_numpy(data)
        waveform = self.speech.transform(data)
        waveform = waveform[..., :-1]
        input_data = waveform.reshape(1, -1).numpy().astype(np.float32)
        input_data = (input_data - input_data.min()) / (input_data.max() - input_data.min())
        if method == 'sim':
            speech_fun = self.speech.get_cal_result_use_tool
        else:
            speech_fun = self.speech.get_cal_result_use_a111
        f_speech = speech_fun(input_data, input_node_name1='graph_input', output_node_name1='Relu_0', input_node_name2='Relu_0', output_node_name2='Relu_1', chip_id=chip_id)
        f_speech = f_speech * self.speech.each_layer_outs_s['Relu_1']['s']
        end = time.time()
        print()
        print()
        self.speech_time = time.time()
        return f_speech

    def get_img_feature(self, msg, method='sim', chip_id=1):
        begin = time.time()
        cv_image = bridge.imgmsg_to_cv2(msg, 'bgr8')[:, :, ::-1]
        image = PILImage.fromarray(cv_image)
        data = self.img.transform(image)[None]
        input_data = np.array(data).astype(np.float32)
        if method == 'sim':
            img_fun = self.img.get_cal_result_use_tool
        else:
            img_fun = self.img.get_cal_result_use_a111
        f_img = img_fun(input_data, input_node_name1='graph_input', output_node_name1='Relu_0', input_node_name2='Relu_0', output_node_name2='Relu_3', chip_id=chip_id)
        f_img = f_img * self.img.each_layer_outs_s['Relu_3']['s']
        end = time.time()
        print()
        print()
        self.img_time = time.time()
        return f_img

    def get_mutimodal_speed(self, f_radar, f_speech, f_img, method='sim', chip_id=1):
        begin = time.time()
        if method == 'sim':
            mutimodal_fun = self.mutimodal.get_cal_result_use_tool
        else:
            mutimodal_fun = self.mutimodal.get_cal_result_use_a111
        mutimodal_data = np.concatenate([f_speech, f_img, f_radar], axis=1).astype(np.float32)
        outputs = mutimodal_fun(mutimodal_data, chip_id=chip_id)
        print()
        print()
        print()
        label = outputs.argmax(axis=1)
        if label == 0:
            (x, y) = (0.1, 0)
        elif label == 1:
            (x, y) = (0.1, 0.02)
        else:
            (x, y) = (0.0, 0.0)
        end = time.time()
        print()
        print()
        self.mutimodal_time = time.time()
        return (x, y)
if __name__ == '__main__':
    controller = Controller()
    main()