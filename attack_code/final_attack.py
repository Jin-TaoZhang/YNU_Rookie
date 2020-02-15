from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import argparse
import functools
import numpy as np
import paddle.fluid as fluid
import cv2
import gc
import models
import paddle.fluid as fluid
from utils import init_prog, save_adv_image, process_img, tensor2img, calc_mse, add_arguments, print_arguments, img2tensor

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

add_arg('class_dim',        int,   121,                  "Class number.")
add_arg('shape',            str,   "3,224,224",          "output image shape")
add_arg('input',            str,   "../input_image/",     "Input directory with images")
add_arg('output',           str,   "../output_image/",    "Output directory with images")

args = parser.parse_known_args()[0]
print_arguments(args)

image_shape = [int(m) for m in args.shape.split(",")]
class_dim=args.class_dim
input_dir = args.input
output_dir = args.output
raw_image_size= 224
resize_image_size = 255
prob = 0.5
val_list = 'val_list.txt'
use_gpu = True


"""
定义缩放函数
"""
def scale(img):
    chance = np.random.normal(0,1,1)[0]
    resize = int(np.random.uniform(raw_image_size,resize_image_size,1)[0])
    resize_image = fluid.layers.image_resize(img,(resize,resize))
    hight = resize_image_size - resize
    width = resize_image_size - resize
    pad_top = int(np.random.uniform(0, hight, 1)[0])
    pad_bottom = hight-pad_top
    pad_left = int(np.random.uniform(0, width, 1)[0])
    pad_right = width - pad_left
    scale_img = fluid.layers.pad(resize_image, [0,0,0,0,pad_top,pad_bottom,pad_left,pad_right],pad_value=0)
    scale_img = fluid.layers.image_resize(scale_img,(224,224))
    if chance>prob:
        return scale_img
    else :
        return img

"""
初始化模型信息（注释部分为对结果影响不大的模型）
"""
resnet_model_name = "ResNeXt50_32x4d"
densenet_model_name = "DenseNet161"
vgg_model_name = "VGG19"
inception_name = "InceptionV4"
mobilenet_model_name = "MobileNetV2_x2_0"
adv_resnet_model_name = 'ADV_ResNeXt50_32x4d'
resnet_vd_model_name = "ResNeXt50_vd_64x4d"
resnet_vd_32_model_name = "ResNeXt50_vd_32x4d"
adv_densenet_name = "adv_DenseNet161"
darknet_model_name = "DarkNet53"
efficient_model_name = "EfficientNetB7"
dpn_model_name = "DPN131"
sufflenet_model_name = "ShuffleNetV2_x2_0"
hrnet_model_name = "HRNet_W64_C"
adv_vgg_model_name = "adv_VGG19"
adv_paper_model_name = "ADV_paper_ResNeXt50_32x4d"
# adv_paper_resnet_vd_name = "adv_paper_ResNeXt50_vd_64x4d"
# adv_paper_darknet_name = "adv_paper_DarkNet53"
# adv_paper_dpn_name = "adv_paper_DPN131"
# adv_paper_efficient_name = "adv_paper_EfficientNetB7"
# adv_paper_resnet_1000_name = "ADV_paper_1000_ResNeXt50_32x4d"

pretrained_model = "../moadel_all"


input_layer = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
y_label = fluid.layers.data(name='label', shape=[1], dtype='int64')
input_layer.stop_gradient=False

model_resnet = models.__dict__[resnet_model_name]()
out_logits_resnet = model_resnet.net(input=scale(input_layer), class_dim=class_dim)
out_resnet = fluid.layers.softmax(out_logits_resnet)

model_densenet = models.__dict__[densenet_model_name]()
out_logits_densenet = model_densenet.net(input=scale(input_layer), class_dim=class_dim)
out_densenet = fluid.layers.softmax(name='densenet_softmax',input=out_logits_densenet)

model_vgg = models.vgg.VGG19()
out_logits_vgg = model_vgg.net(input=scale(input_layer), class_dim=class_dim)
out_vgg = fluid.layers.softmax(out_logits_vgg)

model_inception = models.inception_v4.InceptionV4()
out_logits_inception = model_inception.net(input=scale(input_layer), class_dim=class_dim)
out_inception = fluid.layers.softmax(out_logits_inception)

model_mobilenet = models.__dict__[mobilenet_model_name]()
out_logits_mobilenet = model_mobilenet.net(input=scale(input_layer), class_dim=class_dim)
out_mobilenet = fluid.layers.softmax(out_logits_mobilenet)

model_adv_resnet= models.__dict__[adv_resnet_model_name]()
out_logits_adv_resnet = model_adv_resnet.net(input=scale(input_layer), class_dim=class_dim)
out_adv_resnet= fluid.layers.softmax(out_logits_adv_resnet)

model_resnet_vd = models.__dict__[resnet_vd_model_name]()
out_logits_resnet_vd = model_resnet_vd.net(input=scale(input_layer), class_dim=class_dim)
out_resnet_vd = fluid.layers.softmax(out_logits_resnet_vd)

model_resnet_vd_32 = models.__dict__[resnet_vd_32_model_name]()
out_logits_resnet_vd_32 = model_resnet_vd_32.net(input=scale(input_layer), class_dim=class_dim)
out_resnet_vd_32 = fluid.layers.softmax(out_logits_resnet_vd_32)

model_adv_densenet = models.adv_DenseNet161()
out_logits_adv_densenet = model_adv_densenet.net(input=scale(input_layer), class_dim=class_dim)
out_adv_densenet = fluid.layers.softmax(out_logits_adv_densenet)

model_darknet = models.__dict__[darknet_model_name]()
out_logits_darknet = model_darknet.net(input=scale(input_layer), class_dim=class_dim)
out_darknet = fluid.layers.softmax(out_logits_darknet)

model_efficient = models.__dict__[efficient_model_name]()
out_logits_efficient = model_efficient.net(input=scale(input_layer), class_dim=class_dim)
out_efficient = fluid.layers.softmax(out_logits_efficient)

model_dpn = models.__dict__[dpn_model_name]()
out_logits_dpn = model_dpn.net(input=scale(input_layer), class_dim=class_dim)
out_dpn = fluid.layers.softmax(out_logits_dpn)

model_suffle = models.__dict__[sufflenet_model_name]()
out_logits_suffle = model_suffle.net(input=scale(input_layer), class_dim=class_dim)
out_suffle = fluid.layers.softmax(out_logits_suffle)

# model_hrnet = models.__dict__[hrnet_model_name]()
# out_logits_hrnet = model_hrnet.net(input=input_layer, class_dim=class_dim)
# out_hrnet = fluid.layers.softmax(out_logits_hrnet)

model_adv_vgg = models.__dict__[adv_vgg_model_name]()
out_logits_adv_vgg = model_adv_vgg.net(input=scale(input_layer), class_dim=class_dim)
out_adv_vgg = fluid.layers.softmax(out_logits_adv_vgg)

model_paper_resnet = models.__dict__[adv_paper_model_name]()
out_logits_paper_resnet = model_paper_resnet.net(input=scale(input_layer), class_dim=class_dim)
out_paper_resnet = fluid.layers.softmax(out_logits_paper_resnet)

# model_paper_dpn = models.__dict__[adv_paper_dpn_name]()
# out_logits_paper_dpn = model_paper_dpn.net(input=input_layer, class_dim=class_dim)
# out_paper_dpn = fluid.layers.softmax(out_logits_paper_dpn)


# model_paper_resnet_vd = models.__dict__[adv_paper_resnet_vd_name]()
# out_logits_paper_resnet_vd = model_paper_resnet_vd.net(input=input_layer, class_dim=class_dim)
# out_paper_resnet_vd = fluid.layers.softmax(out_logits_paper_resnet_vd)

# model_paper_darknet = models.__dict__[adv_paper_darknet_name]()
# out_logits_paper_darknet = model_paper_darknet.net(input=input_layer, class_dim=class_dim)
# out_paper_darknet = fluid.layers.softmax(out_logits_paper_darknet)

# model_paper_efficient = models.__dict__[adv_paper_efficient_name]()
# out_logits_paper_efficient = model_paper_efficient.net(input=input_layer, class_dim=class_dim)
# out_paper_efficient = fluid.layers.softmax(out_logits_paper_efficient)

# model_paper_resnet_1000 = models.__dict__[adv_paper_resnet_1000_name]()
# out_logits_paper_resnet_1000 = model_paper_resnet_1000.net(input=scale(input_layer), class_dim=class_dim)
# out_paper_resnet_1000 = fluid.layers.softmax(out_logits_paper_resnet_1000)

place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)

fluid.io.load_persistables(exe,pretrained_model, main_program=fluid.default_main_program())
init_prog(fluid.default_main_program())
eval_program = fluid.default_main_program().clone(for_test=True)

logits = (out_logits_resnet+out_logits_densenet+out_logits_vgg+out_logits_inception+out_logits_mobilenet+ out_logits_adv_resnet +out_logits_resnet_vd+out_logits_resnet_vd_32+out_logits_adv_densenet+out_logits_darknet+out_logits_efficient+out_logits_dpn + out_logits_suffle + out_logits_adv_vgg  +out_logits_paper_resnet)/15
# logits = out_logits_adv_resnet
loss = fluid.layers.softmax_with_cross_entropy(logits=logits,label=y_label)
gradients = fluid.backward.gradients(targets=loss,inputs=[input_layer])[0]

def inference(img,label):
    label = np.expand_dims(np.expand_dims(label,axis=0),axis=0).astype('int64')
    result_resnet = exe.run(eval_program,
                     fetch_list=[out_resnet],
                     feed={input_layer.name:img,y_label.name:label})
    result_resnet = result_resnet[0][0]
    pred_resnet_label = np.argmax(result_resnet)
    pred_resnet_score = result_resnet[pred_resnet_label].copy()



    result_densenet = exe.run(eval_program,
                            fetch_list=[out_densenet],
                              feed={input_layer.name: img, y_label.name: label})
    result_densenet = result_densenet[0][0]
    pred_densenet_label = np.argmax(result_densenet)
    pred_densenet_score = result_densenet[pred_densenet_label].copy()


    result_vgg = exe.run(eval_program,
                            fetch_list=[out_vgg],
                            feed={input_layer.name:img,y_label.name:label})
    result_vgg = result_vgg[0][0]
    pred_vgg_label = np.argmax(result_vgg)
    pred_vgg_score = result_vgg[pred_vgg_label].copy()


    result_inc = exe.run(eval_program,
                         fetch_list=[out_inception],
                         feed={input_layer.name:img,y_label.name:label})
    result_inc = result_inc[0][0]
    pred_inc_label = np.argmax(result_inc)
    pred_inc_score = result_inc[pred_inc_label].copy()


    result_mobile = exe.run(eval_program,
                            fetch_list=[out_mobilenet],
                            feed={input_layer.name:img,y_label.name:label})
    result_mobile = result_mobile[0][0]
    pred_mobile_label = np.argmax(result_mobile)
    pred_mobile_score = result_mobile[pred_mobile_label].copy()


    result_adv_resnet = exe.run(eval_program,
                            fetch_list=[out_adv_resnet],
                            feed={input_layer.name:img,y_label.name:label})
    result_adv_resnet = result_adv_resnet[0][0]
    pred_adv_resnet_label = np.argmax(result_adv_resnet)
    pred_adv_resnet_score = result_adv_resnet[pred_adv_resnet_label].copy()


    result_resnet_vd = exe.run(eval_program,
                            fetch_list=[out_resnet_vd],
                            feed={input_layer.name:img,y_label.name:label})
    result_resnet_vd = result_resnet_vd[0][0]
    pred_resnet_vd_label = np.argmax(result_resnet_vd)
    pred_resnet_vd_score = result_resnet_vd[pred_resnet_vd_label].copy()


    result_resnet_vd_32 = exe.run(eval_program,
                            fetch_list=[out_resnet_vd_32],
                            feed={input_layer.name:img,y_label.name:label})
    result_resnet_vd_32 = result_resnet_vd_32[0][0]
    pred_resnet_vd_32_label = np.argmax(result_resnet_vd_32)
    pred_resnet_vd_32_score = result_resnet_vd_32[pred_resnet_vd_32_label].copy()


    result_adv_densenet = exe.run(eval_program,
                            fetch_list=[out_adv_densenet],
                            feed={input_layer.name:img,y_label.name:label})
    result_adv_densenet = result_adv_densenet[0][0]
    pred_adv_densenet_label = np.argmax(result_adv_densenet)
    pred_adv_densenet_score = result_adv_densenet[pred_adv_densenet_label].copy()


    result_darknet = exe.run(eval_program,
                                   fetch_list=[out_darknet],
                                   feed={input_layer.name:img,y_label.name:label})
    result_darknet = result_darknet[0][0]
    pred_darknet_label = np.argmax(result_darknet)
    pred_darknet_score = result_darknet[pred_darknet_label].copy()


    result_efficient = exe.run(fluid.default_main_program(),
                             fetch_list=[out_efficient],
                             feed={input_layer.name:img,y_label.name:label})
    result_efficient = result_efficient[0][0]
    pred_efficient_label = np.argmax(result_efficient)
    pred_efficient_score = result_darknet[pred_efficient_label].copy()


    result_dpn = exe.run(fluid.default_main_program(),
                               fetch_list=[out_dpn],
                               feed={input_layer.name:img,y_label.name:label})
    result_dpn = result_dpn[0][0]
    pred_dpn_label = np.argmax(result_dpn)
    pred_dpn_score = result_dpn[pred_dpn_label].copy()

    result_suffle = exe.run(fluid.default_main_program(),
                         fetch_list=[out_suffle],
                         feed={input_layer.name: img, y_label.name: label})
    result_suffle = result_suffle[0][0]
    pred_suffle_label = np.argmax(result_suffle)
    pred_suffle_score = result_suffle[pred_suffle_label].copy()

    # result_hrnet = exe.run(fluid.default_main_program(),
    #                         fetch_list=[out_hrnet],
    #                         feed={input_layer.name: img, y_label.name: label})
    # result_hrnet = result_hrnet[0][0]
    # pred_hrnet_label = np.argmax(result_hrnet)
    # pred_hrnet_score = result_hrnet[pred_hrnet_label].copy()



    final_model_score = (pred_densenet_score + pred_resnet_score + pred_vgg_score + pred_inc_score +pred_mobile_score +pred_adv_resnet_score + pred_resnet_vd_score + pred_resnet_vd_32_score + pred_adv_densenet_score + \
                         pred_darknet_score + pred_efficient_score + pred_dpn_score + pred_suffle_score)/14
    return pred_mobile_label,pred_densenet_label, pred_resnet_label, pred_vgg_label,pred_inc_label,pred_adv_resnet_label,pred_resnet_vd_label,pred_resnet_vd_32_label,pred_adv_densenet_label,pred_darknet_label,pred_dpn_label,pred_efficient_label,pred_suffle_label,final_model_score


def get_original_file(filepath):
    with open(filepath, 'r') as cfile:
        full_lines = [line.strip() for line in cfile]
    cfile.close()
    original_files = []
    for line in full_lines:
        label, file_name = line.split()
        original_files.append([file_name, int(label)])
    return original_files


def Attack(o, label, step_size, epsilon=16.0 / 256,isTarget=False, use_gpu=True,epoch_num=13):
    """
    我们发现结合原图，水平翻转以及水平翻转之后在进行上下翻转的三种梯度最好
    这里注释掉的inference部分是为了能在攻击过程中动态输出标签信息，如果启用会极大降低运行速度
    """
    g = 0
    T = epoch_num
    adv = o
    chan_noise = 1
    print(label)
    input_label = np.expand_dims(np.expand_dims(label,axis=0),axis=0).astype('int64')
    # pred_mobile_label, pred_densenet_label, pred_resnet_label, pred_vgg_label, pred_inc_label, pred_adv_resnet_label, pred_resnet_vd_label, pred_resnet_vd_32_label, pred_adv_densenet_label, pred_darknet_label, pred_dpn_label, pred_efficient_label, final_model_score = inference(o)
    for i in range(T):
        flip_img = adv
        flip_img = np.flip(flip_img,3)
        flip_flip_img = np.flip(flip_img,2)
        flip_right_img = np.transpose(flip_flip_img,[0,1,3,2])
        print('epoch:', i)
        ###未翻转
        grad = exe.run(fluid.default_main_program(),
                       fetch_list=[gradients],
                       feed={input_layer.name: adv, y_label.name:input_label}
                       )
        grad = grad[0][0]
        grad = grad / np.mean(np.abs(grad))
        ###翻转
        grad_flip = exe.run(fluid.default_main_program(),
                            fetch_list=[gradients],
                            feed={input_layer.name: flip_img, y_label.name:input_label})
        grad_flip = grad_flip[0][0]
        grad_flip = np.expand_dims(grad_flip,axis=0)
        grad_flip = np.flip(grad_flip, 3)
        grad_flip = grad_flip / np.mean(np.abs(grad_flip))
        ###翻转两次
        grad_flip_flip = exe.run(fluid.default_main_program(),
                            fetch_list=[gradients],
                            feed={input_layer.name: flip_flip_img, y_label.name: input_label})
        grad_flip_flip = grad_flip_flip[0][0]
        grad_flip_flip = np.expand_dims(grad_flip_flip, axis=0)
        grad_flip_flip = np.flip(grad_flip_flip, 2)
        grad_flip_flip = np.flip(grad_flip_flip, 3)
        grad_flip_flip = grad_flip_flip / np.mean(np.abs(grad_flip_flip))
        # ###向右翻转
        # grad_flip_right = exe.run(fluid.default_main_program(),
        #                     fetch_list=[gradients],
        #                     feed={input_layer.name: flip_right_img, y_label.name: input_label})
        # grad_flip_right = grad_flip_right[0][0]
        # grad_flip_right = np.expand_dims(grad_flip_right, axis=0)
        # grad_flip_right = np.transpose(grad_flip_right,[0,1,3,2])
        # grad_flip_right = np.flip(grad_flip_right, 2)
        # grad_flip_right = np.flip(grad_flip_right, 3)
        # grad_flip_right = grad_flip_right / np.mean(np.abs(grad_flip_right))
        grad = grad + grad_flip/2 + grad_flip_flip/2
        g = 0.8 * g + grad
        adv = adv + np.clip(np.round(g), -20, 20) * step_size / T
        # pred_label_mobile, pred_label_densenet, pred_label_resnet, pred_label_vgg, pred_label_inc, pred_label_adv_resnet, pred_label_resnet_vd, pred_label_resnet_vd_32, pred_label_adv_densenet, pred_label_dark, pred_label_dpn, pred_label_eff, pred_score = \
        # inference(adv,label)
        # print(
        #     'mobilenet,resnet,densenet,vgg,inception,adv_resnet,resnet_vd,resnet_vd_32,adv_densenet,darknet,dpn,efficient.....:',
        #     pred_label_mobile, pred_label_resnet, pred_label_densenet, pred_label_vgg, pred_label_inc,
        #     pred_label_adv_resnet, pred_label_resnet_vd, pred_label_resnet_vd_32, pred_label_adv_densenet,
        #     pred_label_dark, pred_label_dpn, pred_label_eff)

    return adv


def attack_nontarget(img,label):
    step = 8.0 / 256.0
    eps = 32.0 / 256.0
    adv = img
    epoch_id = 0
    epoch_num = 10
    adv = Attack(o=adv, label=label, step_size=step, epsilon=eps,
               isTarget=False, use_gpu=use_gpu, epoch_num=epoch_num)
    adv_img = tensor2img(adv)
    return adv_img


def gen_adv():
    original_files = get_original_file(input_dir + val_list)
    for filename, label in original_files:
        img_path = input_dir + filename
        print("Image: {0} ".format(img_path))
        img=process_img(img_path)
        adv_img = attack_nontarget(img, label)
        image_name, image_ext = filename.split('.')
        save_adv_image(adv_img, output_dir+image_name+'.png')
        org_img = tensor2img(img)
    print("Attack completion")

