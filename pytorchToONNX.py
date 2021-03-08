from dl2.models import FASHIONSmall, GTSRBSmall
from PIL import Image
from torch import nn
import onnxruntime as ort
import numpy as np
import torch.onnx
import onnx
import io


def pytorchToONNX(model_path, name, model, img_size):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Input to the model
    batch_size = 1    # just a random number
    x = torch.randn(batch_size, 1, img_size, img_size, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,                   # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    name,                      # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    #dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
                    )


def onnx_infer(name, image):
    ort_session = ort.InferenceSession(name)
    outputs = ort_session.run(None, {'input': image.astype(np.float32)})
    print(outputs[0].argmax())



if __name__ == "__main__":
    '''
    image = np.array(Image.open('onnx/fashion_mnist_1.png').getdata())
    image = np.reshape(image, (1, 1, 28, 28))

    model_path = 'models/fashion_mnist/baseline/.pth'
    name = 'onnx/fashion_baseline.onnx'
    model = FASHIONSmall(dim=1)
    img_size = 28
    pytorchToONNX(model_path, name, model, img_size)
    onnx_infer(name, image)
    '''
    #########################################################################################
    
    image = np.array(Image.open('onnx/gtsrb_6.png').getdata())
    image = np.reshape(image, (1, 1, 48, 48))
    img_size = 48

    model_path = 'models/gtsrb/baseline/gtsrb_baseline_100Epochs_50Samples_TrueRobustness_83.pth'
    name = 'onnx/gtsrb_baseline.onnx'
    model = GTSRBSmall(dim=1)
    pytorchToONNX(model_path, name, model, img_size)
    onnx_infer(name, image)

    model_path = 'models/gtsrb/augmented/gtsrb_augmented_100Epochs_50Samples_0Pretrain_79.pth'
    name = 'onnx/gtsrb_data_aug_uniform.onnx'
    model = GTSRBSmall(dim=1)
    pytorchToONNX(model_path, name, model, img_size)
    onnx_infer(name, image)

    model_path = 'models/gtsrb/augmented_FGSM/gtsrb_augmented_FGSM_100Epochs_50Samples_TrueRobustness_86.pth'
    name = 'onnx/gtsrb_data_aug_FGSM.onnx'
    model = GTSRBSmall(dim=1)
    pytorchToONNX(model_path, name, model, img_size)
    onnx_infer(name, image)

    model_path = 'models/gtsrb/baseline/gtsrb_baseline_100Epochs_95_small_adversarial_FGSM.pth'
    name = 'onnx/gtsrb_adv_training.onnx'
    model = GTSRBSmall(dim=1)
    pytorchToONNX(model_path, name, model, img_size)
    onnx_infer(name, image)

    model_path = 'models/gtsrb/dl2/gtsrb_dl2_100Epochs_50Samples_TrueRobustness_83.pth'
    name = 'onnx/gtsrb_dl2_robustness.onnx'
    model = GTSRBSmall(dim=1)
    pytorchToONNX(model_path, name, model, img_size)
    onnx_infer(name, image)

    model_path = 'models/gtsrb/dl2/gtsrb_dl2_100Epochs_50Samples_FGSM_83.pth'
    name = 'onnx/gtsrb_dl2_FGSM.onnx'
    model = GTSRBSmall(dim=1)
    pytorchToONNX(model_path, name, model, img_size)
    onnx_infer(name, image)
