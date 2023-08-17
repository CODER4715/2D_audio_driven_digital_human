import netron
import numpy as np
import onnxruntime
import torch.onnx
import torchvision
from models import Wav2Lip
import onnx
from onnx import load_model, save_model
from onnxmltools.utils import float16_converter
device = 'cuda'


def zhuanma(modelname):
    checkpoint_path = './saved_checkpoints/' + modelname + '.pth'

    def _load(checkpoint_path):
        if device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    # Standard ImageNet input - 3 channels, 224x224,
    # values don't matter as we care about network structure.
    # But they can also be real inputs.
    mel = torch.randn(128, 1, 80, 16)
    img = torch.randn(128, 6, 96, 96)
    # Obtain your model, it can be also constructed in your script explicitly
    model = Wav2Lip()
    checkpoint = _load(checkpoint_path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    # Invoke export
    # torch.onnx.export(model, (dummy_input,input2), "wav2lip.onnx")
    # Export the model

    dynamic_axes = {'mel': {0: 'pillar_num'},
                    'img': {0: 'pillar_num'},

                    'output': {0: 'batch_size'}
                    }


    torch.onnx.export(model,  # model being run
                      (mel, img),  # model input (or a tuple for multiple inputs)
                      "weights/" + modelname + '.onnx',  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['mel', 'img'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      dynamic_axes=dynamic_axes
					  )


def test(modelname):
    onnx_model = onnx.load("weights/" + modelname + '.onnx')
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    ort_sess = onnxruntime.InferenceSession("weights/" + modelname + '.onnx', providers=['CPUExecutionProvider'])

    # Check that the IR is well formed
    # onnx.checker.check_model(model)
    input1 = torch.FloatTensor(torch.randn(128, 1, 80, 16))
    input2 = torch.FloatTensor(torch.randn(128, 6, 96, 96))

    # input1 =np.random.randn(128, 1, 80, 16)
    # input2 = np.random.randn(128, 6, 96, 96)
    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model.graph))

    # input_name1 = ort_session.get_inputs()[0].name
    # input_name2 = ort_session.get_inputs()[1].name
    # output_name = ort_session.get_outputs()[0].name
    # print('input_name1:', input_name1)
    # print('input_name2:', input_name2)
    # print('output_name:', output_name)
    # pred = ort_session.run([], ({input_name1: input1}, {input_name2: input2}))

    outputs = ort_sess.run(None, {'mel': input1.numpy(),
                                  'img': input2.numpy()})
    # print(input1.numpy())

    print(outputs[0].shape)

def to_fp16(modelname):
    onnx_model = load_model("weights/" + modelname + '.onnx')
    trans_model = float16_converter.convert_float_to_float16(onnx_model, keep_io_types=True)
    save_model(trans_model, "weights/" + modelname + "_fp16.onnx")

if __name__ == '__main__':
    modelname = 'wav2lip_gan'
    zhuanma(modelname)
    test(modelname)
    to_fp16(modelname)
    print(modelname+"convert done!")

