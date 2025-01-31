import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import gen_batches
from tqdm import tqdm


def inference(test_data, test_label, model_name, input_logit_index, target_label, batch_size, num_class, input_shape,
              watermark=None, watermark_indices=None, verbose=False):
    # the input of last layer
    select_input_logits = np.zeros((test_label.shape[0], input_shape))
    select_output_logits = np.zeros((test_label.shape[0], num_class))
    predictions = np.zeros(test_label.shape)

    num_batches = list(gen_batches(test_data.shape[0], batch_size))

    interpreter = tf.lite.Interpreter(
        model_path='protect_models/' + model_name)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # model specific input and output
    interpreter.resize_tensor_input(input_details[0]['index'],
                                    [batch_size, test_data.shape[1], test_data.shape[2], test_data.shape[3]])
    interpreter.resize_tensor_input(output_details[0]['index'],
                                    [batch_size, test_data.shape[0], num_class])

    interpreter.allocate_tensors()

    # batch inference
    for batch in tqdm(num_batches):
        # input_details[0]['index'] = the index which accepts the input
        interpreter.set_tensor(input_details[0]['index'], test_data[batch, :])

        # run the inference
        interpreter.invoke()

        # output_details[0]['index'] = the index which provides the input
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # the index of input of last layer
        select_input_logits[batch] = interpreter.get_tensor(input_logit_index)
        select_output_logits[batch] = interpreter.get_tensor(input_logit_index + 1)
        predictions[batch] = np.argmax(output_data, axis=1)

    target_indices = np.where(test_label == target_label)[0]

    if watermark and verbose:
        cle_accuracy = 0
        wsr = 0

        cle_accuracy = round(
            accuracy_score(np.delete(test_label, watermark_indices), np.delete(predictions, watermark_indices)), 4)
        wsr = round(accuracy_score(test_label[watermark_indices], predictions[watermark_indices]), 4)

        print("Overall accuracy:", cle_accuracy)

        print("Non-target label accuracy:",
              round(accuracy_score(np.delete(test_label, np.concatenate((target_indices, watermark_indices))),
                                   np.delete(predictions, np.concatenate((target_indices, watermark_indices)))), 4))
        print("Target label w/o trigger accuracy:", round(accuracy_score(test_label[target_indices],
                                                                         predictions[target_indices]), 4))
        print("Target label w/ trigger watermark success rate:", wsr)

        return select_input_logits, select_output_logits, target_indices, cle_accuracy, wsr


def inference_synthesis(test_data, test_label, model_name, batch_size, num_class):
    predictions = np.zeros(test_label.shape)
    num_batches = list(gen_batches(test_data.shape[0], batch_size))

    interpreter = tf.lite.Interpreter(
        model_path='protect_models/' + model_name)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # model specific input and output
    interpreter.resize_tensor_input(input_details[0]['index'],
                                    [batch_size, test_data.shape[1], test_data.shape[2], test_data.shape[3]])
    interpreter.resize_tensor_input(output_details[0]['index'],
                                    [batch_size, test_data.shape[0], num_class])

    interpreter.allocate_tensors()

    # batch inference
    for batch in tqdm(num_batches):
        # input_details[0]['index'] = the index which accepts the input
        interpreter.set_tensor(input_details[0]['index'], test_data[batch, :])

        # run the inference
        interpreter.invoke()

        # output_details[0]['index'] = the index which provides the input
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions[batch] = np.argmax(output_data, axis=1)

    print("Overall accuracy:", round(accuracy_score(test_label, predictions), 4))

    return predictions
