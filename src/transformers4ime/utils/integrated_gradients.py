import numpy as np
import torch


def scale_input(q_emb, padding_embedding, num_batches=10):
    """ Create scaled versions of input and stack along batch dimension
    q_emb shape = (q_length, emb_dim)
    """
    num_points = num_batches
    scale = 1.0 / num_points
    delta = (q_emb - padding_embedding.unsqueeze(1)) * scale
    batch = torch.cat([torch.add(padding_embedding, delta * i) for i in range(num_points)], dim=0)
    return batch, delta.squeeze(0)


def compute_attributions(sentence1, sentence2, model, padding_embedding, tokenizer, target_label_idx=None):
    batch_size = 100
    total_grads = 0
    inputs = tokenizer.encode_plus(sentence1, sentence2, return_tensors='pt', add_special_tokens=True)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    # diff = 0

    if target_label_idx is None:
        with torch.no_grad():
            prediction = model(input_ids=input_ids,
                               token_type_ids=token_type_ids)
            target_label_idx = prediction[0].max(dim=1)[1].item()

    emb = model.bert.embeddings.word_embeddings(input_ids)

    with torch.autograd.set_grad_enabled(True):
        scaled_q_emb, delta = scale_input(emb, padding_embedding, batch_size)

        scaled_answer = model(input_ids=None,
                              token_type_ids=token_type_ids,
                              inputs_embeds=scaled_q_emb)

        output = torch.softmax(scaled_answer[0], dim=-1)

        gradients = torch.autograd.grad(torch.unbind(output[:, target_label_idx]), scaled_q_emb)
        # diff -= output[0, target_label_idx]
        # baseline_softmax = output[0, :]
        # diff += output[-1, target_label_idx]

    total_grads += torch.sum(gradients[0], dim=0)

    attributions = torch.sum(total_grads * delta, dim=1)

    # area = torch.sum(attributions, dim=0)

    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    return tokens, attributions, target_label_idx


def calculate_outputs_and_gradients(inputs, model, target_label_idx, cuda=False):
    # do the pre-processing
    predict_idx = None
    gradients = []
    for input in inputs:
        input = pre_processing(input, cuda)
        output = model(input)
        output = F.softmax(output, dim=1)
        if target_label_idx is None:
            target_label_idx = torch.argmax(output, 1).item()
        index = np.ones((output.size()[0], 1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        if cuda:
            index = index.cuda()
        output = output.gather(1, index)
        # clear grad
        model.zero_grad()
        output.backward()
        gradient = input.grad.detach().cpu().numpy()[0]
        gradients.append(gradient)
    gradients = np.array(gradients)
    return gradients, target_label_idx


def pre_processing(obs, cuda):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    obs = obs / 255
    obs = (obs - mean) / std
    obs = np.transpose(obs, (2, 0, 1))
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)
    if cuda:
        torch_device = torch.device('cuda:0')
    else:
        torch_device = torch.device('cpu')
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device, requires_grad=True)
    return obs_tensor


# generate the entire images
def generate_entrie_images(img_origin, img_grad, img_grad_overlay, img_integrad, img_integrad_overlay):
    blank = np.ones((img_grad.shape[0], 10, 3), dtype=np.uint8) * 255
    blank_hor = np.ones((10, 20 + img_grad.shape[0] * 3, 3), dtype=np.uint8) * 255
    upper = np.concatenate([img_origin[:, :, (2, 1, 0)], blank, img_grad_overlay, blank, img_grad], 1)
    down = np.concatenate([img_origin[:, :, (2, 1, 0)], blank, img_integrad_overlay, blank, img_integrad], 1)
    total = np.concatenate([upper, blank_hor, down], 0)
    total = cv2.resize(total, (550, 364))

    return total


# integrated gradients
def integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, baseline, steps=50, cuda=False):
    if baseline is None:
        baseline = 0 * inputs
        # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, cuda)
    avg_grads = np.average(grads[:-1], axis=0)
    avg_grads = np.transpose(avg_grads, (1, 2, 0))
    integrated_grad = (inputs - baseline) * avg_grads
    return integrated_grad


def random_baseline_integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, steps,
                                         num_random_trials, cuda):
    all_intgrads = []
    for i in range(num_random_trials):
        integrated_grad = integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, \
                                               baseline=255.0 * np.random.random(inputs.shape), steps=steps, cuda=cuda)
        all_intgrads.append(integrated_grad)
        print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads
