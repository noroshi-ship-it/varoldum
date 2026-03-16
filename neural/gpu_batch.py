
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

DEVICE = None


def get_device():
    global DEVICE
    if DEVICE is not None:
        return DEVICE
    if HAS_TORCH and torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"[GPU] Using {torch.cuda.get_device_name(0)}")
    elif HAS_TORCH:
        DEVICE = torch.device('cpu')
        print("[GPU] CUDA not available, using CPU torch")
    else:
        DEVICE = None
        print("[GPU] PyTorch not available, numpy fallback")
    return DEVICE


def _arch_hash(agent):
    arch = agent.brain.arch
    return (
        tuple(arch["layer_sizes"]),
        arch["gru_size"],
        arch["bottleneck_size"],
        agent.brain.context_dim,
    )


def _batch_gpu_forward(group, contexts, device):
    n = len(group)
    ref = group[0].brain

    enc_layers = list(ref.encoder_layers) + [ref.bottleneck_layer]
    raw_dim = ref.raw_input_dim
    bn_size = ref.bottleneck_size
    ctx_dim = ref.context_dim
    gru_hidden = ref.gru_size
    action_dim = ref.action_dim
    gru_in = bn_size + ctx_dim

    raw_np = np.zeros((n, raw_dim), dtype=np.float32)
    ctx_np = np.zeros((n, ctx_dim), dtype=np.float32)
    h_np = np.zeros((n, gru_hidden), dtype=np.float32)

    for i, a in enumerate(group):
        ri = a._raw_input
        raw_np[i, :len(ri)] = ri[:raw_dim]
        c = contexts[a.id]
        ctx_np[i, :len(c)] = c[:ctx_dim]
        h_np[i] = a.brain.gru.h

    raw_t = torch.from_numpy(raw_np).to(device)
    ctx_t = torch.from_numpy(ctx_np).to(device)
    h_t = torch.from_numpy(h_np).to(device)

    h_enc = raw_t
    for li, layer_ref in enumerate(enc_layers):
        in_d, out_d = layer_ref.W.shape
        W_np = np.zeros((n, in_d, out_d), dtype=np.float32)
        b_np = np.zeros((n, out_d), dtype=np.float32)
        for i, a in enumerate(group):
            a_layers = list(a.brain.encoder_layers) + [a.brain.bottleneck_layer]
            W_np[i] = a_layers[li].W
            b_np[i] = a_layers[li].b
        W_t = torch.from_numpy(W_np).to(device)
        b_t = torch.from_numpy(b_np).to(device)
        h_enc = torch.bmm(h_enc.unsqueeze(1), W_t).squeeze(1) + b_t
        h_enc = torch.tanh(h_enc)

    concepts_t = h_enc

    policy_in = torch.cat([concepts_t, ctx_t], dim=1)

    Wr_np = np.zeros((n, gru_in + gru_hidden, gru_hidden), dtype=np.float32)
    br_np = np.zeros((n, gru_hidden), dtype=np.float32)
    Wz_np = np.zeros((n, gru_in + gru_hidden, gru_hidden), dtype=np.float32)
    bz_np = np.zeros((n, gru_hidden), dtype=np.float32)
    Wh_np = np.zeros((n, gru_in + gru_hidden, gru_hidden), dtype=np.float32)
    bh_np = np.zeros((n, gru_hidden), dtype=np.float32)

    for i, a in enumerate(group):
        gru = a.brain.gru
        Wr_np[i] = gru.W_r
        br_np[i] = gru.b_r
        Wz_np[i] = gru.W_z
        bz_np[i] = gru.b_z
        Wh_np[i] = gru.W_h
        bh_np[i] = gru.b_h

    Wr_t = torch.from_numpy(Wr_np).to(device)
    br_t = torch.from_numpy(br_np).to(device)
    Wz_t = torch.from_numpy(Wz_np).to(device)
    bz_t = torch.from_numpy(bz_np).to(device)
    Wh_t = torch.from_numpy(Wh_np).to(device)
    bh_t = torch.from_numpy(bh_np).to(device)

    combined = torch.cat([policy_in, h_t], dim=1)
    r = torch.sigmoid(torch.bmm(combined.unsqueeze(1), Wr_t).squeeze(1) + br_t)
    z = torch.sigmoid(torch.bmm(combined.unsqueeze(1), Wz_t).squeeze(1) + bz_t)
    combined_r = torch.cat([policy_in, r * h_t], dim=1)
    h_cand = torch.tanh(torch.bmm(combined_r.unsqueeze(1), Wh_t).squeeze(1) + bh_t)
    h_new = (1 - z) * h_t + z * h_cand

    Wo_np = np.zeros((n, gru_hidden, action_dim), dtype=np.float32)
    bo_np = np.zeros((n, action_dim), dtype=np.float32)
    Wv_np = np.zeros((n, gru_hidden, 1), dtype=np.float32)
    bv_np = np.zeros((n, 1), dtype=np.float32)

    for i, a in enumerate(group):
        Wo_np[i] = a.brain.output_layer.W
        bo_np[i] = a.brain.output_layer.b
        Wv_np[i] = a.brain.value_layer.W
        bv_np[i] = a.brain.value_layer.b

    Wo_t = torch.from_numpy(Wo_np).to(device)
    bo_t = torch.from_numpy(bo_np).to(device)
    Wv_t = torch.from_numpy(Wv_np).to(device)
    bv_t = torch.from_numpy(bv_np).to(device)

    actions_t = torch.tanh(torch.bmm(h_new.unsqueeze(1), Wo_t).squeeze(1) + bo_t)
    values_t = torch.bmm(h_new.unsqueeze(1), Wv_t).squeeze(1) + bv_t

    concepts_out = concepts_t.cpu().numpy()
    actions_out = actions_t.cpu().numpy()
    values_out = values_t.cpu().numpy()
    h_out = h_new.cpu().numpy()

    results = {}
    for i, a in enumerate(group):
        results[a.id] = (concepts_out[i], actions_out[i], float(values_out[i, 0]), h_out[i])
    return results


def batch_think(agents, contexts):
    device = get_device()
    if device is None or not HAS_TORCH:
        return None

    groups = {}
    for agent in agents:
        if not agent.is_alive:
            continue
        key = _arch_hash(agent)
        if key not in groups:
            groups[key] = []
        groups[key].append(agent)

    results = {}
    for arch_key, group in groups.items():
        if len(group) == 0:
            continue
        group_results = _batch_gpu_forward(group, contexts, device)
        results.update(group_results)
    return results


def batch_encode(agents) -> dict:
    device = get_device()
    if device is None or not HAS_TORCH:
        results = {}
        for agent in agents:
            concepts = agent.brain.encode(agent._raw_input)
            results[agent.id] = concepts
        return results

    groups = {}
    for agent in agents:
        key = _arch_hash(agent)
        if key not in groups:
            groups[key] = []
        groups[key].append(agent)

    results = {}
    for arch_key, group in groups.items():
        n = len(group)
        ref = group[0].brain
        enc_layers = list(ref.encoder_layers) + [ref.bottleneck_layer]
        raw_dim = ref.raw_input_dim

        raw_np = np.zeros((n, raw_dim), dtype=np.float32)
        for i, a in enumerate(group):
            ri = a._raw_input
            raw_np[i, :len(ri)] = ri[:raw_dim]
        h = torch.from_numpy(raw_np).to(device)

        for li, layer_ref in enumerate(enc_layers):
            in_d, out_d = layer_ref.W.shape
            W_np = np.zeros((n, in_d, out_d), dtype=np.float32)
            b_np = np.zeros((n, out_d), dtype=np.float32)
            for i, a in enumerate(group):
                a_layers = list(a.brain.encoder_layers) + [a.brain.bottleneck_layer]
                W_np[i] = a_layers[li].W
                b_np[i] = a_layers[li].b
            W_t = torch.from_numpy(W_np).to(device)
            b_t = torch.from_numpy(b_np).to(device)
            h = torch.bmm(h.unsqueeze(1), W_t).squeeze(1) + b_t
            h = torch.tanh(h)

        concepts_np = h.cpu().numpy()
        for i, a in enumerate(group):
            results[a.id] = concepts_np[i]
    return results
