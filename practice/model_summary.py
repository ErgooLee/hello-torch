import torch
import torch.nn as nn


def _format_param_calc(m: nn.Module) -> tuple[str, int]:
    """计算层参数量并生成直观的计算表达式 (公式 + 结果)。"""
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    if total_params == 0:
        return "", 0

    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        cin = m.in_channels
        cout = m.out_channels
        k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,) * (len(m.weight.shape) - 2)
        k_str = "×".join(map(str, k))
        has_bias = m.bias is not None and m.bias.requires_grad
        w_p = cout * (cin // m.groups) * torch.prod(torch.tensor(k)).item()
        b_p = cout if has_bias else 0
        b_str = f" + {b_p}" if has_bias else ""
        return f"{cin}×{cout}×{k_str}{b_str} = {total_params:,}", total_params

    elif isinstance(m, nn.Linear):
        fin = m.in_features
        fout = m.out_features
        has_bias = m.bias is not None and m.bias.requires_grad
        b_p = fout if has_bias else 0
        b_str = f" + {b_p}" if has_bias else ""
        return f"{fin}×{fout}{b_str} = {total_params:,}", total_params

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
        if getattr(m, "affine", True) or getattr(m, "elementwise_affine", True):
            nf = getattr(m, "num_features", None) or getattr(m, "normalized_shape", None)
            if isinstance(nf, tuple):
                nf = torch.prod(torch.tensor(nf)).item()
            return f"2×{nf}(γ,β) = {total_params:,}", total_params

    elif isinstance(m, nn.Embedding):
        num_emb = m.num_embeddings
        emb_dim = m.embedding_dim
        return f"{num_emb}×{emb_dim} = {total_params:,}", total_params

    return f"{total_params:,}", total_params


def print_model_flow(model: nn.Module, input_size=(1, 1, 28, 28)):
    """自顶向下打印模型数据流管道，包含输入输出shape及卷积/池化/全连接层参数计算过程。"""
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    hooks = []
    flow_info = []

    def make_hook(layer):
        def hook(m, inp, out):
            in_shape = list(inp[0].shape)
            out_shape = list(out.shape)

            k = getattr(m, "kernel_size", None)
            s = getattr(m, "stride", None)
            p = getattr(m, "padding", None)

            detail = []
            if k is not None:
                detail.append(f"k={k}")
            if s is not None and s != 1 and s != (1, 1):
                detail.append(f"s={s}")
            if p is not None and p != 0 and p != (0, 0):
                detail.append(f"p={p}")

            param_expr, params = _format_param_calc(m)
            if param_expr:
                detail.append(f"params: [{param_expr}]")

            detail_str = f" ({', '.join(detail)})" if detail else ""
            flow_info.append({
                "name": f"{layer.__class__.__name__}{detail_str}",
                "in": in_shape,
                "out": out_shape,
                "params": params
            })
        return hook

    # 为所有叶子子模块注册 hook
    for name, layer in model.named_modules():
        if len(list(layer.children())) == 0 and layer != model:
            hooks.append(layer.register_forward_hook(make_hook(layer)))

    model_training = model.training
    model.eval()
    with torch.no_grad():
        dummy_x = torch.zeros(input_size, device=device)
        model(dummy_x)
    model.train(model_training)

    for h in hooks:
        h.remove()

    total_params = sum(item["params"] for item in flow_info)

    # 格式化输出流图
    print("\n" + "=" * 80)
    print(f" 🚀 {model.__class__.__name__} Data Flow Pipeline (Input: {list(input_size)})")
    print("=" * 80)

    for i, step in enumerate(flow_info):
        if i == 0:
            print(f"   [Input Tensor] ──▶ {step['in']}")
        print("         │")
        print("         ▼")
        print(f" ┌── [Layer {i+1:02d}: {step['name']}]")
        print(f" └──▶ Output: {step['out']}")

    print("=" * 80)
    print(f" Total Trainable Params: {total_params:,}")
    print("=" * 80 + "\n")
