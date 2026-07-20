# engine/validate.py
import torch
import torch.nn.functional as F
from data.utils import instance_normalize, reverse_instance_normalize


@torch.no_grad()
def validate_model(
    encoder_online,
    quantizer_online,
    predictor,
    decoder,
    encoder_target,
    quantizer_target,
    val_loader,
    device,
    pred_temp,
):
    """
    Run validation and return averaged losses.
    """
    encoder_online.eval()
    quantizer_online.eval()
    predictor.eval()
    decoder.eval()
    encoder_target.eval()
    quantizer_target.eval()

    total_loss = 0.0
    total_kl = 0.0
    total_recon = 0.0

    for x_past, x_future in val_loader:
        x_past_norm, mean, std = instance_normalize(x_past)
        x_future_norm, _, _ = instance_normalize(x_future)

        x_past_norm = x_past_norm.to(device)
        x_future_norm = x_future_norm.to(device)
        mean, std = mean.to(device), std.to(device)

        h_past = encoder_online(x_past_norm)
        p_past, z_q_past = quantizer_online(h_past)

        x_recon = decoder(z_q_past)
        x_recon = reverse_instance_normalize(x_recon, mean, std)

        x_target = x_past.permute(0, 1, 3, 2).to(device)
        loss_recon = F.mse_loss(x_recon, x_target)

        logits_pred, _ = predictor(p_past)
        logp_pred = F.log_softmax(logits_pred / pred_temp, dim=-1)

        h_future = encoder_target(x_future_norm)
        p_future, _ = quantizer_target(h_future)

        loss_kl = F.kl_div(logp_pred, p_future, reduction="batchmean")

        total_loss += (loss_kl + 0.01 * loss_recon).item()
        total_kl += loss_kl.item()
        total_recon += loss_recon.item()

    n = len(val_loader)
    return total_loss / n, total_kl / n, total_recon / n
