"""
Stage 3 driver — trains both networks and extracts 256-D embeddings
on the gallery + probe sets.

For each method, this writes:
    outputs/features/<name>_eval.npz   : g_emb, g_y, p_emb, p_y,
                                         class_sims (max-cosine per class),
                                         classes, ranks, cmc,
                                         genuine, imposter
    outputs/results/training_history.json
    outputs/checkpoints/<name>_best.pth
"""

import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg
from stage_3_feature_extraction.iris_dataset         import (
    IrisDataset, make_gallery_probe_split,
)
from stage_3_feature_extraction.irisdeepnet_resnet18 import IrisDeepNetResNet18
from stage_3_feature_extraction.dbn_style_cnn        import DBNStyleCNN


torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)


# ----------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimiser=None):
    is_train = optimiser is not None
    model.train(is_train)
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)
        out  = model(x)
        loss = criterion(out, y)
        if is_train:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total


def train_one_network(model, train_ds, probe_ds, name):
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH,
                              shuffle=True,  num_workers=0)
    probe_loader = DataLoader(probe_ds, batch_size=cfg.BATCH,
                              shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH)
    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=cfg.EPOCHS)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}
    best = 0.0
    print(f"\n--- Training {name} ---")
    print(f"  params : {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, optimiser)
        with torch.no_grad():
            vl, va = run_epoch(model, probe_loader, criterion, None)
        scheduler.step()
        for k, v in zip(history, (tl, vl, ta, va)):
            history[k].append(v)
        if va > best:
            best = va
            torch.save(model.state_dict(),
                       cfg.OUT_CKPT / f"{name}_best.pth")
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.EPOCHS:
            print(f"  ep {epoch:3d}/{cfg.EPOCHS}  "
                  f"trL={tl:.3f} trA={ta*100:5.2f}%  "
                  f"prL={vl:.3f} prA={va*100:5.2f}%  "
                  f"({time.time() - t0:.1f}s)")
    print(f"  best probe accuracy : {best * 100:.2f}%")
    return history


def extract_embeddings(model, ds):
    loader = DataLoader(ds, batch_size=cfg.BATCH, shuffle=False)
    model.eval()
    embs, lbls = [], []
    with torch.no_grad():
        for x, y in loader:
            _, e = model(x.to(cfg.DEVICE), return_embedding=True)
            embs.append(e.cpu().numpy())
            lbls.append(y.numpy())
    return np.concatenate(embs, 0), np.concatenate(lbls, 0)


def nearest_neighbour_max(g_emb, g_y, p_emb, p_y, K=None):
    """Class score = MAX cosine to any gallery sample of that class."""
    K = K or cfg.CMC_K
    sims = p_emb @ g_emb.T
    classes = np.unique(g_y)
    class_sims = np.full((sims.shape[0], len(classes)), -1.0)
    for i, c in enumerate(classes):
        class_sims[:, i] = sims[:, g_y == c].max(1)
    rank_order = np.argsort(-class_sims, axis=1)
    ranks = np.array([np.where(rank_order[i] == gt)[0][0] + 1
                      for i, gt in enumerate(p_y)])
    cmc = np.array([(ranks <= k).mean() for k in range(1, K + 1)]) * 100
    return ranks, cmc, class_sims, classes


def split_genuine_imposter(class_sims, p_y, classes):
    """Return (genuine, imposter) cosine scores."""
    same = np.zeros_like(class_sims, dtype=bool)
    for i in range(class_sims.shape[0]):
        same[i, p_y[i]] = True
    gen, imp = class_sims[same], class_sims[~same]
    if len(imp) > cfg.IMP_SUBSAMPLE:
        rng = np.random.RandomState(cfg.SEED)
        imp = imp[rng.choice(len(imp), cfg.IMP_SUBSAMPLE, replace=False)]
    return gen.astype(np.float32), imp.astype(np.float32)


# ----------------------------------------------------------------------
def main():
    print(f"[Stage 3] Feature extraction — device: {cfg.DEVICE}")
    print("Loading templates ...")
    templates = np.load(cfg.NORM_NPZ)["templates"]
    labels    = np.load(cfg.LBL_NPZ)
    subjects  = labels["subjects"]
    eyes      = labels["eyes"]

    # Right eye -> IrisDeepNet, Left eye -> DBN-style
    (XR_g, yR_g), (XR_p, yR_p), classes_R = make_gallery_probe_split(
        templates, subjects, eyes, eye_filter=1)
    (XL_g, yL_g), (XL_p, yL_p), classes_L = make_gallery_probe_split(
        templates, subjects, eyes, eye_filter=0)
    print(f"  RIGHT  gallery={len(XR_g):5d}  probe={len(XR_p):5d}  classes={len(classes_R)}")
    print(f"  LEFT   gallery={len(XL_g):5d}  probe={len(XL_p):5d}  classes={len(classes_L)}")

    # ----- IrisDeepNet (ResNet-18) on right-eye -----
    cnn        = IrisDeepNetResNet18(n_classes=len(classes_R)).to(cfg.DEVICE)
    hist_cnn   = train_one_network(
        cnn,
        IrisDataset(XR_g, yR_g, augment=True),
        IrisDataset(XR_p, yR_p, augment=False),
        name="cnn",
    )

    # ----- DBN-style on left-eye -----
    dbn        = DBNStyleCNN(n_classes=len(classes_L)).to(cfg.DEVICE)
    hist_dbn   = train_one_network(
        dbn,
        IrisDataset(XL_g, yL_g, augment=True),
        IrisDataset(XL_p, yL_p, augment=False),
        name="dbn",
    )

    (cfg.OUT_RES / "training_history.json").write_text(
        json.dumps({"cnn": hist_cnn, "dbn": hist_dbn}, indent=1))

    # Re-load best weights and extract embeddings (no augmentation)
    cnn.load_state_dict(torch.load(cfg.OUT_CKPT / "cnn_best.pth"))
    dbn.load_state_dict(torch.load(cfg.OUT_CKPT / "dbn_best.pth"))

    for net, name, Xg, yg, Xp, yp in [
        (cnn, "cnn", XR_g, yR_g, XR_p, yR_p),
        (dbn, "dbn", XL_g, yL_g, XL_p, yL_p),
    ]:
        g_emb, g_y = extract_embeddings(net, IrisDataset(Xg, yg, augment=False))
        p_emb, p_y = extract_embeddings(net, IrisDataset(Xp, yp, augment=False))
        ranks, cmc, csims, classes = nearest_neighbour_max(g_emb, g_y, p_emb, p_y)
        gen, imp = split_genuine_imposter(csims, p_y, classes)
        print(f"  {name.upper():4s}  rank-1={cmc[0]:5.2f}%  rank-5={cmc[4]:5.2f}%  "
              f"rank-20={cmc[19]:5.2f}%  gen-mean={gen.mean():.3f}  imp-mean={imp.mean():.3f}")
        np.savez(cfg.OUT_FEAT / f"{name}_eval.npz",
                 g_emb=g_emb, g_y=g_y, p_emb=p_emb, p_y=p_y,
                 class_sims=csims, classes=classes,
                 ranks=ranks, cmc=cmc,
                 genuine=gen, imposter=imp)

    print(f"\n[Stage 3] Done. Features written to {cfg.OUT_FEAT}")


if __name__ == "__main__":
    main()
