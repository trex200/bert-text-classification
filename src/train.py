"""Train Bert CLS classifier on SST-2."""

import torch
import torch.nn as nn
from tqdm import tqdm


def train(model, train_loader, device="cuda", epochs=10, lr=2e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for input_ids, labels in pbar:
            input_ids, labels = input_ids.to(device), labels.to(device)
            mask = (input_ids != 0).unsqueeze(1).unsqueeze(2).float().to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(input_ids, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / len(train_loader)
        print(f"→ Epoch {epoch+1} Average Loss: {avg:.4f}")

    return model
