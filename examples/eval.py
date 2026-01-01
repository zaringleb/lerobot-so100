# %%
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.vla0_smol.configuration_vla0_smol import VLA0SmolConfig
from lerobot.policies.vla0_smol.modeling_vla0_smol import VLA0SmolPolicy

# %%
output_directory = Path("outputs/train/example_pusht")
output_directory.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda")

# %%
DATASET_PATH = "lerobot/pusht_image"

dataset_metadata = LeRobotDatasetMetadata(DATASET_PATH)
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

cfg = VLA0SmolConfig(input_features=input_features, output_features=output_features)

delta_timestamps = {
    "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
}

# We can then instantiate the dataset with these delta_timestamps configuration.
dataset = LeRobotDataset(DATASET_PATH, delta_timestamps=delta_timestamps)

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=8,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)
dl_iter = cycle(dataloader)

# %%
policy = VLA0SmolPolicy(cfg, dataset_stats=dataset_metadata.stats)
policy.train()
policy.to(device)

preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)

optimizer = torch.optim.Adam(policy.parameters(), lr=5e-5)

# %%
raw_batch = next(dl_iter)
batch = preprocessor(raw_batch)

# %%
for step in tqdm(range(50)):
    # batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    loss, _ = policy.forward(batch)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"step: {step} loss: {loss.item():.3f}")

# %% [markdown]
# 

# %%
decoded_actions = policy.model.generate_actions(batch)
decoded_actions = postprocessor(decoded_actions)

# %%
error: torch.tensor = torch.sqrt((decoded_actions.detach().cpu() - raw_batch["action"].detach().cpu()) ** 2)

print(f"RMSE {(error.mean(dim=1)).tolist()}")


