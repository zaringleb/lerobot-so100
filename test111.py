from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("southfreebird/lerobot_test_100", tag="v3.0", repo_type="dataset")
