from environs import Env

env = Env()
env.read_env()  # read .env file, if it exists

class Config:
    tokenizers_config_dir: str = env.str("TOKENIZERS_CONFIG_DIR")
    hf_token: str = env.str("HF_TOKEN")
    horde_url: str = env.str("HORDE_URL", default="aihorde.net")
    horde_proxy_passkey: str = env.str("HORDE_PROXY_PASSKEY")