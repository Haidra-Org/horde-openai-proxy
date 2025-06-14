import threading
import time
import requests
from typing import List, Optional, Dict, Any
from .consts import VERSION
from loguru import logger
class Worker:
  def __init__(
    self,
    requests_fulfilled: int,
    kudos_rewards: int,
    kudos_details: Dict[str, Any],
    performance: str,
    threads: int,
    uptime: int,
    maintenance_mode: bool,
    info: str,
    nsfw: bool,
    trusted: bool,
    flagged: bool,
    uncompleted_jobs: int,
    models: List[str],
    team: Dict[str, Optional[Any]],
    bridge_agent: str,
    max_length: int,
    max_context_length: int,
    type: str,
    name: str,
    id: str,
    online: bool,
  ):
    self.requests_fulfilled = requests_fulfilled
    self.kudos_rewards = kudos_rewards
    self.kudos_details = kudos_details
    self.performance = performance
    self.threads = threads
    self.uptime = uptime
    self.maintenance_mode = maintenance_mode
    self.info = info
    self.nsfw = nsfw
    self.trusted = trusted
    self.flagged = flagged
    self.uncompleted_jobs = uncompleted_jobs
    self.models = models
    self.team = team
    self.bridge_agent = bridge_agent
    self.max_length = max_length
    self.max_context_length = max_context_length
    self.type = type
    self.name = name
    self.id = id
    self.online = online

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "Worker":
    return cls(
      requests_fulfilled=data.get("requests_fulfilled"),
      kudos_rewards=data.get("kudos_rewards"),
      kudos_details=data.get("kudos_details"),
      performance=data.get("performance"),
      threads=data.get("threads"),
      uptime=data.get("uptime"),
      maintenance_mode=data.get("maintenance_mode"),
      info=data.get("info"),
      nsfw=data.get("nsfw"),
      trusted=data.get("trusted"),
      flagged=data.get("flagged"),
      uncompleted_jobs=data.get("uncompleted_jobs"),
      models=data.get("models"),
      team=data.get("team"),
      bridge_agent=data.get("bridge_agent"),
      max_length=data.get("max_length"),
      max_context_length=data.get("max_context_length"),
      type=data.get("type"),
      name=data.get("name"),
      id=data.get("id"),
      online=data.get("online"),
    )


class HordeWorkers():
  def __init__(self):
    self._stop_event = threading.Event()
    self.workers: List[Worker] = []
    self.thread = threading.Thread(target=self.start_worker_retrieval_thread, daemon=True)
    self.thread.start()

  def start_worker_retrieval_thread(self):
    while not self._stop_event.is_set():
      self.fetch_workers()
      time.sleep(60)

  def fetch_workers(self):
    url = "https://aihorde.net/api/v2/workers?type=text"
    headers = {
      "accept": "application/json",
      "Client-Agent": f"horde-openai-proxy:{VERSION}:db0"
    }
    try:
      logger.info("Fetching workers from the AI Horde...")
      response = requests.get(url, headers=headers, timeout=30)
      response.raise_for_status()
      workers_data = response.json()
      self.workers = [Worker.from_dict(w) for w in workers_data]
      logger.info(f"Fetched {len(self.workers)} workers from the AI Horde.")
    except Exception as e:
      # Optionally log the error
      logger.error(f"Failed to fetch workers: {e}")

  def stop(self):
    self._stop_event.set()
    
  def get_max_tokens_for_model(self, model: str) -> int:
    """
    Get the maximum tokens for a specific model.
    :param model: Model name
    :return: Maximum tokens or None if not found
    """
    highest_max_length = 0
    for worker in self.workers:
      if model in worker.models:
        if worker.max_length > highest_max_length:
          highest_max_length = worker.max_length
    if highest_max_length > 0:
      return highest_max_length
    return 120

  def get_max_context_length_for_model(self, model: str) -> int:
    """
    Get the maximum context length for a specific model.
    :param model: Model name
    :return: Maximum context length or None if not found
    """
    highest_max_context_length = 0
    for worker in self.workers:
      if model in worker.models:
        if worker.max_context_length > highest_max_context_length:
          highest_max_context_length = worker.max_context_length
    if highest_max_context_length > 0:
      return highest_max_context_length
    return 2048

horde_workers = HordeWorkers()