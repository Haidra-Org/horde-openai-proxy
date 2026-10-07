import time
from json import JSONDecodeError
from typing import List
from loguru import logger
import requests
from fastapi import HTTPException
from .consts import VERSION

from .types import HordeRequest, TextGeneration
from .config import Config

def remove_stop_words(text: str, stop_sequence: List[str]) -> str:
    """
    Clean up the response text by removing trailing stop words.
    :param text: The text to clean up.
    :param stop_sequence: The stop sequence to remove.
    :return:
    """
    for stop_word in stop_sequence:
        text = text.rstrip(stop_word)
    return text


# def get_data(response: requests.Response):
#     if response.status_code not in (200, 202):
#         try:
#             error_data = response.json()
#             message = error_data.get("message")
#             errors = error_data.get("errors")
#             detail = f"Error: {message}"
#             if errors:
#                 detail += f" - {errors}"
#             raise HTTPException(status_code=response.status_code, detail=detail)
#         except (requests.exceptions.JSONDecodeError, KeyError, ValueError):
#             # Fallback if the remote service didn't return valid JSON
#             raise HTTPException(
#                 status_code=response.status_code, 
#                 detail=f"Error: Received status code {response.status_code} from upstream"
#             )
            
#     return response.json()


@logger.catch(reraise=True)
def get_data(response: requests.Response, client_ip: str):
    if response.status_code not in (200, 202):
        target_url = response.url 
        
        try:
            error_data = response.json()
            message = error_data.get("message")
            errors = error_data.get("errors")
            detail = f"Error: {message}"
            # Extract the target URL that failed
            if errors:
                detail += f" - {errors}"
                
            logger.error(f"Upstream error {response.status_code} at {target_url} | Client IP: {client_ip} | Detail: {detail}")
            raise HTTPException(status_code=response.status_code, detail=detail)
            
        except (requests.exceptions.JSONDecodeError, KeyError, ValueError):
            fallback_detail = f"Error: Received status code {response.status_code} from upstream"
            logger.error(f"Upstream error {response.status_code} at {target_url} | Client IP: {client_ip} | Invalid JSON payload")
            
            raise HTTPException(
                status_code=response.status_code, 
                detail=fallback_detail
            )
            
    return response.json()

@logger.catch(reraise=True)
def get_horde_completion(
    apikey: str,
    request: HordeRequest,
    *,
    trusted_workers: bool = False,
    validated_backends: bool = True,
    slow_workers: bool = True,
    allow_downgrade: bool = False,
) -> List[TextGeneration]:
    """
    Request text completions from the StableHorde API and awaits the completions.
    Raises a ValueError if the request is not possible, faulted, timed out, or if there are not enough generations.
    :param apikey: API key for the StableHorde API.
    :param request: HordeRequest
    :param trusted_workers: Only use workers that have been trusted.
    :param validated_backends: Only use backends that have been validated.
    :param slow_workers: Allow slow workers to be used.
    :param allow_downgrade: Allow downgrading context length if necessary.
    :return: List of TextGeneration
    :raises ValueError
    """
    body = {
                "prompt": request.prompt,
                "models": request.models,
                "params": request.params.model_dump(exclude_none=True),
                "trusted_workers": trusted_workers,
                "validated_backends": validated_backends,
                "slow_workers": slow_workers,
                "allow_downgrade": allow_downgrade,
            }
    initial_request = get_data(
        response = requests.post(
            f"{Config.horde_url}/api/v2/generate/text/async",
            headers={
                "apikey": apikey,
                "Client-Agent": f"horde-openai-proxy:{VERSION}:db0",
                "Proxied-For": request.origin_ip,
                "Proxy-Authorization": Config.horde_proxy_passkey,
            },
            json=body,
        ),
        client_ip = request.origin_ip
    )
    uuid = initial_request["id"]

    # Await the completion
    initial_time = time.time()
    while time.time() - initial_time < request.timeout:
        data = get_data(
            response = requests.get(
                f"{Config.horde_url}/api/v2/generate/text/status/{uuid}",
                headers={
                    "Client-Agent": f"horde-openai-proxy:{VERSION}:db0",
                    "Proxied-For": request.origin_ip,
                    "Proxy-Authorization": Config.horde_proxy_passkey,
                },
            ),
            client_ip = request.origin_ip

        )
        if not data["is_possible"]:
            data = get_data(
                response = requests.delete(
                    f"{Config.horde_url}/api/v2/generate/text/status/{uuid}",
                    headers={
                        "Client-Agent": f"horde-openai-proxy:{VERSION}:db0",
                        "Proxied-For": request.origin_ip,
                        "Proxy-Authorization": Config.horde_proxy_passkey,
                    },
                ),
                client_ip = request.origin_ip
            )
            raise ValueError("Request is not possible.")

        if data["faulted"]:
            raise ValueError("Request has faulted.")

        if data["done"]:
            if len(data["generations"]) < (
                1 if request.params.n is None else request.params.n
            ):
                raise ValueError("Not enough generations.")

            # Parse the generations
            generations = []
            for generation in data["generations"]:
                text = remove_stop_words(
                    generation["text"],
                    request.params.stop_sequence,
                )
                generations.append(
                    TextGeneration(
                        uuid=str(uuid),
                        model=generation["model"],
                        text=text,
                        kudos=data["kudos"],
                    )
                )
            return generations
        else:
            time.sleep(1)

    raise ValueError("Request timed out.")


@logger.catch(reraise=True)
def get_horde_models(origin_ip) -> List[dict]:
    """
    Get the models available on the StableHorde API.
    :return: List of models.
    :raises ValueError
    """
    return get_data(
        response = requests.get(
            f"{Config.horde_url}/api/v2/status/models",
            params={
                "type": "text",
                "min_count": 1,
            },
            headers={
                "Client-Agent": f"horde-openai-proxy:{VERSION}:db0",
            },
        ),
        client_ip = origin_ip
    )
