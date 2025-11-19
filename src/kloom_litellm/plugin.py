import os
from typing import Literal, Optional, Union

import httpx
import litellm
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.utils import get_llm_provider


# Define your plugin class, inheriting from CustomLogger
class KloomPlugin(CustomLogger):
    def __init__(
        self,
        kloom_base_url: Optional[str] = None,
        kloom_api_key: Optional[str] = None,
        kloom_project_id: Optional[str] = None,
        multi_model_alpha: Optional[float] = None,
        default_multi_model: Optional[str] = None,
        model_routing_enabled: Optional[bool] = None,
        request_tracking_enabled: Optional[bool] = None,
    ):
        """
        Initialize the plugin, potentially with configuration.
        """
        super().__init__()  # Call the base class initializer
        self.async_client = httpx.AsyncClient()
        self.sync_client = httpx.Client()
        self.kloom_base_url = (
            kloom_base_url
            if kloom_base_url is not None
            else os.environ.get("KLOOM_BASE_URL", "https://api.kloom.ai")
        )
        self.kloom_api_key = (
            kloom_api_key
            if kloom_api_key is not None
            else os.environ.get("KLOOM_API_KEY")
        )
        self.kloom_project_id = (
            kloom_project_id
            if kloom_project_id is not None
            else os.environ.get("KLOOM_PROJECT_ID")
        )
        self.default_multi_model = (
            default_multi_model
            if default_multi_model is not None
            else os.environ.get("KLOOM_DEFAULT_MULTI_MODEL")
        )
        self.multi_model_alpha = float(
            multi_model_alpha
            if multi_model_alpha is not None
            else os.environ.get("KLOOM_MULTI_MODEL_ALPHA", "0.5")
        )
        print("KloomPlugin Configuration:")
        print(f"\tkloom_base_url = {self.kloom_base_url}")
        print(f"\tkloom_api_key  = ...{self.kloom_api_key[-4:]}")
        print(f"\tdefault_multi_model = {self.default_multi_model}")
        print(f"\tmulti_model_alpha   = {self.multi_model_alpha}")
        disabled_due_to_config_issue = False
        if not self.kloom_base_url:
            # This only happens if the user explicitly sets the environment variable to ""
            disabled_due_to_config_issue = True
            print(
                "KloomPlugin was not properly configured with a URL. Please either pass a valid URL or remove the KLOOM_BASE_URL environment variable."
            )
        if not self.kloom_api_key:
            disabled_due_to_config_issue = True
            print(
                "KloomPlugin was not configured with an API key. Please either set this manually or via the KLOOM_API_KEY environment variable."
            )
        if disabled_due_to_config_issue:
            self.model_routing_enabled = False
            self.request_tracking_enabled = False
        else:
            self.model_routing_enabled = (
                model_routing_enabled
                if model_routing_enabled is not None
                else os.environ.get("KLOOM_MODEL_ROUTING_ENABLED", "") == "1"
            )
            self.request_tracking_enabled = (
                request_tracking_enabled
                if request_tracking_enabled is not None
                else os.environ.get("KLOOM_REQUEST_TRACKING_ENABLED", "") == "1"
            )
        litellm.print_verbose(
            f"KloomPlugin Initialized with model_routing_enabled={self.model_routing_enabled} and request_tracking_enabled={self.request_tracking_enabled}"
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",  # Add other relevant types
        ],
    ) -> Optional[Union[dict, str]]:
        litellm.print_verbose("------ KloomPlugin Start ------")
        if call_type not in {"completion", "text_completion"}:
            litellm.print_verbose("------ KloomPlugin End (not an LLM query) ------")
            return None
        if not self.model_routing_enabled:
            litellm.print_verbose("------ KloomPlugin: Model Routing disabled ------")
        else:
            if data["model"] != "auto" and not data["model"].startswith("kloom/auto-"):
                litellm.print_verbose(
                    '------ KloomPlugin End (Skipping; model != "auto") ------'
                )
                return None
            if data["model"].startswith("kloom/auto-"):
                router_id = data["model"].lstrip("kloom/auto-")
            else:
                # "model": "auto"
                router_id = self.default_multi_model
            if router_id is None:
                litellm.print_verbose(
                    '------ KloomPlugin End (Skipping; model == "auto" and no default set. Set with KLOOM_DEFAULT_MULTI_MODEL) ------'
                )
                return None
            router_alpha = float(
                data.get("metadata", {}).get("router_alpha", self.multi_model_alpha)
            )
            # Store router information in metadata for use in logging hooks
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["kloom_router_id"] = router_id
            data["metadata"]["kloom_router_alpha"] = router_alpha
            data["metadata"]["kloom_original_model"] = data["model"]
            # [TODO] Support selecting models via model-specific API key (requires new endpoint)
            async with self.async_client as client:
                resp = await client.post(
                    f"{self.kloom_base_url}/router-management/router/{router_id}/recommend",
                    json={
                        "input": data,
                        "alpha": router_alpha,
                    },
                )
                cand_found = False
                if resp.get("all_candidates", []):
                    worst_cand = min(
                        resp["all_candidates"], key=lambda x: x["price_score"]
                    )
                    data["metadata"]["kloom_worst_price_model"] = worst_cand["model"]
                    data["metadata"]["kloom_worst_price_provider"] = worst_cand[
                        "provider"
                    ]
                for cand in resp.get("all_candidates", []):
                    if cand.get("litellm_slug", None):
                        data["model"] = cand["litellm_slug"]
                        cand_found = True
                if not cand_found:
                    raise RuntimeError(
                        "No multi-model candidate has a corresponding LiteLLM model. Check your Kloom multi-model's configuration (note that LiteLLM does not support NAVER or Alibaba)."
                    )
        litellm.print_verbose("------ KloomPlugin End ------")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log successful LLM completions to Kloom."""
        try:
            # Retrieve metadata stored during pre_call_hook
            metadata = kwargs.get("litellm_params", {}).get("metadata", {})
            router_id = metadata.get("kloom_router_id", None)
            project_id = self.kloom_project_id if router_id is None else None

            try:
                model, litellm_model_provider, _, _ = get_llm_provider(
                    kwargs.get("model", "unknown")
                )
                invoked_model_name = {
                    "type": "litellm",
                    "slug": f"{litellm_model_provider}/{model}",
                }
            except Exception:
                invoked_model_name = None

            # Extract data from LiteLLM response
            request_data = {
                "router_id": router_id,
                "project_id": project_id,
                "model_name": kwargs.get("metadata", {}).get(
                    "kloom_original_model", kwargs.get("model", "unknown")
                ),
                "invoked_model_name": invoked_model_name,
                "provider_returned_model": response_obj.get("model"),
                "request_payload": {
                    "model": kwargs.get("model"),
                    "messages": kwargs.get("messages", []),
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "top_p": kwargs.get("top_p"),
                    "stream": kwargs.get("stream", False),
                },
                "response_payload": response_obj.model_dump()
                if hasattr(response_obj, "model_dump")
                else response_obj,
                "consumed_tokens": response_obj.get("usage", {}).get("total_tokens"),
                "cost_usd": kwargs.get("response_cost"),
                "provider_id": kwargs.get("litellm_params", {}).get(
                    "custom_llm_provider"
                ),
                "status": "success",
                "baseline_model_id": metadata.get("kloom_worst_price_model", None),
                "baseline_provider_id": metadata.get(
                    "kloom_worst_price_provider", None
                ),
            }

            # POST to Kloom logging endpoint
            try:
                async with self.async_client as client:
                    response = await client.post(
                        url=f"{self.kloom_base_url}/logging/chat-completions",
                        json=request_data,
                        headers={
                            "Authorization": f"Bearer {self.kloom_api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=10.0,
                    )
            except RuntimeError:
                response = self.sync_client.post(
                    url=f"{self.kloom_base_url}/logging/chat-completions",
                    json=request_data,
                    headers={
                        "Authorization": f"Bearer {self.kloom_api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )

            if response.status_code == 200:
                log_id = response.json().get("id")
                print(f"✓ Logged to Kloom: {log_id}")
            else:
                print(
                    f"✗ Kloom logging failed: {response.status_code} - {response.text}"
                )

        except Exception as e:
            print(f"✗ Error logging to Kloom: {e}")
            import traceback

            traceback.print_exc()

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failed LLM completions to Kloom."""
        try:
            # Retrieve metadata stored during pre_call_hook
            metadata = kwargs.get("litellm_params", {}).get("metadata", {})
            router_id = metadata.get("kloom_router_id", None)
            project_id = self.kloom_project_id if router_id is None else None

            # Extract exception info
            # exception = kwargs.get("exception")

            request_data = {
                "router_id": router_id,
                "project_id": project_id,
                "model_name": kwargs.get("model", "unknown"),
                "request_payload": {
                    "model": kwargs.get("model"),
                    "messages": kwargs.get("messages", []),
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                },
                "response_payload": None,
                "status": "failed",
                "provider_id": kwargs.get("litellm_params", {}).get(
                    "custom_llm_provider"
                ),
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.kloom_base_url}/logging/chat-completions",
                        json=request_data,
                        headers={
                            "Authorization": f"Bearer {self.kloom_api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=10.0,
                    )
            except RuntimeError:
                response = self.sync_client.post(
                    f"{self.kloom_base_url}/logging/chat-completions",
                    json=request_data,
                    headers={
                        "Authorization": f"Bearer {self.kloom_api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )

            if response.status_code == 200:
                log_id = response.json().get("id")
                print(f"✓ Logged failure to Kloom: {log_id}")
            else:
                print(f"✗ Kloom failure logging failed: {response.status_code}")

        except Exception as e:
            print(f"✗ Error logging failure to Kloom: {e}")
