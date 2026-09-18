class ASRWorkerLifecycle:
    """vLLM extension for explicit NCCL teardown through a named RPC."""

    def close_mybot_distributed(self) -> None:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
