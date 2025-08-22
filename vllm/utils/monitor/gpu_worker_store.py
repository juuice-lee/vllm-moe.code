from vllm.utils.monitor.worker_store import WorkerStore
from typing import Optional, Any
from torch.utils.hooks import RemovableHandle
import torch
import inspect
from vllm.utils.custom.helpers import Colors
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.platforms import current_platform
import types
import torch.nn as nn

class RPCCallFunction:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        local_rank = kwargs.get("local_rank")
        rank = kwargs.get("rank")
        self._worker_store = WorkerStore()
        self._worker_store.set_device_info(device_id=local_rank, worker_id=rank)
    
    def check_method_exists(self, method: str) -> bool:
        """Check if a method exists in the worker."""
        return hasattr(self, method)

    def get_worker_store_rank(self) -> tuple[Optional[int], Optional[int]]:
        """Check the rank and local rank of the worker."""
        return self._worker_store.device_id, self._worker_store.worker_id

    # Call like llm.llm_engine.collective_rpc("get_module_names")
    def get_module_names(self) -> list[str]:
        """
        모델 내부 모든 서브모듈의 이름만 반환한다.
        C-확장 객체를 포함한 nn.Module 인스턴스는 직렬화 대상에서 제외해
        PyCapsule 오류를 방지한다.
        """
        # ''(루트) 이름은 필요 없으면 필터링
        return [
            name for name, _ in self.model_runner.model.named_modules() if name
        ]

    def get_worker_store_dict(self) -> dict[int, dict[str, Any]]:
        """Get the dictionary of worker store data."""
        return self._worker_store.to_dict()

    def register_latency_hooks(
        self,
        module_names: list[str],
    ) -> dict[str, tuple[RemovableHandle, RemovableHandle]]:
        """
        여러 모듈에 GPU-forward latency(ms)를 비동기 방식으로 수집하는
        훅을 등록한다.

        Parameters
        ----------
        module_names : list[str]
            `model.named_modules()` 로 얻을 수 있는 완전 경로 이름들의
            리스트.

        Returns
        -------
        handles_dict : dict[str, (pre_handle, post_handle)]
            모듈 이름 → (pre-hook, post-hook) 튜플.
            필요 시 `pre_handle.remove()`, `post_handle.remove()` 로 해제.
        """
        from vllm.v1.worker.gpu_worker import logger
        model = self.model_runner.get_model()
        named_mods = dict(model.named_modules())
        store = self._worker_store  # singleton

        handles_dict: dict[str, tuple[RemovableHandle, RemovableHandle]] = {}

        for mod_name in module_names:
            if mod_name not in named_mods:
                raise ValueError(f"Module '{mod_name}' not found in model.")
            module = named_mods[mod_name]

            # ── hook 정의 (클로저로 mod_name 캡처) ──────────────────────────
            def _pre_hook(_m, _inp, _name=mod_name):
                start_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                store.record_hook_data(_name, "_evt_pairs", start_evt)

                # Try to get parameter names from the forward method
                try:
                    forward_sig = inspect.signature(_m.forward)
                    param_names = list(forward_sig.parameters.keys())
                    # Remove 'self' if present
                    if param_names and param_names[0] == "self":
                        param_names = param_names[1:]
                except Exception:
                    param_names = None

                # Record input shapes with parameter names
                if isinstance(_inp, torch.Tensor):
                    if param_names and len(param_names) >= 1:
                        shape_info = {param_names[0]: list(_inp.shape)}
                    else:
                        shape_info = {"input": list(_inp.shape)}
                    store.record_hook_data(_name, "_input_shapes", shape_info)
                elif isinstance(_inp, (tuple, list)):
                    shape_info = {}
                    for idx, inp in enumerate(_inp):
                        if isinstance(inp, torch.Tensor):
                            if param_names and idx < len(param_names):
                                param_name = param_names[idx]
                            else:
                                param_name = f"input_{idx}"
                            shape_info[param_name] = list(inp.shape)
                    store.record_hook_data(_name, "_input_shapes", shape_info)

            def _post_hook(_m, _inp, _out, _name=mod_name):
                end_evt = torch.cuda.Event(enable_timing=True)
                end_evt.record()
                store.record_hook_data(_name, "_evt_pairs", end_evt)

            # ── hook 등록 ─────────────────────────────────────────────────
            # print(f"{mod_name} hooking")
            pre_h = module.register_forward_pre_hook(_pre_hook)
            post_h = module.register_forward_hook(_post_hook)
            handles_dict[mod_name] = (pre_h, post_h)

            # logger.debug(f"Registered latency hooks on {mod_name}")
        def _model_pre_hook(_m, _inp, _name="model"):
            start_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            store.record_hook_data(_name, "_evt_pairs", start_evt)

        def _model_post_hook(_m, _inp, _out, _name="model"):
            end_evt = torch.cuda.Event(enable_timing=True)
            end_evt.record()
            store.record_hook_data(_name, "_evt_pairs", end_evt)
        # model도 pre-hook, post-hook 등록해서 전체 chat time 측정
        # print(f"model hooking")
        model.register_forward_pre_hook(_model_pre_hook)
        model.register_forward_hook(_model_post_hook)

        self._worker_store._is_capturing_latency = True
        # logger.infowc(Colors.BLUE, "Capturing latency...")

        return True
    
    def register_moe_hooks(
        self,
        module_names: list[str],
        moe_gate_name: str = "gate",
    ):
        """
        Args:
            module_names (list[str]): list of module names to hook

        
        """
        from vllm.v1.worker.gpu_worker import logger
        model = self.model_runner.get_model()
        named_mods = dict(model.named_modules())
        store = self._worker_store # singleton

        for mod_name in module_names:
            if mod_name not in named_mods:
                raise ValueError(f"Module '{mod_name}' not found in model.")
            module = named_mods[mod_name]
            
            def _pre_hook(_m, _inp, _name=mod_name):
                # 모든 module의 시간 측정을 위해 event singleton에 추가 + input shape 기록
                start_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                store.record_hook_data(_name, "_evt_pairs", start_evt)
                
                try:
                    forward_sig = inspect.signature(_m.forward)
                    param_names = list(forward_sig.parameters.keys())
                    # Remove 'self' if present
                    if param_names and param_names[0] == "self":
                        param_names = param_names[1:]
                except Exception:
                    param_names = None
                    
                # Record input shapes with parameter names
                if isinstance(_inp, torch.Tensor):
                    if param_names and len(param_names) >= 1:
                        shape_info = {param_names[0]: list(_inp.shape)}
                    else:
                        shape_info = {"input": list(_inp.shape)}
                    store.record_hook_data(_name, "_input_shapes", shape_info)
                elif isinstance(_inp, (tuple, list)):
                    shape_info = {}
                    for idx, inp in enumerate(_inp):
                        if isinstance(inp, torch.Tensor):
                            if param_names and idx < len(param_names):
                                param_name = param_names[idx]
                            else:
                                param_name = f"input_{idx}"
                            shape_info[param_name] = list(inp.shape)
                    store.record_hook_data(_name, "_input_shapes", shape_info)
            
            def _post_hook(_m, _inp, _out, _name=mod_name):
                # 모든 module의 시간 event singleton에 추가
                # gate function일 때는 output을 singleton에 store해야 한다.
                end_evt = torch.cuda.Event(enable_timing=True)
                end_evt.record()
                store.record_hook_data(_name, "_evt_pairs", end_evt)
                # gate function일 경우: output을 _expert_score에 저장하기
                # mixtral은 gate function을 "gate"로 표현, llama4에서는 "router"로 표현
                if _name.split(".")[-1] == moe_gate_name:
                    if isinstance(_out, tuple):
                        store.record_hook_data(_name, "_expert_score", _out[0])
                    else:
                        store.record_hook_data(_name, "_expert_score", _out)
            # ── hook 등록 ─────────────────────────────────────────────────
            # print(f"{mod_name} hooking")
            module.register_forward_pre_hook(_pre_hook)
            module.register_forward_hook(_post_hook)

            logger.debug(f"Registered latency hooks on {mod_name}")
        def _model_pre_hook(_m, _inp, _name="model"):
            start_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            store.record_hook_data(_name, "_evt_pairs", start_evt)

        def _model_post_hook(_m, _inp, _out, _name="model"):
            end_evt = torch.cuda.Event(enable_timing=True)
            end_evt.record()
            store.record_hook_data(_name, "_evt_pairs", end_evt)
        # model도 pre-hook, post-hook 등록해서 전체 chat time 측정
        model.register_forward_pre_hook(_model_pre_hook)
        model.register_forward_hook(_model_post_hook)

        self._worker_store._is_capturing_latency = True

    def register_slo_hooks(
        self,
        module_names: list[str],
    ):
        """
        Args:
            module_names (list[str]): list of module names to hook
        """
        from vllm.v1.worker.gpu_worker import logger
        model = self.model_runner.get_model()
        named_mods = dict(model.named_modules())
        store = self._worker_store # singleton

        for mod_name in module_names:
            if mod_name not in named_mods:
                raise ValueError(f"Module '{mod_name}' not found in model.")
            module = named_mods[mod_name]
            
            def _pre_hook(_m, _inp, _name=mod_name):
                # 모든 module의 시간 측정을 위해 event singleton에 추가 + input shape 기록
                start_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                store.record_hook_data(_name, "_evt_pairs", start_evt)
                
                try:
                    forward_sig = inspect.signature(_m.forward)
                    param_names = list(forward_sig.parameters.keys())
                    # Remove 'self' if present
                    if param_names and param_names[0] == "self":
                        param_names = param_names[1:]
                except Exception:
                    param_names = None
                    
                # Record input shapes with parameter names
                if isinstance(_inp, torch.Tensor):
                    if param_names and len(param_names) >= 1:
                        shape_info = {param_names[0]: list(_inp.shape)}
                    else:
                        shape_info = {"input": list(_inp.shape)}
                    store.record_hook_data(_name, "_input_shapes", shape_info)
                elif isinstance(_inp, (tuple, list)):
                    shape_info = {}
                    for idx, inp in enumerate(_inp):
                        if isinstance(inp, torch.Tensor):
                            if param_names and idx < len(param_names):
                                param_name = param_names[idx]
                            else:
                                param_name = f"input_{idx}"
                            shape_info[param_name] = list(inp.shape)
                    store.record_hook_data(_name, "_input_shapes", shape_info)
            
            def _post_hook(_m, _inp, _out, _name=mod_name):
                # 모든 module의 시간 event singleton에 추가
                # gate function일 때는 output을 singleton에 store해야 한다.
                end_evt = torch.cuda.Event(enable_timing=True)
                end_evt.record()
                store.record_hook_data(_name, "_evt_pairs", end_evt)
            # ── hook 등록 ─────────────────────────────────────────────────
            # print(f"{mod_name} hooking")
            module.register_forward_pre_hook(_pre_hook)
            module.register_forward_hook(_post_hook)

            logger.debug(f"Registered latency hooks on {mod_name}")
            
        def _model_pre_hook(_m, _inp, _name="model"):
            start_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            store.record_hook_data(_name, "_evt_pairs", start_evt)

        def _model_post_hook(_m, _inp, _out, _name="model"):
            end_evt = torch.cuda.Event(enable_timing=True)
            end_evt.record()
            store.record_hook_data(_name, "_evt_pairs", end_evt)
        # model도 pre-hook, post-hook 등록해서 전체 chat time 측정
        model.register_forward_pre_hook(_model_pre_hook)
        model.register_forward_hook(_model_post_hook)

        self._worker_store._is_capturing_latency = True
    
    def aggregate_async_latencies(
        self,
        module_names: list[str] | None = None,
        pop: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        WorkerStore 에 비동기로 저장된 (start, end) CUDA 이벤트를 집계하여
        모듈별 latency(ms) 리스트와 input shapes를 반환한다.

        Parameters
        ----------
        store : WorkerStore
            단일 GPU-worker 프로세스의 singleton data store.
        module_names : list[str] | None, default None
            집계할 모듈 이름 리스트. None 이면 store 내 모든 모듈 대상.
        pop : bool, default False
            True 이면 집계 완료 후 이벤트 리스트를 store 에서 제거.

        Returns
        -------
        result_map : dict[str, dict[str, Any]]
            {module_name: {"latencies": [latency_ms, ...], "input_shapes": [shapes, ...]}} 형태의 결과.
        """
        store = self._worker_store

        # 모든 GPU 작업이 완료되었는지 확인
        torch.cuda.synchronize()

        # 집계 대상 결정
        hook_data = store.get_hook_data()
        target_modules = (
            module_names if module_names is not None else list(hook_data.keys())
        )

        result_map: dict[str, dict[str, Any]] = {}
        for mod in target_modules:
            evt_list: list[torch.cuda.Event] = hook_data.get(mod, {}).get(
                "_evt_pairs", []
            )
            input_shapes_list = hook_data.get(mod, {}).get("_input_shapes", [])

            # 짝수(시작/끝) 개수가 아니면 skip
            if len(evt_list) % 2 != 0 or len(evt_list) == 0:
                continue

            # Get metadata if available
            metadata_list = hook_data.get(mod, {}).get("_metadata", [])

            # Pair latencies with their corresponding input shapes and metadata
            paired_results = []
            num_iterations = len(evt_list) // 2

            for i in range(num_iterations):
                start_evt = evt_list[i * 2]
                end_evt = evt_list[i * 2 + 1]

                start_evt.synchronize()
                end_evt.synchronize()
                latency = start_evt.elapsed_time(end_evt)  # ms 단위

                # Get corresponding input shape if available
                input_shape = (
                    input_shapes_list[i] if i < len(input_shapes_list) else None
                )

                # Get corresponding metadata if available
                metadata = metadata_list[i] if i < len(metadata_list) else None

                paired_results.append(
                    {
                        "latency_ms": latency,
                        "input_shapes": input_shape,
                        "metadata": metadata,
                    }
                )

            result_map[mod] = {
                "paired_results": paired_results,
                # Keep backward compatibility
                # "latencies": [r["latency_ms"] for r in paired_results],
                # "input_shapes": input_shapes_list,
                # "_metadata": metadata_list,
            }

            if pop:
                # 집계가 끝났으면 이벤트 목록 초기화
                hook_data[mod]["_evt_pairs"].clear()
                if "_input_shapes" in hook_data[mod]:
                    hook_data[mod]["_input_shapes"].clear()

        return result_map
    
    def aggregate_async_moe_results(
        self,
        module_names: list[str] | None = None,
        pop: bool = False,
    ) -> dict[str, dict[str, Any]]:
        store = self._worker_store
        torch.cuda.synchronize()
        hook_data = store.get_hook_data()
        target_modules = module_names if module_names is not None else list(hook_data.keys())
        
        result_map: dict[str, dict[str, Any]] = {}
        for mod in target_modules:
            evt_list = hook_data.get(mod, {}).get("_evt_pairs", [])
            input_shapes_list = hook_data.get(mod, {}).get("_input_shapes", [])
            expert_scores_list = hook_data.get(mod, {}).get("_expert_score", [])
            
            if len(evt_list) % 2 != 0 or len(evt_list) == 0:
                continue
            paired_results = []
            num_iteration = len(evt_list)//2
            for i in range(num_iteration):
                start_evt = evt_list[i*2]
                end_evt = evt_list[i*2+1]
                start_evt.synchronize()
                end_evt.synchronize()
                latency = start_evt.elapsed_time(end_evt)
                
                input_shape = input_shapes_list[i] if i < len(input_shapes_list) else None
                
                expert_score = expert_scores_list[i] if i < len(expert_scores_list) else None
                
                paired_results.append({
                    "latency_ms": latency,
                    "input_shapes": input_shape,
                    "expert_scores": expert_score
                })
            result_map[mod] = {
                "paired_results": paired_results
            }
            if pop:
                # 집계가 끝났으면 이벤트 목록 초기화
                hook_data[mod]["_evt_pairs"].clear()
                if "_input_shapes" in hook_data[mod]:
                    hook_data[mod]["_input_shapes"].clear()
        return result_map
        
def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(
        parallel_config.world_size,
        rank,
        distributed_init_method,
        local_rank,
        backend,
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
    )

    ensure_kv_transfer_initialized(vllm_config)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the "
                "`dtype` flag in CLI, for example: --dtype=half."
            )
