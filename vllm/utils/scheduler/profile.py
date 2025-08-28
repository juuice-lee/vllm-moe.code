import functools
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

logger = init_logger(__name__)

"""
Step별로 scheduled 된 request 정보를 profile 하기 위한 utility
"""

class SchedulerProfiler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            mm_registry = mm_registry,
            include_finished_set = include_finished_set,
            log_stats = log_stats,
        )
        self.scheduled_outputs: list[SchedulerOutput] =[] # list of scheduler output per step
        # schedule 호출 후에는 scheduled_outputs에 step별 output을 append해야 한다.
    
    def schedule(self)->SchedulerOutput:
        out = super().schedule()
        # 결과를 기록
        if out is None:
            return None
        self.scheduled_outputs.append(out)
        return out
    