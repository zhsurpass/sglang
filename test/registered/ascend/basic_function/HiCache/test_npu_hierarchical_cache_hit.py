import time
import unittest

import requests
from transformers import AutoTokenizer

from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    NIC_NAME,
    TestAscendMultiNodePdSepTestCaseBase,
    check_role,
)
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
)

MODEL_CONFIG_HIERARCHICAL_CACHE = {
    "model_path": DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
    "prefill_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "ENABLE_MOE_NZ": "1",
        "HCCL_BUFFSIZE": "1536",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "TASK_QUEUE_ENABLE": "2",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "decode_envs": {
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "ENABLE_MOE_NZ": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "HCCL_BUFFSIZE": "720",
        "SGLANG_DP_ROUND_ROBIN": "1",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_SOCKET_IFNAME": NIC_NAME,
        "GLOO_SOCKET_IFNAME": NIC_NAME,
    },
    "prefill_args": [
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-transfer-backend",
        "ascend",
        "--tp-size",
        "16",
        "--mem-fraction-static",
        "0.8",
        "--quantization",
        "modelslim",
        "--context-length",
        "8192",
        "--chunked-prefill-size",
        "-1",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--trust-remote-code",
        "--disable-cuda-graph",
        # enable L1&L2 cache in prefill node
        "--enable-hierarchical-cache",
        "--dtype",
        "bfloat16",
    ],
    "decode_args": [
        "--nnodes",
        "1",
        "--disaggregation-mode",
        "decode",
        "--disaggregation-transfer-backend",
        "ascend",
        "--tp-size",
        "16",
        "--mem-fraction-static",
        "0.8",
        "--quantization",
        "modelslim",
        "--context-length",
        "8192",
        "--chunked-prefill-size",
        "-1",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--trust-remote-code",
        "--cuda-graph-bs",
        "256",
        "128",
        "64",
        "--watchdog-timeout",
        "9000",
        "--dtype",
        "bfloat16",
    ],
    "router_args": [],
}


class TestNPUHierarchicalCacheHit(TestAscendMultiNodePdSepTestCaseBase):
    """Testcase：Verify enabling L1 and L2 cache by configuring --enable-hierarchical-cache in PD separation scenario,
    sending two requests with the same prefix achieves standard cache hit count and reduced TTFT.

    [Test Category] Parameter
    [Test Target] --enable-hierarchical-cache
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tokenizer = AutoTokenizer.from_pretrained(
            DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH, trust_remote_code=True
        )
        cls.__class__.model_config = MODEL_CONFIG_HIERARCHICAL_CACHE
        cls.start_pd_server()
        cls.start_router_server()

    @classmethod
    def tearDownClass(cls):
        cls.stop_sglang_thread()
        super().tearDownClass()

    @check_role(allowed_roles=["router"])
    def send_long_prompt_request(self, prompt_token_len=600, max_new_tokens=1):
        """Send requests with specified prompt token length and identical prefix content.
        Return TTFT and cached_tokens of the request.

        :param prompt_token_len: Token length of prompt content, default 600
        :param max_new_tokens: Maximum number of generated new tokens, default 1
        :return: tuple(TTFT, cached_tokens)
        """

        prompt = "hello world " * (prompt_token_len // 2 + 1)
        prompt = self.tokenizer.decode(
            self.tokenizer.encode(prompt, add_special_tokens=False)[:prompt_token_len]
        )

        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            },
        )
        ttft = time.time() - start_time

        self.assertEqual(response.status_code, 200, "Failed to call generate API")
        result = response.json()
        cached_tokens = result.get("meta_info").get("cached_tokens", 0)
        return ttft, cached_tokens

    def test_hierarchical_cache_hit_and_ttft_reduce(self):
        """Send two requests with identical prefix content.
        Verify the first request has no cache hit, the second request gets expected cached tokens,
        and TTFT decreases after cache hits.
        """

        ttft_1, cached_tokens_1 = self.send_long_prompt_request(
            prompt_token_len=600, max_new_tokens=1
        )
        self.assertEqual(
            cached_tokens_1, 0, msg="First request cached tokens should be 0"
        )

        ttft_2, cached_tokens_2 = self.send_long_prompt_request(
            prompt_token_len=600, max_new_tokens=1
        )
        self.assertEqual(cached_tokens_2, 512, msg="Cache hit tokens should be 512")
        self.assertLess(ttft_2, ttft_1, msg="TTFT should be reduced after cache hit")


if __name__ == "__main__":
    unittest.main()
