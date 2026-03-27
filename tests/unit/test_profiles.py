"""Unit tests for dynamic backend profile orchestration."""

from tokenflow.classifier import RequestClassifier
from tokenflow.models import BackendType, CostClass, EndpointProfile, EndpointRegisterRequest, GPUClass
from tokenflow.profiles import BackendProfileTemplate, ProfileManager


class DummyRegistry:
    def __init__(self):
        self.items = {}

    async def register(self, req: EndpointRegisterRequest):
        profile = EndpointProfile(id=req.name, **req.model_dump())
        self.items[profile.id] = profile
        return profile

    async def delete(self, endpoint_id: str):
        self.items.pop(endpoint_id, None)
        return True

    async def find_by_model(self, model_name: str):
        return [p for p in self.items.values() if p.model_name == model_name]


class DummyCollector:
    def __init__(self):
        self.registered = []
        self.unregistered = []

    def register_endpoint(self, endpoint):
        self.registered.append(endpoint.id)

    def unregister_endpoint(self, endpoint_id: str):
        self.unregistered.append(endpoint_id)


class DummyTelemetryStore:
    def get(self, endpoint_id: str):
        return None


clf = RequestClassifier()


def make_template(name: str, backend: BackendType, gpu: GPUClass, model: str):
    return BackendProfileTemplate(
        name=name,
        nim_url=f"http://{name}:8000",
        backend_type=backend,
        model_name=model,
        gpu_name=gpu,
        cost_class=CostClass.STANDARD,
        activation_model_names=[model],
        idle_ttl_seconds=60,
    )


async def _setup_manager():
    manager = ProfileManager()
    manager.attach(DummyRegistry(), DummyCollector(), DummyTelemetryStore())
    return manager


async def test_ensure_capacity_activates_best_matching_template():
    manager = await _setup_manager()
    nim = make_template("nim-h100", BackendType.NIM, GPUClass.H100, "meta/llama-3.1-70b-instruct")
    vllm = make_template("vllm-h200", BackendType.VLLM, GPUClass.H200, "meta/llama-3.1-70b-instruct")
    await manager.add_template(nim)
    await manager.add_template(vllm)

    profile = clf.classify(
        {
            "model": "meta/llama-3.1-70b-instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "max_tokens": 128,
        }
    )

    await manager.ensure_capacity_for_request(profile)
    active = [t for t in await manager.list_templates() if t.activated]
    assert len(active) == 1
    assert active[0].backend_type == BackendType.NIM


async def test_exclusive_model_residency_deactivates_siblings():
    manager = await _setup_manager()
    sglang = make_template("sglang-h100", BackendType.SGLANG, GPUClass.H100, "meta/llama-3.1-8b-instruct")
    dynamo = make_template("dynamo-h100", BackendType.DYNAMO, GPUClass.H100, "meta/llama-3.1-8b-instruct")
    await manager.add_template(sglang)
    await manager.add_template(dynamo)

    await manager.activate_template(sglang.id)
    assert (await manager.get_template(sglang.id)).activated is True

    profile = clf.classify(
        {
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": "x " * 3000}],
            "max_tokens": 128,
        }
    )
    await manager._activate(dynamo, profile)

    assert (await manager.get_template(dynamo.id)).activated is True
    assert (await manager.get_template(sglang.id)).activated is False
