# CARLA installation and cost decision

## Is CARLA paid?

CARLA is not a paid API or a subscription service. The official repository describes CARLA as an open-source simulator for autonomous-driving research. CARLA-specific code is distributed under the MIT License and the included CARLA assets are distributed under CC-BY. Unreal Engine has its own license terms, so those terms still apply if CARLA is built from source. [1] [2]

For this project, the practical conclusion is:

- no CARLA API key is required;
- no per-request or monthly CARLA fee is required;
- local execution is free if the computer already has suitable hardware;
- a cloud GPU, electricity, disk and operating-system setup may still cost money independently of CARLA.

## Recommended route for this project

Use the **packaged CARLA 0.9.15 release** first instead of building the Unreal Engine branch. The official quick-start guide says the package includes both the CARLA server and the Python client library, and it recommends the current packaged release as 0.9.15 in that documentation. [2]

The packaged route is the smallest integration step:

1. Download the Linux or Windows package from the official CARLA releases.
2. Extract it on a local computer with a supported dedicated GPU.
3. Install the matching Python client with `python3 -m pip install carla` or the wheel included in `PythonAPI/carla/dist`.
4. Start the server.
5. Run ihateparking’s `CarlaScenarioAdapter` from a second terminal.

The server/client relationship is important: the CARLA server owns the simulated world, while the Python client connects to it over the default port 2000. The official first-steps guide demonstrates `carla.Client('localhost', 2000)` followed by `client.get_world()`. [3]

## Practical requirements

The packaged quick-start documentation lists Ubuntu 20.04/22.04 or Windows 10/11, around 20 GB of disk, two available TCP ports (2000 and 2001), Python 3.7–3.12 and a dedicated GPU around an NVIDIA RTX 2070 or better with at least 8 GB VRAM. [2]

The newer UE5 development branch has higher requirements: the official repository lists Ubuntu 22.04/24.04 or Windows 11, at least 32 GB RAM and an NVIDIA RTX 3070/3080/3090/4090/5090-class GPU with 16 GB or more VRAM. That is why the packaged 0.9.15 path is preferable for the first portfolio milestone. [1]

The current Manus Sandbox has no CARLA Python API and no live CARLA server. It is still useful for developing and testing scenario generation, contracts, GroundTruthBackend and the planner. The live simulator should run on a suitable local machine or GPU computer, while the project code can be copied there or connected through the existing repository.

## Example local setup

```bash
# 1. Inside the extracted CARLA package
cd ~/CARLA_0.9.15

# 2. Optional isolated Python environment for the project
python3 -m venv ~/venvs/ihateparking-carla
source ~/venvs/ihateparking-carla/bin/activate
python -m pip install --upgrade pip
python -m pip install carla==0.9.15

# 3. Install ihateparking dependencies in the same environment
cd /path/to/ihateparking
python -m pip install -r requirements.txt
python -m pip install -r requirements-vision.txt

# 4. Terminal A: start the CARLA server
cd ~/CARLA_0.9.15
./CarlaUE4.sh -RenderOffScreen -quality-level=Low

# 5. Terminal B: run the project-side smoke test
cd /path/to/ihateparking
python3 - <<'PY'
from ihateparking.carla_adapter import CarlaScenarioAdapter
from ihateparking.scenario_generator import ScenarioConfig, ScenarioGenerator

scenario = ScenarioGenerator().generate(ScenarioConfig(seed=7))
adapter = CarlaScenarioAdapter()
try:
    adapter.connect(scenario)
    plan = adapter.spawn_vehicle_and_sensors(scenario)
    print(plan.to_dict())
    for _ in range(20):
        adapter.tick()
finally:
    adapter.close()
PY
```

If the Python wheel bundled with the extracted package is the authoritative one for that release, install that wheel instead of mixing it with a different CARLA version:

```bash
cd ~/CARLA_0.9.15/PythonAPI/carla/dist
python -m pip install carla-0.9.15-*.whl
```

## If there is no suitable GPU

We can still continue the project without paying for anything:

- keep developing the planner, scenario generator and GroundTruthBackend in the current sandbox;
- use the generated scenario suite as a deterministic contract test;
- run CARLA later on an existing local NVIDIA machine;
- use a rented GPU only for the periods when live CARLA rendering or model training is required.

A fully GPU-free fallback would be a lower-fidelity simulator such as Webots or a custom 2-D/3-D renderer, but that would be a different sensor realism target. For this portfolio goal, CARLA remains the stronger free simulator if the hardware is available.

## References

[1]: https://github.com/carla-simulator/carla "Official CARLA repository and license notes"

[2]: https://carla.readthedocs.io/en/0.9.16/start_quickstart/ "CARLA 0.9.16 quick-start package installation"

[3]: https://carla.readthedocs.io/en/0.9.16/tuto_first_steps/ "CARLA first steps, client connection, maps and sensors"
