# Edge Impulse model export

To proceed with export and to generate a target specific executable .eim file,
use following docker command.
As output, executable will be saved in mounted volume.

```sh
docker run --rm -it \
    -v /home/arduino/ei-models:/data \
    public.ecr.aws/z9b3d4t5/inference-container-qc-adreno-702:524ed24b0e99ffad63f12dc248f37fbc809ba5d3 \
        --api-key <model api key> \
        --download /data/out-model.eim	
```

before proceeding, update container reference and generate a valid project API key.

To force GPU mode, add following parameter
```sh
--force-target runner-linux-aarch64-gpu
```
