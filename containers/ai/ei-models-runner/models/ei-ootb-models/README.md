# Edge Impulse model export

To proceed with export and to generate a target specific executable .eim file,
use following docker command.
As output, executable will be saved in mounted volume.

```sh
docker run --rm -it \
    -v /home/arduino/ei-models:/data \
    public.ecr.aws/g7a8t7v6/inference-container-qc-adreno-702:latest \
        --api-key <model api key> \
        --download /data/out-model.eim	
```

before proceeding, update container reference and generate a valid project API key.

To force GPU mode, add following parameter
```sh
--force-target runner-linux-aarch64-gpu
```
Default exported models will be executed in CPU mode. 
