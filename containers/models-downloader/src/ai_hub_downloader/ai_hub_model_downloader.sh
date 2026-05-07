#!/bin/bash

cd /models

qai_hub_models fetch ${model_name} -r ${model_type} -p ${quantization} -c ${chipset} -v ${version}
if [ $? -ne 0 ]; then
    echo "Failed to download the model: ${model_name}"
    exit 1
fi