
# Abstract

We need to release some artifacts for different purposes (logical list of what is required, not yet described as package):
 - WHL file containing code. To be published on python repository (PyPI)
 - A file containing a materialized list of modules (for AppsLab)
 - A file (set of?) containing the conf variables for the module (to be configured by AppsLab/User)
 - A list of supported models (probably static containing LLMs/AI models) that can be used/installed by Lab
 - Module code examples (for AppsLab)
 - Container images to be used by modules

# Possible packages (release artifacts)

How we can organize above content:
 - 0:N containers will be internally developed on private ECR and finally pushed to Docker Hub for public
 - 1 WHL file published on PiPI
 - 1 YAML file for available MODELS (static - for now only for Arduino modules - no custom)

options:
 - 1 overall index of different libraries (for OOTB not needed, there is only one. For custom, will list all possible custom modules with a brief exaplanation of what is it and download link and PyPI package name -> needed to know what to add in requirements.txt)
 - for every library, 1 YAML file for the list of modules available inside library
   - releted archive (zip) with examples/variables OR everything inside this YAML file

# How to release

## 1. Python module release process



## How to release containers (how to release and update dependencies inside code)

Options:
 - version will be discovered at runtime for arduino_bricks library and will be exported in compose file as variable called APPSLAB_LIB_VERSION. Then we need to refernce it and will be reseolved while running compose. Version can be extracted at runtime from: "pip show arduino_bricks" (to be checked how to do it in code)
   - like: arduino/appslab-modules:models-runner-v${APPSLAB_LIB_VERSION}
   - to pin a specifc container, do not add any variable
  

# NOTES from DESIGN dock

When I click on a brick, show the module description and info, RELATED models if available and Code snippets.
