# Delegate Generation

Use generate_delegate.py to generate new boilerplate delegates.
- The default output directory is with in ```/generated```.
- The default template delegate is located within ```templates/temp_delegate```.
- You can also customise/create new template for future use within ```templates/```.



## How to use
* Create config for the new delegate
  * Look at the omni.json configs and make sure to replace the fields with similar values for your delegate
  * Place the new delegate.json config in the configs directory
* Run the ```python generate_delegate.py (config)```
  * Example: ```python generate_delegate.py omni```
* New delegate will be created in ```/generated```
* We recommend copying the new delegate into the [secda_delegates folder](../secda_delegates/) directory
  