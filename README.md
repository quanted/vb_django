## WVB Django

Django repository for managing the API for VB web. Static angular template files are managed from django.

## API Details

[![Run in Postman](https://run.pstmn.io/button.svg)](https://god.postman.co/run-collection/76a3c0e9b1fe2872f695)

## Adding Pipelines

Pipeline experimental development and testing repo: https://github.com/DouglasPatton/vbflow

Steps to add a new pipeline:
  * Individual pipelines are copied from vbflow into pipeline type files, such as gradient boosters going to gbr.py, in vb_django/vb_django/app.
  * Structure the class in the same manner as the existing examples, elasticnet.py and gbr.py.
    * Inherit from BaseEstimator, TransformerMixin, BaseHelper
    * Have a name, ptype, description, hyper_parameters and metrics class attributes
    * Initialize the class, also hitting super().__init__(pid)
    * Have a hyper_parameter validation function
    * Have a get pipeline function
  * Import the new pipeline class into vb_django.task_controller.py
  * Add the ptype of the new pipeline to the pipelines dict, starting at line 25 in task_controller.py
  * Add new conditional block in DaskTasks.execute_task, after line 92.
    * Class will be initialized
    * Hyper-parameters set
    * Call the fit function
    * Finally, call the save function
    * Any other additional functions that are necessary can be added here, such as the setup for cvpipe.
    
Some pipelines may not fall under this pattern, and will need to be more individually designed. But the sequential function calls should try to be maintained to help with code readability, troubleshooting and logging.