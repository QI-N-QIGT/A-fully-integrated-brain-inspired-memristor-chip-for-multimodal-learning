import importlib.util

def convert_module_to_object(path, obj_name, *args, **kwargs):
    # Specify the absolute path of the module
    module_path = path

    # Define a name for the module (optional)
    module_name = 'module'

    # Create a spec for the module
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    # Load the module using the spec
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # get the object in module
    obj = getattr(module, obj_name)(*args, **kwargs)
    return obj

    
    