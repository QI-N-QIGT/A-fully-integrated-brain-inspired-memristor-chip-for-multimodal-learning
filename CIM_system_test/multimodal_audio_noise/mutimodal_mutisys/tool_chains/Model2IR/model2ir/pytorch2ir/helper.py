import importlib.util

def convert_module_to_object(path, obj_name, *args, **kwargs):
    module_path = path
    module_name = 'module'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    obj = getattr(module, obj_name)(*args, **kwargs)
    return obj