import importlib

def import_class(name=None):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)