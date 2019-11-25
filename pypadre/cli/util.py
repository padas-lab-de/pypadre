import os
import shutil
from click_shell import make_click_shell

from pypadre.pod.app.base_app import BaseEntityApp


def shorten_prompt_id(id):
    id = str(id)
    return (id[:5] + '..' + id[-5:]) if len(id) > 12 else id


def make_sub_shell(ctx, obj_name, obj, intro):
    wrapping_prompt = ctx.obj['prompt']
    prompt = ctx.obj['prompt'] + (obj_name[:3] if len(obj_name) > 3 else obj_name) + ": " + shorten_prompt_id(
        obj.id) + ' > '
    s = make_click_shell(ctx, prompt=prompt, intro=intro,
                         hist_file=os.path.join(os.path.expanduser('~'), '.click-pypadre-history'))
    ctx.obj['prompt'] = prompt
    ctx.obj[obj_name] = obj
    s.cmdloop()
    ctx.obj['prompt'] = wrapping_prompt
    del ctx.obj[obj_name]


def get_by_app(ctx, app: BaseEntityApp, id):
    objects = app.list({"id": id})
    if len(objects) == 0:
        print(app.model_clz.__name__ + " {0} not found!".format(id))
        return None
    if len(objects) > 1:
        print("Multiple matching entries of type " + app.model_clz.__name__ + " found!")
        _print_class_table(ctx, app.model_clz, objects)
        return None
    return objects.pop(0)


def _print_class_table(ctx, clz, *args, **kwargs):
    ctx.obj["pypadre-app"].print_tables(clz, *args, **kwargs)


def _create_experiment_file(path=None, file_name=None):
    if not os.path.exists(path):
        os.makedirs(path)
    src = os.path.join(os.path.dirname(__file__), 'experiment/experiment_template.py')
    dst = path + '/' + file_name + '.py'
    shutil.copyfile(src=src, dst=dst)
    return dst
