import os
from shutil import copyfile
from click_shell import make_click_shell


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


def _create_experiment_file(path):
    if not os.path.exists(path):
        os.makedirs(path)

    #TODO


