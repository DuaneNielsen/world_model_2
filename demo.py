from pathlib import Path
import torch
import os
from cursesmenu import CursesMenu
from cursesmenu.items import FunctionItem, SubmenuItem, CommandItem
from worldmodel import get_args, demo, defaults, make_env
import worldmodel

if __name__ == '__main__':

    args = get_args(defaults)
    worldmodel.args = args

    wandb_run_dirs = Path().glob(f'wandb/*')
    wandb_run_dirs = sorted(wandb_run_dirs, key=os.path.getmtime, reverse=True)

    env = make_env()

    menu = CursesMenu()

    for dir in wandb_run_dirs[:10]:
        policy_file = dir/'policy_best.pt'
        if policy_file.exists():
            file = torch.load(str(policy_file))
            stats = ''
            for key, value in file.items():
                if key != 'model':
                    stats += f'{key}: {value:4f} '
            f = FunctionItem(stats, demo, kwargs={'dir': str(dir), 'env': env, 'n': 5})
            menu.append_item(f)

    menu.show()