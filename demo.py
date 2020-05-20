from pathlib import Path
import torch
import os
from cursesmenu import CursesMenu
from cursesmenu.items import FunctionItem, SubmenuItem, CommandItem
from worldmodel import get_args, demo, defaults, make_env
import worldmodel
import curses
import shutil

KEY_DELETE = 330


class DuaneCursesMenu(CursesMenu):
    def __init__(self):
        super().__init__()

    def process_user_input(self):
        """
        Gets the next single character and decides what to do with it
        """
        user_input = self.get_input()

        go_to_max = ord("9") if len(self.items) >= 9 else ord(str(len(self.items)))

        if ord('1') <= user_input <= go_to_max:
            self.go_to(user_input - ord('0') - 1)
        elif user_input == curses.KEY_DOWN:
            self.go_down()
        elif user_input == curses.KEY_UP:
            self.go_up()
        elif user_input == ord("\n"):
            self.select()
        elif user_input == KEY_DELETE:
            dir = self.current_item.kwargs['dir']
            try:
                shutil.rmtree(dir)
            except:
                print('delete failed')

        return user_input


if __name__ == '__main__':

    args = get_args(defaults)
    worldmodel.args = args

    wandb_run_dirs = Path().glob(f'wandb/*')
    wandb_run_dirs = sorted(wandb_run_dirs, key=os.path.getmtime, reverse=True)

    env = make_env()

    menu = DuaneCursesMenu()

    for dir in wandb_run_dirs[:20]:
        policy_file = dir/'policy_best.pt'
        if policy_file.exists():
            file = torch.load(str(policy_file))
            stats = f'{dir} '
            for key, value in file.items():
                if key != 'model':
                    stats += f'{key}: {value:.2f} '
            f = FunctionItem(stats, demo, kwargs={'dir': str(dir), 'env': env, 'n': 5})
            menu.append_item(f)

    menu.show()