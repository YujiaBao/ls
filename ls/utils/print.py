'''
Add datetime for each print

Yujia
2022-07-28

'''
from datetime import datetime

from rich import print

rich_print = print

def print(*args, **kw):
    print_time = False

    if 'time' in kw:
        if kw['time'] == True:
            print_time = True

        del kw['time']

    if print_time:
        cur_time = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        rich_print(f'[not bold][default]{cur_time}[/default][/not bold]',
                   end=' ')

    rich_print(*args, **kw)
