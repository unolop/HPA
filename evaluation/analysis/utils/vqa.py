import re

def extract_number_or_other(x):
    if not isinstance(x, str):
        return 'others'
    
    match = re.search(r'\d+', x)
    if match:
        return int(match.group())
    else:
        # print(f'cannot find {x}')
        return 'others' 
        
def yes_or_no(df):
    out = df['output'].astype(str).str.lower()

    df = df.copy()
    df['y/n'] = 'others'
    df.loc[out.str.contains('yes', na=False), 'y/n'] = 'yes'
    df.loc[out.str.contains('no', na=False), 'y/n'] = 'no'
    return df

def bin_number(x):
    if x == 'others':
        return 'others'
    if not isinstance(x, (int, float)):
        return 'others'

    if x == 0:
        return '0'
    elif x == 1:
        return '1'
    elif 2 <= x <= 3:
        return '2–3'
    elif 4 <= x <= 5:
        return '4–5'
    elif 6 <= x <= 10:
        return '6–10'
    elif 11 <= x <= 20:
        return '11–20'
    else:
        return '>20'
