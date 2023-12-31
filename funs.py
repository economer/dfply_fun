import re
import pandas as pd
from dfply import *

# str_detect
@make_symbolic
def str_detect(column, pattern):
    return column.apply(lambda x: bool(re.search(pattern, x)))

# str_replace
@make_symbolic
def str_replace(column, pattern, replacement):
    return column.apply(lambda x: re.sub(pattern, replacement, x, count=1))

# str_replace_all
@make_symbolic
def str_replace_all(column, pattern, replacement):
    return column.apply(lambda x: re.sub(pattern, replacement, x))

# str_extract
@make_symbolic
def str_extract(column, pattern):
    return column.apply(lambda x: re.search(pattern, x).group(0) if re.search(pattern, x) else None)

# str_extract_all
@make_symbolic
def str_extract_all(column, pattern):
    return column.apply(lambda x: re.findall(pattern, x))

# str_count
@make_symbolic
def str_count(column, pattern):
    return column.apply(lambda x: len(re.findall(pattern, x)))

# str_trim
@make_symbolic
def str_trim(column):
    return column.apply(lambda x: x.strip())

# str_length
@make_symbolic
def str_length(column):
    return column.apply(len)

# str_c
@make_symbolic
def str_c(*columns, sep=''):
    return columns[0].apply(lambda x: sep.join(map(str, [x] + [col[x.name] for col in columns[1:]])))

# str_split
@make_symbolic
def str_split(column, pattern, n=-1):
    return column.apply(lambda x: re.split(pattern, x, maxsplit=n))

# str_sub
@make_symbolic
def str_sub(column, start, end=None):
    return column.apply(lambda x: x[start:end])

# str_pad
@make_symbolic
def str_pad(column, width, side='left', padchar=' '):
    if side == 'left':
        return column.apply(lambda x: x.rjust(width, padchar))
    elif side == 'right':
        return column.apply(lambda x: x.ljust(width, padchar))
    elif side == 'both':
        return column.apply(lambda x: x.center(width, padchar))

@make_symbolic
def pivot_longer(df, columns_to_longer, names_to="name", values_to="value"):
    return pd.melt(df, id_vars=[col for col in df.columns if col not in columns_to_longer], 
                   value_vars=columns_to_longer, var_name=names_to, value_name=values_to)

@make_symbolic
def pivot_wider(df, index, columns, values):
    return df.pivot(index=index, columns=columns, values=values)


@make_symbolic
def str_remove(column, pattern):
    """
    Remove the first occurrence of a pattern in each string of the column.
    
    Args:
    - column: Pandas Series (string column).
    - pattern: Regular expression pattern to remove.
    
    Returns:
    - Pandas Series with the pattern removed.
    """
    return column.apply(lambda x: re.sub(pattern, '', x, count=1))

@make_symbolic
def str_remove_all(column, pattern):
    """
    Remove all occurrences of a pattern in each string of the column.
    
    Args:
    - column: Pandas Series (string column).
    - pattern: Regular expression pattern to remove.
    
    Returns:
    - Pandas Series with all instances of the pattern removed.
    """
    return column.apply(lambda x: re.sub(pattern, '', x))

@make_symbolic
def str_to_title(column):
    """
    Convert each string in the column to title case.
    
    Args:
    - column: Pandas Series (string column).
    
    Returns:
    - Pandas Series with each string in title case.
    """
    return column.apply(lambda x: x.title())

@make_symbolic
def str_to_sentence(column):
    """
    Convert each string in the column to sentence case.
    
    Args:
    - column: Pandas Series (string column).
    
    Returns:
    - Pandas Series with each string in sentence case.
    """
    return column.apply(lambda x: x.capitalize())

@make_symbolic
def str_to_upper(column):
    """
    Convert each string in the column to upper case.
    
    Args:
    - column: Pandas Series (string column).
    
    Returns:
    - Pandas Series with each string in upper case.
    """
    return column.apply(lambda x: x.upper())

@make_symbolic
def case_when(*conditions):
    """
    Apply conditions to create a new column, similar to dplyr's case_when.
    """
    def f(df):
        result = pd.Series([None] * len(df), index=df.index)
        for condition, value in conditions:
            if callable(condition):
                mask = condition(df)
            elif condition is True:  # Default case
                mask = pd.Series([True] * len(df), index=df.index)
            else:
                mask = condition

            # Apply condition only if result is still None at that position
            result = result.mask(result.isna() & mask, value)
        return result
    return f


# Example usage
if __name__ == "__main__":
    df = pd.DataFrame({'text': [' apple ', 'banana', ' cherry', 'grape']})
    result = (df >>
              mutate(trimmed_text = str_trim(X.text),
                     contains_a = str_detect(X.text, 'a'),
                     replaced_text = str_replace(X.text, 'a', 'X'),
                     all_replaced_text = str_replace_all(X.text, 'a', 'X'),
                     first_extract = str_extract(X.text, '[aeiou]'),
                     all_extracts = str_extract_all(X.text, '[aeiou]'),
                     vowel_count = str_count(X.text, '[aeiou]'),
                     text_length = str_length(X.text),
                     combined_text = str_c(X.text, X.text, sep='_'),
                     split_text = str_split(X.text, 'a', n=2),
                     substring = str_sub(X.text, 1, 4),
                     padded_text = str_pad(X.text, 10, side='right', padchar='-')))
    
    print(result)
