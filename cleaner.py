import re
import subprocess

def comment_remover(code):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, code)

def format_code(code, style='google'):
    # Run clang-format with the specified style and capture the formatted code
    formatted_code = subprocess.run(
        ['clang-format', '-style=' + style], 
        input=code.encode('utf-8'), 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )

    # Return the formatted code as a string
    return formatted_code.stdout.decode('utf-8')

def code_preprocess(code):
    comment_free_code = comment_remover(code)
    formatted_code = format_code(comment_free_code)

    return formatted_code