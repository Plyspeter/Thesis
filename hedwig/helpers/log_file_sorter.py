import os
import re
from datetime import datetime

regex = r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}.\d{3}"

def string_to_time_obj(str):
    obj = datetime.strptime(str, "%d-%m-%Y %H:%M:%S.%f")

def main():

    f = open(f"{os.path.dirname(__file__)}/../../logs/log.log", 'r+')
    lines = f.readlines()
    grouped_lines = []
    group = lines[0]
    group_match = re.match(regex, lines[0]).group()
    
    for line in lines[1:]:
        match = re.match(regex, line)
        if match:
            grouped_lines.append((group, group_match))
            group = ""
            group_match = match.group()
        group += line
    grouped_lines.append((group + "\n", match.group()))
    
    grouped_lines.sort(key=lambda group: group[1])
    
    f.seek(0)
    f.writelines([group for (group, _) in grouped_lines])
    f.close()
        







if __name__ == "__main__":
    main()