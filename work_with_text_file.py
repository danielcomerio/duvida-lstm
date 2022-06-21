from pathlib import Path
from os.path import join
from typing import List


def get_files_name(base_path: Path, files_list: List[str], fileName_position: int = 0) -> List[List[str]]:
    filesName_list = []

    for index in range(len(files_list)):
        file = open(join(base_path, files_list[index]), 'r')
        line = file.readline()  # discard header line
        line = file.readline()

        filesName = []
        while line != '':
            # get file name
            file_name = str(line.split(';')[fileName_position])
            filesName.append(file_name)
            line = file.readline()
        file.close()
        filesName_list.append(filesName)

    return [lst for lst in filesName_list]
