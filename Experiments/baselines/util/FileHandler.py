def read_text_file_line_by_line(fname:str):
    try:
        with open(fname) as f:
            lines = f.readlines()
            raw = "".join(lines)
            return raw
    except Exception as ex:
        raise Exception (f"Error in reading file:\n {ex}")


def save_text_file(fname: str, data):
    try:
        f = open(fname, 'w')
        f.write(data)
        f.close()
    except Exception as ex:
        raise Exception (f"Error in save file:\n {ex}")