import ast

with open("/home/saeed/Downloads/tmp/gemini-1.5-pro-latest-CatDB-Random-0-SHOT-No-iteration-1-RUN.py", "r") as file:
    tree = ast.parse(file.read())

for node in ast.walk(tree):

    # if isinstance(node, ast.Ex):
    #     print("**************************************************")
    #     print("Import found:", node.value)
    #     print("**************************************************")


    print(ast.dump(node))
    print(f"{node.type_comment} +++++++++++++++++++++++++++++++++")
    #
    # if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
    #     print("Import found:", ast.dump(node))
    # elif isinstance(node, ast.FunctionDef):
    #     print("Function definition:", node.name)