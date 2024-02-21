import ast
from io import StringIO
import sys
import traceback
from .CodeResultTemplate import CodeResultTemplate
from .InterpreterError import InterpreterError


class RunCode(object):
    def __init__(self):
        pass

    def parse_code(self, src: str):
        result = CodeResultTemplate()
        try:
            parse = ast.parse(src)
            result.set_result(result=parse)
            result.set_status(status=True)

        except Exception as ex:
            result.set_exception(exception=ex)
            result.set_status(status=False)

        return result

    def execute_code(self, src: str, parse=None):
        result = CodeResultTemplate()
        try:
            loc = {}
            if parse is None:
                parse = ast.parse(src)
            tmp = sys.stdout
            pipeline_result = StringIO()
            sys.stdout = pipeline_result

            exec(compile(parse, filename="<ast>", mode="exec"), loc)
            sys.stdout = tmp
            result.set_status(status=True)
            result.set_result(pipeline_result.getvalue())

            return result

        except SyntaxError as err:
            error_class = err.__class__.__name__
            detail = err.args[0]
            line_number = err.lineno
            exception = err
        except Exception as err:
            error_class = err.__class__.__name__
            detail = err.args[0]
            cl, exc, tb = sys.exc_info()
            line_number = traceback.extract_tb(tb)[-1][1]
            exception = err

        ie = InterpreterError("%s at line %d of %s: %s" % (error_class, line_number, "source string", detail))
        result.set_status(status=False)
        result.set_exception(exception=exception)
        result.set_result(result=ie)

        return result
