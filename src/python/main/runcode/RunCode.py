import ast
from io import StringIO
import sys
import traceback
from .CodeResultTemplate import CodeResultTemplate
from .InterpreterError import InterpreterError


class RunCode:

    @staticmethod
    def parse_code(src: str):
        result = CodeResultTemplate()
        try:
            parse = ast.parse(src)
            result.set_result(result=parse)
            result.set_status(status=True)

        except Exception as ex:
            result.set_exception(exception=ex)
            result.set_status(status=False)

        return result

    @staticmethod
    def execute_code(src: str, parse=None, run_mode: str = None):
        result = CodeResultTemplate(run_mode=run_mode, code = src)
        tmp = sys.stdout
        pipeline_result = StringIO()
        sys.stdout = pipeline_result
        try:
            loc = {}
            if parse is None:
                parse = ast.parse(src)

            exec(compile(parse, filename="<ast>", mode="exec"), loc)
            sys.stdout = tmp
            result.set_status(status=True)
            result.set_result(pipeline_result.getvalue())
            return result

        except SyntaxError as err:
            sys.stdout = tmp
            error_class = err.__class__.__name__
            detail = err.args[0]
            line_number = err.lineno
            exception = err
            error_type = "Syntax"
        except Exception as err:
            sys.stdout = tmp
            error_class = err.__class__.__name__
            cl, exc, tb = sys.exc_info()
            line_number = traceback.extract_tb(tb)[-1][1]
            detail = err.args[0]
            exception = traceback.format_exc()
            error_type = f"Exception-{cl}"

        ie = InterpreterError("%s at line %d of %s: %s" % (error_class, line_number, "source string", detail))
        result.set_status(status=False)
        result.set_exception(exception=exception)
        result.set_result(result=ie)
        result.set_error_exception(exception)
        result.set_error_class(error_class)
        result.set_error_type(error_type)
        result.set_error_detail(detail)

        return result