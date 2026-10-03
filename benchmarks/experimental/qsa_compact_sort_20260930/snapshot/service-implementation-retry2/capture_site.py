"""Identify the formal FULL capture call, not nested compile/dummy captures."""
import ast
from pathlib import Path


def formal_call_line(path):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'CudaGraphManager')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'capture')
    sites = []
    for node in ast.walk(method):
        if isinstance(node, ast.With) and any(
                isinstance(item.context_expr, ast.Call)
                and ast.unparse(item.context_expr.func) == 'torch.cuda.graph'
                for item in node.items):
            for statement in node.body:
                if (isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Call)
                        and ast.unparse(statement.value.func) == 'forward_fn'):
                    sites.append(statement.lineno)
    assert len(sites) == 1, f'formal capture site mismatch {sites}'
    return sites[0]


def is_formal(frame, code, line, cuda_capturing):
    return frame.f_code is code and frame.f_lineno == line and cuda_capturing
