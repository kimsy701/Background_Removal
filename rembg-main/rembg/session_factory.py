import os
from typing import Type

import onnxruntime as ort

from .sessions import sessions_class
from .sessions.base import BaseSession
#from .sessions.u2net import U2netSession
from .sessions.u2net_custom import U2netCustomSession


def new_session(
    #model_name: str = "u2net", providers=None, *args, **kwargs
    model_name: str = "u2net_custom", providers=None, *args, **kwargs
) -> BaseSession:
    #session_class: Type[BaseSession] = U2netSession
    session_class: Type[BaseSession] = U2netCustomSession

    for sc in sessions_class:
        if sc.name() == model_name:
            session_class = sc
            break
    print("now printint session options")
    sess_opts = ort.SessionOptions()
    print("sess_opts",sess_opts)

    #if "OMP_NUM_THREADS" in os.environ:
    #  print("OMP_NUM_THREADS",OMP_NUM_THREADS)
    #    sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

    return session_class(model_name, sess_opts, providers, *args, **kwargs)
