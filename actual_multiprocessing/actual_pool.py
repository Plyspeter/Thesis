import multiprocessing as mp
import hedwig

def _name_proc(parentName, name):
    proc = mp.current_process()
    if parentName == "MainProcess":
        i = proc.name.find('-') + 1
        preName = name
    else:
        i = proc.name.find(':') + 1
        preName = f"{parentName} || {name}"
    proc.name = f"{preName}-{proc.name[i:]}"
    
def _run(func, arg):
    try:
        return func(arg)
    except Exception as e:
        hedwig.exception(e)
        

class ActualPool:
    def __init__(self, num_of_procs, name) -> None:
        self.__num_of_procs = num_of_procs
        self.__name = name
        self.reset()
    
    def reset(self) -> None:
        self.__needs_reset = False
        self.__pool = mp.Pool(processes=self.__num_of_procs, initializer=_name_proc, initargs=[mp.current_process().name, self.__name])
    
    def run(self, func, args) -> list:
        assert not self.__needs_reset
        self.__needs_reset = True 
        result = list(self.__pool.starmap(_run, [(func, arg) for arg in args]))
        self.__pool.close() #Says that no more tasks can be given, but doesn't terminate current tasks
        self.__pool.join() #As for processes but close must be called first
        self.__pool.terminate()
        del self.__pool
        self.__pool = None
        return result