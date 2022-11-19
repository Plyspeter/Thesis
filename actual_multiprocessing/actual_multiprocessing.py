import multiprocessing as mp
import queue
from typing import Iterable

from torch import argmax
import hedwig

SENTINEL = None

def _run(input_q: mp.Queue, output_q: mp.Queue, func, items):
    while True:
        try:
            try:
                hedwig.debug("Trying to get from queue")
                arg = input_q.get(block=True, timeout=0.01)
            except queue.Empty:
                hedwig.debug("Queue was empty or blocked! Retrying...")
                continue
            hedwig.debug("Acquired object from queue. Processing object")
            
            if arg == SENTINEL:
                hedwig.debug("Sentinel found, exciting process")
                break
            item = items[arg]
            res = func(item)
            items[arg] = item
            output_q.put((arg, res))
            hedwig.debug("Object processed")
        except Exception as ex:
            hedwig.exception(ex)

class ActualMultiprocessing:
    def __init__(self, num_of_procs: int, name: str, func, args: Iterable) -> None:
        self.__num_of_procs = num_of_procs
        self.__name = name
        self.__func = func
        self.reset(args)
    
    def reset(self, args) -> None:
        self.__num_of_args = len(args)
        self.__input_q = mp.Queue()
        self.__output_q = mp.Queue()
        self.__procs = []
        self.__needs_reset = False
        
        self.__manager = mp.Manager()
        
        for arg in list(range(len(args))) + ([SENTINEL] * self.__num_of_procs):
            self.__input_q.put(arg)
            
        self.__managedArgs = self.__manager.list(args)
        
        for i in range(self.__num_of_procs):
            p = mp.Process(name=f'{self.__name}-{i:02}', target=_run, kwargs={"input_q": self.__input_q, "output_q": self.__output_q, "func": self.__func, "items": self.__managedArgs})
            self.__procs.append(p)
                
    def run(self) -> list:
        assert not self.__needs_reset
        self.__needs_reset = True
        for proc in self.__procs:
            proc.start()
            hedwig.debug(f"Process {proc.name} started")
        
        results = []
        num_of_results = 0
        hedwig.debug("Getting results!")
        while num_of_results < self.__num_of_args:
            res = self.__output_q.get()
            results.append((res[1], self.__managedArgs[res[0]]))
            num_of_results += 1
            hedwig.debug(f"Result returned! {num_of_results}/{self.__num_of_args}")
        
        for proc in self.__procs:
            proc.join()
            hedwig.debug(f"Process {proc.name} joined")
            proc.terminate()
            hedwig.debug(f"Process {proc.name} closed")

        return results