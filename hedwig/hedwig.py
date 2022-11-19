import logging
import logging.config
import json
import sys
import traceback

def init(config_path):
    add_logging_level("CRITICAL_INFO", 45)
    load_config(config_path)
    log_for_exceptions()

def load_config(config_path):
    f = open(config_path)
    config = json.load(f)
    f.close()
    logging.config.dictConfig(config)

def add_logging_level(level_name, level_num, method_name=None):
    if not method_name:
        method_name = level_name.lower()

    if hasattr(logging, level_name):
       raise AttributeError('{} already defined in logging module'.format(level_name))
    if hasattr(logging, method_name):
       raise AttributeError('{} already defined in logging module'.format(method_name))
    if hasattr(logging.getLoggerClass(), method_name):
       raise AttributeError('{} already defined in logger class'.format(method_name))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, logForLevel)
    setattr(logging, method_name, logToRoot)
    
def log_for_exceptions():
    def handle_exception(*exc_info):
        if issubclass(exc_info[0], KeyboardInterrupt):
            sys.__excepthook__(*exc_info)
            return
        
        formatted = "".join(traceback.format_exception(*exc_info))
        msg = f'Uncaught exception:\n{formatted}'
        
        logging.getLogger("exception_logger").critical(msg)
    sys.excepthook = handle_exception
    
def exception(exception: Exception):
    if issubclass(type(exception), KeyboardInterrupt):
            return
        
    formatted = traceback.format_exc()
    msg = f'Uncaught exception:\n{formatted}'
    logging.getLogger("exception_logger").critical(msg)
    
def __get_logger(name=None):
    if name is None:
        name = sys._getframe(2).f_globals.get('__name__')
    return logging.getLogger(name)

def debug(msg, name=None):
    logger = __get_logger(name)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(msg)
    
def info(msg, name=None):
    logger = __get_logger(name)
    if logger.isEnabledFor(logging.INFO):
        logger.info(msg)
    
def warning(msg, name=None):
    logger = __get_logger(name)
    if logger.isEnabledFor(logging.WARNING):
        logger.warning(msg)
    
def error(msg, name=None):
    logger = __get_logger(name)
    if logger.isEnabledFor(logging.ERROR):
        logger.error(msg)
    
def critical_info(msg, name=None):
    logger = __get_logger(name)
    if logger.isEnabledFor(logging.CRITICAL_INFO):
        logger.critical_info(msg)
    
def critical(msg, name=None):
    logger = __get_logger(name)
    if logger.isEnabledFor(logging.CRITICAL):
        logger.critical(msg)
