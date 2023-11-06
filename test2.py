# this is a simple example
import logging
# define the log file, file mode and logging level
logging.basicConfig(filename='example.log', filemode="w", level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')