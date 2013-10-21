"""Implement a timer that can be used to easily record time."""

import time

class Timer(object):
    """Timer implements some sugar functions that works like a stopwatch.
    
    timer.reset() resets the watch
    timer.lap()   returns the time elapsed since the last lap() call
    timer.total() returns the total time elapsed since the last reset
    """

    def __init__(self, template = '{0}h{1}m{2:.2f}s'):
        """Initializes a timer.
        
        Input: 
            template: (optional) a template string that can be used to format
                the timer. Inside the string, use {0} to denote the hour, {1}
                to denote the minute, and {2} to denote the seconds. Default
                '{0}h{1}m{2:.2f}s'
        """
        self._total = time.time()
        self._lap = time.time()
        if template:
            self._template = template
    
    def _format(self, timeval):
        """format the time value according to the template
        """
        hour = int(timeval / 3600.0)
        timeval = timeval % 3600.0
        minute = int (timeval / 60)
        timeval = timeval % 60
        return self._template.format(hour, minute, timeval)

    def reset(self):
        """Press the reset button on the timer
        """
        self._total = time.time()
        self._lap = time.time()
        
    def lap(self, use_template = True):
        """Report the elapsed time of the current lap, and start counting the
        next lap.
        Input:
            use_template: (optional) if True, returns the time as a formatted
                string. Otherwise, return the real-valued time. Default True.
        """
        diff = time.time() - self._lap
        self._lap = time.time()
        if use_template:
            return self._format(diff)
        else:
            return diff
    
    def total(self, use_template = True):
        """Report the total elapsed time of the timer.
        Input:
            use_template: (optional) if True, returns the time as a formatted
                string. Otherwise, return the real-valued time. Default True.
        """
        if use_template:
            return self._format(time.time() - self._total)
        else:
            return time.time() - self._total
