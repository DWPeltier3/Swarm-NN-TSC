## Used to display time taken for script to run
# input: start time

from timeit import default_timer as timer
import math

def elapse_time(start):
    end = timer()
    elapse=end-start
    hours=0
    minutes=0
    seconds=0
    remainder=0
    if elapse>3600:
        hours=math.trunc(elapse/3600)
        remainder=elapse%3600
    if elapse>60:
        if remainder>60:
            minutes=math.trunc(remainder/60)
            seconds=remainder%60
            seconds=math.trunc(seconds)
        else:
            minutes=math.trunc(elapse/60)
            seconds=elapse%60
            seconds=math.trunc(seconds)
    if elapse<60:
        seconds=math.trunc(elapse)

    print(f"\nElapse Time: {hours}h, {minutes}m, {seconds}s\n")