import math


def stopwatch(time_now, time_ini):
  time_native = time_now - time_ini
  time_minutes = int(time_native // 60)
  time_seconds = int(time_native % 60)
  time_milliseconds = int(1000*(time_native - math.floor(time_native)))
  return str(time_minutes).zfill(2) + ':' + str(time_seconds).zfill(2) + ':' + str(time_milliseconds).zfill(3)
