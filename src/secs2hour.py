def secs2hour (time_in_sec):
    """
    Translate a time given by seconds (counted from 00:00am) to a string, expressing the 24-hours time (in resolution of minutes).
    Examples:
    - secs2hour(60) = '0001'
    - sec2hour (3600) = '0100'
    - sec2hour (7.5*3600) = '0730   
    """
    
    time_in_sec = int(time_in_sec) #convert the given time to int, so that the answer won't contain ".", as done for float
    hour        = time_in_sec // 3600 # full hours
    hour_str    = '0' + str(hour) if (hour < 10) else str(hour)
    min         = (time_in_sec - 3600*hour) // 60 # full minutes
    min_str     = '0' + str(min) if (min < 10) else str(min)    
    return hour_str + min_str

