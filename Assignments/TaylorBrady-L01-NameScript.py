#File: TaylorBrady-L01-NameScript.py
#Description: Prints name and date
#Author: Taylor Brady
#Date: 7/1/2020
#Other comments: cannot improt datetime.datetime, module not found error

#Description: Returns name
#Pre: none
#Post: Returns string containing my name
def my_name():
    return "Taylor Brady"


#Description: Returns current date and time
#Pre: none
#Post: Returns datetime object, can be printed using print()
def date_and_time():
    import datetime as dt
    return dt.datetime.now()

if __name__ == "__main__":
    print(my_name())
    print(date_and_time())
