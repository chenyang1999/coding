'''
题目：请输入星期几的第一个字母来判断一下是星期几，如果第一个字母一样，则继续判断第二个字母。
'''
letter = input("please input:")
#while letter  != 'Y':
if letter == 'S':
	print ('please input second letter:')
	letter = input("please input:")
	if letter == 'a':
		print ('Saturday')
	elif letter  == 'u':
		print ('Sunday')
	else:
		print ('data error')
	
elif letter == 'F':
	print ('Friday')
	
elif letter == 'M':
	print ('Monday')
	
elif letter == 'T':
	print ('please input second letter')
	letter = raw_input("please input:")
 
	if letter  == 'u':
		print ('Tuesday')
	elif letter  == 'h':
		print ('Thursday')
	else:
		print ('data error')
		
elif letter == 'W':
	print ('Wednesday')
else:
	print ('data error')