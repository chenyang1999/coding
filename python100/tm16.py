#题目：输出指定格式的日期。
#
#程序分析：使用 datetime 模块。
import datetime
 
if __name__ == '__main__':
 
	# 输出今日日期，格式为 dd/mm/yyyy。更多选项可以查看 strftime() 方法
	print(datetime.date.today().strftime('%d/%m/%Y'))
 
	# 创建日期对象
	miyazakiBirthDate = datetime.date(1941, 1, 5)
 
	print(miyazakiBirthDate.strftime('%d/%m/%Y'))
 
	# 日期算术运算
	miyazakiBirthNextDay = miyazakiBirthDate + datetime.timedelta(days=1)
 
	print(miyazakiBirthNextDay.strftime('%d/%m/%Y'))
 
	# 日期替换
	miyazakiFirstBirthday = miyazakiBirthDate.replace(year=miyazakiBirthDate.year + 1)
 
	print(miyazakiFirstBirthday.strftime('%d/%m/%Y'))