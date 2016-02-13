

def add_ten_and_return_x(x):
	print x
	print "now adding 10"
	x += 10
	print "in add_ten_and_return_x_loop", x

	return x

def add_ten_and_dont_return_x():
	global x
	print x
	print "now adding 10"
	x += 10
	print "in add_ten_and_dont_return_x loop", x

	return

def add_twenty_and_dont_return_x():
	global x
	print x
	print "now adding 20"
	x += 20
	print "in add_twenty_and_dont_return_x loop", x
	add_ten_and_return_x(x)
	print x

	return 


if __name__ == "__main__":
	#global x
	x = 10
	
	print x
	x = add_ten_and_return_x(x)
	
	print "out of loop", x
	add_ten_and_dont_return_x()
	add_ten_and_dont_return_x()
	add_ten_and_dont_return_x()
	add_twenty_and_dont_return_x()
	add_ten_and_dont_return_x()

	print "out of loop", x
