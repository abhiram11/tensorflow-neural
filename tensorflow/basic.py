import tensorflow as tf 

x1 = tf.constant([5])
x2 = tf.constant([6])

res = tf.multiply(x1,x2) # tf.mul changed to tf.multiply

#print(res)

"""sess = tf.Session()
print(sess.run(res))
sess.close()
"""
with tf.Session() as sess:
	output = sess.run(res) # now no need to close session after the work is done
	print(output)
"""print(output) #will work
print(sess.run(res)) #wont work"""