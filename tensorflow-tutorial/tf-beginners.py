# import tensorflow libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

-
'''
Declare 4 constant a,b,c and d which we will use as the input
to our computation graph.
Constant in tensorflow are immutable, so once assigned a particular
value to it, it can't be change within the course of the program
'''

a = tf.constant(6, name='constant_a')
b = tf.constant(3, name='constant_b')
c = tf.constant(10, name='constant_c')
d = tf.constant(5, name='constant_d')

'''
tf.constant is used to speficy consants.
Along with constant, name parameter allows us to identify these
constants when we visualize it using TensorBoard
'''

# a,b,c and d are the tensors flowing through the graph

mul = tf.multiply(a, b, name='mul')

# mul is a node which specify the multiply operation on `a` and `b`

div = tf.div(c, d, name='div')

# division operation which divides c by d and have the name 'div'

# tf.add_n will sum up the element in an array

addn = tf.add_n([mul, div], name='addn')

'''
specified 2 computational node into the array, multiplication and division.
So, the output of multiplication and division will be added

if we simply try to print the node, it will print:
    the name of the tensor, the shape, and the datatype

It because we haven't run the graph we built up above. To obtain value
after all the computations, we need to execute any tensorflow program
'''

session = tf.Session()
session.run(addn)

'''
Run `session.run(name of node)` on any intermediate steps will
get the result.
It will perform all the computation till the output of the node.
To visualize the graph using tensorflow browser tools, we need to
instantiate a filewritter which is in the tf.summary namespace which
allows us to write out events to a log file directly.
'''

writer = tf.summary.FileWriter('./exampleTensorflow1', session.graph)

# close the writer and session

writer.close()
session.close()