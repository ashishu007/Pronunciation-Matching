import tensorflow as tf
gf = tf.GraphDef()
gf.ParseFromString(open('retrained_graph.pb','rb').read())
for n in gf.node:
    print(n.name + '=>' + n.op )
